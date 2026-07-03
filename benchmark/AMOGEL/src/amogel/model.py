# build simple GCN model for graph classification 
from torch_geometric.loader import DataLoader   
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy , Precision , Recall , AUROC , ConfusionMatrix , F1Score , Specificity
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , BatchNorm , GATConv 
from torch_geometric.nn import global_mean_pool , SAGPooling , TopKPooling
from torch_geometric.utils import to_dense_batch
import mlflow
from sklearn.metrics import classification_report , roc_auc_score
import numpy as np
import argparse 
import os 
import yaml
from pytorch_lightning import Trainer
from dotenv import load_dotenv
import torch.nn as nn

class GCN(pl.LightningModule):
    def __init__(self, 
        in_channels ,  hidden_channels , num_classes , no_of_nodes ,  lr=0.0001 , drop_out=0.0, weight=None, pooling_ratio=0 ,
        mlflow:mlflow = None , decay=0.0 , GNN=True , DNN=True ):
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(GCN, self).__init__()
        self.DNN = DNN
        self.GNN = GNN
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.decay = decay
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin_hidden = Linear(hidden_channels, hidden_channels)
        self.pooling = SAGPooling(hidden_channels, ratio=self.pooling_ratio)
        self.lin = Linear(hidden_channels, num_classes)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 if GNN and DNN else hidden_channels, hidden_channels), # combined feed x and graph x 
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Dropout1d(drop_out),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        self.weight = weight if weight is None else torch.tensor(weight, device=device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.mlflow = mlflow
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(no_of_nodes , 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024 , 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(512 , hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Dropout(drop_out),
        )

        self.acc_train = Accuracy(task="multiclass", num_classes=num_classes)
        self.acc_test = Accuracy(task='multiclass' , num_classes=num_classes)
        self.cfm_training = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.cfm_testing = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        self.actual = []
        self.predicted = []
        self.predicted_proba = []
        self.lr = lr
        self.drop_out = drop_out
        self.edge_attn_l1 = []
        self.edge_attn_l2 = []
        self.batches = []
        self.pool_batches = []
        self.pool_perm = []
        self.pool_score = []
        self.embeddings = []

    def forward(self, x, edge_index, edge_attr, batch):
        
        # convert node embeddign to dense batch 
        batch_x , batch_mask = to_dense_batch(x , batch) # dimension ( no_batch , number_of_node )
        batch_x = batch_x.view(batch_x.shape[0] , batch_x.shape[1] * batch_x.shape[2]) 
        
        feed_x  = self.feedforward(batch_x) # [ batch_size  hidden_channels ]
        
        # 1. Obtain node embeddings 
        x , edge_attn_l1 = self.conv1(x, edge_index , edge_attr , return_attention_weights=True)
        x = self.bn1(x)
        x = x.relu()
        x , edge_attn_l2 = self.conv2(x, edge_index , edge_attr , return_attention_weights=True)
        x = self.bn2(x)
        x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        if self.pooling_ratio > 0:
            x , edge_index , edge_attr , batch , perm , score = self.pooling(x , edge_index , edge_attr , batch)
        else:
            perm = None 
            score = None
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        concat_layer = []
        if self.GNN:
            concat_layer.append(x)
        if self.DNN: 
            concat_layer.append(feed_x)
        global_x = torch.concat(concat_layer , dim=1)
        
        # 3. Apply a final classifier
        # x = F.dropout(x, p=self.drop_out , training=self.training)
        # x = self.lin_hidden(x).relu()
        # x = self.lin(x)
        x = self.mlp(global_x)
        
        return x , edge_attn_l1 , edge_attn_l2 , batch , perm , score

    def training_step(self, batch, batch_idx):
        x , edge_index , edge_attr, batch , y = batch.x , batch.edge_index , batch.edge_attr , batch.batch , batch.y
        out , edge_attn_l1 , edge_attn_l2 , _ , _ , _ = self(x, edge_index, edge_attr, batch)
        loss = self.criterion(out, y)
        acc = self.acc_train(out, y)
        self.cfm_training(out , y)
        
        self.log('train_loss' , loss , prog_bar=False, on_epoch=True , on_step=False)
        self.log('train_acc' , acc , prog_bar=False , on_epoch=True , on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x , edge_index , edge_attr, batch , y = batch.x , batch.edge_index , batch.edge_attr , batch.batch , batch.y
        self.batches.append(batch)
        out , edge_attn_l1, edge_attn_l2 , pool_batch , pool_perm , pool_score = self(x, edge_index, edge_attr, batch)
        
        self.edge_attn_l1.append(edge_attn_l1)
        self.edge_attn_l2.append(edge_attn_l2)
        self.pool_batches.append(pool_batch)
        self.pool_perm.append(pool_perm)
        self.pool_score.append(pool_score)
        
        loss = self.criterion(out, y)
        acc = self.acc_test(out, y)
        self.cfm_testing(out, y)  
        self.actual.extend(y.cpu().numpy())
        self.predicted.extend(out.argmax(dim=1).cpu().numpy())
        self.predicted_proba.extend(F.softmax(out , dim=-1).cpu().numpy())  
        
        self.log('val_loss' , loss , prog_bar=True, on_epoch=True)
        self.log('val_acc' , acc , prog_bar=True, on_epoch=True)
    
    def on_train_epoch_end(self) -> None:
        
        # if self.current_epoch == self.trainer.max_epochs - 1:
        #     cfm = self.cfm_training.compute().cpu().numpy()
        #     print("")
        #     print("-------- Confusion Matrix [Training] --------")
        #     print(cfm)
        self.edge_attn_l2 = []
        self.edge_attn_l1 = []
        self.batches = []
        self.cfm_training.reset()
        
    def on_validation_epoch_end(self):
            
        self.edge_attn_l2 = []
        self.edge_attn_l1 = []
        self.batches = []
        self.pool_batches = []
        self.pool_perm = []
        self.pool_score = []
        self.actual = []
        self.predicted = []
        self.predicted_proba = []
        self.cfm_testing.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr , weight_decay=self.decay)
    
class MLFlowCallback(pl.callbacks.Callback):
    def __init__(self, mlflow:mlflow, save_dir="./" , save_amogel_edge=True): 
        super().__init__()
        self.save_dir = save_dir
        self.mlflow = mlflow
        self.save_amogel_edge = save_amogel_edge
        os.makedirs(save_dir  , exist_ok=True)
        
    def on_train_epoch_end(self , trainer , pl_module):
        pass 
    
    def on_validation_epoch_end(self , trainer , pl_module): 
        if (trainer.current_epoch+1) % 10 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            report = classification_report(trainer.model.actual , trainer.model.predicted , digits=4)
            auc_weighted = roc_auc_score(trainer.model.actual , trainer.model.predicted_proba , multi_class="ovr" , average="weighted")
            auc_macro = roc_auc_score(trainer.model.actual , trainer.model.predicted_proba , multi_class="ovr" , average="macro")
            report += f"roc_auc_score weighted: {auc_weighted:.4f} \n"
            report += f"roc_auc_score macro: {auc_macro:.4f}"
            
            self.mlflow.log_text(report , f"calssification_report_val_{(trainer.current_epoch+1):04d}.txt")
            
            if trainer.current_epoch == trainer.max_epochs - 1:
                print(report)
                if self.save_amogel_edge:
                    torch.save(model.edge_attn_l1 , os.path.join(self.save_dir, f"edge_attn_l1_{trainer.current_epoch+1}.pt"))
                    torch.save(model.edge_attn_l2 , os.path.join(self.save_dir, f"edge_attn_l2_{trainer.current_epoch+1}.pt"))
                    torch.save(model.batches , os.path.join(self.save_dir, f"batches_{trainer.current_epoch+1}.pt"))
                    torch.save(model.pool_batches , os.path.join(self.save_dir, f"pool_batches_{trainer.current_epoch+1}.pt"))
                    torch.save(model.pool_perm , os.path.join(self.save_dir, f"pool_perm_{trainer.current_epoch+1}.pt"))
                    torch.save(model.pool_score , os.path.join(self.save_dir, f"pool_score_{trainer.current_epoch+1}.pt"))
                
    def on_fit_end(self, trainer , pl_module):
        pass 

if __name__ == "__main__": 
    load_dotenv()
    
    parser = argparse.ArgumentParser("Amogel Model Training")
    parser.add_argument("--dataset" , type=str , required=True , help="Dataset name")
    
    args = parser.parse_args()
    
    params = yaml.safe_load(open("params.yaml"))
    
    train_graphs = torch.load(f"./output/graph/{args.dataset}/train.pt") 
    test_graphs = torch.load(f"./output/graph/{args.dataset}/test.pt") 
    
    train_loader = DataLoader(train_graphs , batch_size=params['model']['batch_size'] , shuffle=True) 
    test_loader = DataLoader(test_graphs , batch_size=params['model']['batch_size'] , shuffle=False) 
    
    model = GCN(
        in_channels=train_graphs[0].x.shape[1],
        hidden_channels=params['model']['hidden_unit'],
        num_classes= 5 if args.dataset == "BRCA" else 3, # to be updated
        lr=params['model']['learning_rate'],
        drop_out=params['model']['dropout'], 
        pooling_ratio=0.0, # to be removed
        decay=params['model']['weight_decay'], 
        no_of_nodes=train_graphs[0].x.shape[0] * train_graphs[0].x.shape[1], 
        DNN=True, 
        GNN=True 
    )
    
    output_dir = f"./output/model/{args.dataset}"
    mlflow_callback = MLFlowCallback(mlflow=mlflow , save_dir=output_dir)
    
    mlflow.pytorch.autolog()
    mlflow.set_experiment(f"amogel_{args.dataset}")
    
    with mlflow.start_run() as run:
        trainer = Trainer(max_epochs=params['model']['epochs'] , logger=[] , enable_checkpointing=False , callbacks=[ mlflow_callback ])
        trainer.fit(model , train_loader , test_loader)
        
        mlflow.log_artifacts(output_dir , artifact_path="model")
        mlflow.log_params(params['data_preprocessing'][args.dataset])
        mlflow.log_params(params['arm'][args.dataset])
        mlflow.log_params(params['graph'][args.dataset])
        mlflow.log_params(params['model'])
        