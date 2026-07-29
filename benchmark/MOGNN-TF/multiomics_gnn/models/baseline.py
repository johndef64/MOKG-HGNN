import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(Baseline, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

        self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//2)
        self.parallel_linear2 = nn.Linear(parallel_input//2, parallel_input//4)
        self.parallel_linear3 = nn.Linear(parallel_input//4, self.concate_layer)
        self.classifier = nn.Linear(self.concate_layer, num_classes)
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward_old(self, x, edge_index, edge_weight):

        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        x_parallel = x_parallel.view(batches,-1)
        
        x_parallel = self.parallel_linear1(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear2(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear3(x_parallel)
        x_parallel = F.relu(x_parallel)

        x_parallel = F.dropout(x_parallel, p=self.dropout_rate, training=self.training)
        x_parallel = self.classifier(x_parallel)
        return x_parallel
    
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]

        # Caso: solo gene oppure solo 1 feature
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)

        else:
            # x: [B, N, 2] con N = num_gene + num_mirna
            g = self.num_gene
            m = self.num_mirna

            # GENI: prendo entrambe le features (exp, cnv) correttamente per gene
            x_gene = x[:, :g, :]                 # [B, g, 2]
            x_gene = x_gene.reshape(-1, 2)       # [B*g, 2]
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)
            x_gene = x_gene.view(batches, g, self.raised_dimension)

            # miRNA: solo expression (feature 0). CNV è padding.
            x_mirna = x[:, g:g+m, 0:1]           # [B, m, 1]
            x_mirna = x_mirna.reshape(-1, 1)     # [B*m, 1]
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)
            x_mirna = x_mirna.view(batches, m, self.raised_dimension)

            # Ricompongo nodi: [genes | miRNA]
            x = torch.cat([x_gene, x_mirna], dim=1)

        # Classifier MLP
        x_parallel = x.view(batches, -1)
        x_parallel = F.relu(self.parallel_linear1(x_parallel))
        x_parallel = F.relu(self.parallel_linear2(x_parallel))
        x_parallel = F.relu(self.parallel_linear3(x_parallel))

        x_parallel = F.dropout(x_parallel, p=self.dropout_rate, training=self.training)
        x_parallel = self.classifier(x_parallel)
        return x_parallel

    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization, class_weights=None):
        if class_weights is not None:
            loss2 = nn.CrossEntropyLoss(weight=class_weights)(y, y_target)
        else:
            loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss = 1*loss2
        
        return loss