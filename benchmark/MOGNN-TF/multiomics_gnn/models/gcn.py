import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import math

class GCN(torch.nn.Module):
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

        super(GCN, self).__init__()
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
    
        if method == 'gcn':
            self.conv1 = ChebConv(self.raised_dimension, self.hid, K=5)
            self.conv2 = ChebConv(self.hid, self.hid, K=5)

        if self.poolsize <= 1:
            if method == 'gcn':
                self.linear_input = (self.num_gene + self.num_mirna) * self.hid
        else:
            if method == 'gcn':
                self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:
            #change
            parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            #print(f"[DEBUG POOL] Input shape: {x.shape}")
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            #print(f"[DEBUG POOL] Output shape: {x.shape}")
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]
        #print("[GCN] Input x shape:", x.shape)

        # Pre-convolution linear transforms
        if self.num_mirna == 0 or self.num_features == 1:
            #print("[GCN] Before pre_conv_linear_gene, x shape:", x.shape)
            x = self.pre_conv_linear_gene(x)
            #print("[GCN] After pre_conv_linear_gene, x shape:", x.shape)

            #print("[GCN] Before ReLU (pre_conv_linear_gene)")
            x = F.relu(x)
            #print("[GCN] After ReLU, x shape:", x.shape)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]
            #print("[GCN] x_exp_mirna shape:", x_exp_mirna.shape)
            #print("[GCN] x_cnv shape:", x_cnv.shape)

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]
            #print(f"[DEBUG FORWARD] x_cnv after slicing shape: {x_cnv.shape}")
            #print(f"[DEBUG FORWARD] x_exp after slicing shape: {x_exp.shape}")

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            #print("[GCN] x_gene after concatenation shape:", x_gene.shape)
            x_gene = x_gene.view(-1,self.num_features)
            #print("[GCN] x_gene after view shape:", x_gene.shape)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            #print("[GCN] x_mirna after flatten shape:", x_mirna.shape)
            x_mirna = x_mirna.view(-1, 1)

            #print("[GCN] Before pre_conv_linear_gene (gene branch), x_gene shape:", x_gene.shape)
            x_gene = self.pre_conv_linear_gene(x_gene)
            #print("[GCN] After pre_conv_linear_gene (gene branch), x_gene shape:", x_gene.shape)
            x_gene = F.relu(x_gene)
            #print("[GCN] After ReLU (gene branch), x_gene shape:", x_gene.shape)

            #print("[GCN] Before pre_conv_linear_mirna, x_mirna shape:", x_mirna.shape)
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            #print("[GCN] After pre_conv_linear_mirna, x_mirna shape:", x_mirna.shape)
            x_mirna = F.relu(x_mirna)
            #print("[GCN] After ReLU (mirna), x_mirna shape:", x_mirna.shape)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            #print("[GCN] x_gene reshaped for conv, shape:", x_gene.shape)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)
            #print("[GCN] x_mirna reshaped for conv, shape:", x_mirna.shape)

            x = torch.cat([x_gene,x_mirna],dim=1)
            #print("[GCN] After concatenating gene and mirna branches, x shape:", x.shape)

        #print("[GCN] x shape before conv layers:", x.shape)

        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)
        #print(f"DEBUG: x.size(0)={x.size(0)}, edge_index.max()={edge_index.max().item()}")
        # First convolution
        #print("[GCN] Before conv1, x shape:", x.shape)
        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)
            #print("[GCN] After conv1 (with edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv1), x shape:", x.shape)
        else:
            x = self.conv1(x, edge_index)
            #print("[GCN] After conv1 (no edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv1), x shape:", x.shape)

        # Second convolution
        #print("[GCN] Before conv2, x shape:", x.shape)
        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)
            #print("[GCN] After conv2 (with edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv2), x shape:", x.shape)
        else:
            x = self.conv2(x, edge_index)
            #print("[GCN] After conv2 (no edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv2), x shape:", x.shape)

        #print("[GCN] x shape before pooling:", x.shape)

        ## pooling on the graph to reduce nodes
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                    ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]
        #print("[GCN] x shape after pooling:", x.shape)
        if self.method == 'gcn':
            x = x.view(-1, self.hid) ## output shape:[batches * floor(num_node / poolsize), hid]
            #print("[GCN] After view for gcn method, x shape:", x.shape)

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        #print("[GCN] Before linear1, x shape:", x.shape)
        x = self.linear1(x)
        #print("[GCN] After linear1, x shape:", x.shape)
        x = F.relu(x)
        #print("[GCN] After ReLU (linear1), x shape:", x.shape)
        #print("[GCN] Before linear2, x shape:", x.shape)
        x = self.linear2(x)
        #print("[GCN] After linear2, x shape:", x.shape)
        x = F.relu(x)
        #print("[GCN] After ReLU (linear2), x shape:", x.shape)

        if self.decoder:
            x_reconstruct = x
            #print("[GCN] Before decoder_1, x_reconstruct shape:", x_reconstruct.shape)
            x_reconstruct = self.decoder_1(x_reconstruct)
            #print("[GCN] After decoder_1, x_reconstruct shape:", x_reconstruct.shape)
            x_reconstruct = F.relu(x_reconstruct)
            #print("[GCN] After ReLU (decoder_1), x_reconstruct shape:", x_reconstruct.shape)

            x_reconstruct  = nn.Dropout(0.2)(x_reconstruct)
            #print("[GCN] After Dropout (decoder_1), x_reconstruct shape:", x_reconstruct.shape)
            x_reconstruct = self.decoder_2(x_reconstruct)
            #print("[GCN] After decoder_2, x_reconstruct shape:", x_reconstruct.shape)

        if self.parallel:
            ## the two layer shallow FC network
            #print("[GCN] Before parallel_linear1, x_parallel shape:", x_parallel.shape)
            x_parallel = self.parallel_linear1(x_parallel)
            #print("[GCN] After parallel_linear1, x_parallel shape:", x_parallel.shape)
            #x_parallel = F.relu(x_parallel)
            x_parallel = F.leaky_relu(x_parallel, 0.1)
            #print("[GCN] After ReLU (parallel_linear1), x_parallel shape:", x_parallel.shape)

            #print("[GCN] Before parallel_linear2, x_parallel shape:", x_parallel.shape)
            x_parallel = self.parallel_linear2(x_parallel)
            ##print("[GCN] After parallel_linear2, x_parallel shape:", x_parallel.shape)
            #x_parallel = F.relu(x_parallel)
            x_parallel = F.leaky_relu(x_parallel, 0.1)
            #print("[GCN] After ReLU (parallel_linear2), x_parallel shape:", x_parallel.shape)

            x = torch.cat((x,x_parallel),1)
            #print("[GCN] After concatenating parallel branch, x shape:", x.shape)
        #print("[GCN] Before Dropout, x shape:", x.shape)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        #print("[GCN] After Dropout, x shape:", x.shape)
        #print("[GCN] Before classifier, x shape:", x.shape)
        x = self.classifier(x)
        #print("[GCN] After classifier, x shape:", x.shape)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0
        # Da controllare il fatto log_softmax + CrossEntropyLoss
        #loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss2 = nn.NLLLoss()(y, y_target)
        loss = 1*loss1 + 1*loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.2* l2_regularization* l2_loss
        return loss