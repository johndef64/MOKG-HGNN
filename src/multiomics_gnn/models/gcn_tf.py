import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, JumpingKnowledge
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
                    num_tf, 
                    omic_mode, 
                    num_classes,
                    jumping_knowledge,
                    jk_mode, 
                    dropout_rate):

        super(GCN, self).__init__()
        self.debug = False  # Enable to diagnose edge_index batching issues
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
        # --- JK params (NEW) ---
        self.jumping_knowledge = bool(jumping_knowledge)
        self.jk_mode = jk_mode

        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_tf = num_tf
        self.total_nodes = self.num_gene + self.num_mirna + self.num_tf
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
        if self.num_tf > 0:
            self.pre_conv_linear_tf = nn.Linear(self.num_features, self.raised_dimension)
    
        if method == 'gcn_tf':
            self.conv1 = ChebConv(self.raised_dimension, self.hid, K=5)
            self.conv2 = ChebConv(self.hid, self.hid, K=5)

        # jumping knowledge layer
        base_dim = self.hid
        
        if self.jumping_knowledge:
            if self.jk_mode == "cat":
                self.jk = JumpingKnowledge(mode="cat")
                self.jk_out = 2 * base_dim  # conv1 + conv2
            elif self.jk_mode == "max":
                self.jk = JumpingKnowledge(mode="max")
                self.jk_out = base_dim
            elif self.jk_mode == "lstm":
                self.jk = JumpingKnowledge(mode="lstm", channels=base_dim, num_layers=2)
                self.jk_out = base_dim
            else:
                raise ValueError(f"jk_mode must be one of: cat|max|lstm (got {self.jk_mode})")
        else:
            self.jk = None
            self.jk_out = base_dim

        if self.poolsize <= 1:
            if method == 'gcn_tf':
                self.linear_input = self.total_nodes * self.jk_out
        else:
            if method == 'gcn_tf':
                self.linear_input = math.floor(self.total_nodes / self.poolsize) * self.jk_out

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna+self.num_tf)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna+self.num_tf*self.num_features)


        if self.parallel:

            parallel_input = self.raised_dimension*(self.total_nodes)

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
            batch_index += [i]*self.total_nodes
        return(torch.Tensor(batch_index).type(torch.int64))
    
    def create_batch_index_2(self, batches, num_node):
        # restituisce un vettore lungo batches*num_node: [0..0, 1..1, ..., B-1..B-1]
        batch_index = []
        for i in range(batches):
            batch_index += [i] * num_node
        return torch.tensor(batch_index, dtype=torch.int64)
    
    def _log(self, msg):
        if self.debug:
            print(msg)
        
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

            G = self.num_gene
            M = self.num_mirna
            T = self.num_tf

            # Adesso il primo canale è fatto da Expression gene , mirna, Expression TF
            # Il secondo canale è fatto da CNV gene , padding, CNV TF
            x_exp = x[:, :, 0]
            x_cnv = x[:, :, 1]
            # slicing dei blocchi di dati
            gene_exp = x_exp[:, :G]
            gene_cnv = x_cnv[:, :G]
            mirna_exp = x_exp[:, G:G+M]
            tf_exp = x_exp[:, G+M:G+M+T]
            tf_cnv = x_cnv[:, G+M:G+M+T]

            x_exp = x[:, :, 0]
            x_cnv = x[:, :, 1]

            # slicing dei blocchi di dati
            gene_exp = x_exp[:, :G]
            gene_cnv = x_cnv[:, :G]
            mirna_exp = x_exp[:, G:G+M]
            tf_exp = x_exp[:, G+M:G+M+T]
            tf_cnv = x_cnv[:, G+M:G+M+T]
            # concatenation per ricostruire i nodi con entrambi i canali
            x_gene = torch.stack([gene_exp, gene_cnv], dim=2)   # (B, G, 2)
            x_tf   = torch.stack([tf_exp, tf_cnv], dim=2)         # concatenation finale di gene + tf (B, T, 2)

            # flatten per il linear layer
            x_gene = x_gene.reshape(-1, self.num_features)         # (B*G, 2)
            x_tf   = x_tf.reshape(-1, self.num_features)           # (B*T, 2)
            x_mirna = mirna_exp.reshape(-1, 1)                     # (B*M, 1)
            
            # Rise dimension layer
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)
            if self.num_tf > 0:
                x_tf = self.pre_conv_linear_tf(x_tf)
                x_tf = F.relu(x_tf)
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            # reshape back to (B, N, raised_dimension)
            x_gene = x_gene.reshape(batches, -1, self.raised_dimension)
            x_tf = x_tf.reshape(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.reshape(batches, -1, self.raised_dimension)
            # concatenate all omics
            x = torch.cat([x_gene, x_mirna, x_tf], dim=1)
            num_node = x.shape[1]


            ## the second matrix cnv_data has padding
            #x_exp_mirna = x[:,:,0]
            #x_cnv = x[:,:,1]
            #print("[GCN] x_exp_mirna shape:", x_exp_mirna.shape)
            #print("[GCN] x_cnv shape:", x_cnv.shape)

            ## separate mirna from the rest
            #x_cnv = x_cnv[:,:-100]
            #x_exp = x_exp_mirna[:,:-100]
            #print(f"[DEBUG FORWARD] x_cnv after slicing shape: {x_cnv.shape}")
            #print(f"[DEBUG FORWARD] x_exp after slicing shape: {x_exp.shape}")

            #x_cnv = x_cnv.view(batches,-1,1)
            #x_exp = x_exp.view(batches,-1,1)
            #x_gene = torch.cat([x_exp,x_cnv],dim=1)
            #print("[GCN] x_gene after concatenation shape:", x_gene.shape)
            #x_gene = x_gene.view(-1,self.num_features)
            #print("[GCN] x_gene after view shape:", x_gene.shape)
            #x_mirna = x_exp_mirna[:,-100:]
            #x_mirna = torch.flatten(x_mirna)
            #print("[GCN] x_mirna after flatten shape:", x_mirna.shape)
            #x_mirna = x_mirna.view(-1, 1)

            #print("[GCN] Before pre_conv_linear_gene (gene branch), x_gene shape:", x_gene.shape)
            #x_gene = self.pre_conv_linear_gene(x_gene)
            #print("[GCN] After pre_conv_linear_gene (gene branch), x_gene shape:", x_gene.shape)
            #x_gene = F.relu(x_gene)
            #print("[GCN] After ReLU (gene branch), x_gene shape:", x_gene.shape)

            #print("[GCN] Before pre_conv_linear_mirna, x_mirna shape:", x_mirna.shape)
            #x_mirna = self.pre_conv_linear_mirna(x_mirna)
            #print("[GCN] After pre_conv_linear_mirna, x_mirna shape:", x_mirna.shape)
            #x_mirna = F.relu(x_mirna)
            #print("[GCN] After ReLU (mirna), x_mirna shape:", x_mirna.shape)

            #x_gene = x_gene.view(batches, -1, self.raised_dimension)
            #print("[GCN] x_gene reshaped for conv, shape:", x_gene.shape)
            #x_mirna = x_mirna.view(batches, -1, self.raised_dimension)
            #print("[GCN] x_mirna reshaped for conv, shape:", x_mirna.shape)

            #x = torch.cat([x_gene,x_mirna],dim=1)
            #print("[GCN] After concatenating gene and mirna branches, x shape:", x.shape)

        # print dimensions after pre-conv
        self._log(f"[GCN] x shape before conv layers: {x.shape}")
        self._log(f"edge_index shape: {edge_index.shape}")
        self._log(f"edge_index max: {int(edge_index.max())} expected max ~ {batches*num_node - 1}")
        
        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)
        
        # ------------- Jumping Knowledge (if any) --------------
        xs = []
        ################ Forward pass through GCN layers ################
        # -------------Conv Layers 1--------------
        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
        
        if self.jumping_knowledge is True:
            xs.append(x)
            self._log(f"JK xs length: {len(xs)}")
        self._log(f"[GCN] x shape after conv1: {x.shape}")
        
        # -------------Conv Layers 2--------------
        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
        else:
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        
        if self.jumping_knowledge is True:
            xs.append(x)
            self._log(f"JK xs length after conv2: {len(xs)}")
        
        # jumping knowledge forward
        self._log(f"[GCN] x shape after conv2: {x.shape}")
        if self.jumping_knowledge:
            x = self.jk(xs)
            self._log(f"[GCN] x shape after JK ({self.jk_mode}): {x.shape}")
        else:
            pass
        
        # safety check (helps catch silent dimension bugs)
        if x.size(-1) != self.jk_out:
            raise RuntimeError(f"JK output dim mismatch: got {x.size(-1)} expected {self.jk_out}")

        ## pooling on the graph to reduce nodes
        self._log(f"[GCN] x shape before pooling: {x.shape}")
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, jk_out]
        x = self.graph_max_pool(x, self.poolsize)   ## output shape: [batches, floor(num_node / poolsize), jk_out]
        self._log(f"[GCN] x shape after graph max pool: {x.shape}")
        
        if self.method == 'gcn_tf':
            x = x.view(-1, self.jk_out) ## output shape:[batches * floor(num_node / poolsize), jk_out]

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * jk_out]
        
        # -------------Fully Connected Layers--------------
        self._log(f"[GCN] x shape before linear layers: {x.shape}")
        x = self.linear1(x)
        self._log(f"[GCN] x shape after linear1: {x.shape}")
        x = F.relu(x)
        self._log(f"[GCN] x shape after ReLU: {x.shape}")
        x = self.linear2(x)
        self._log(f"[GCN] x shape after linear2: {x.shape}")
        x = F.relu(x)
        self._log(f"[GCN] x shape after ReLU: {x.shape}")
        # -------------Decoder Layers--------------

        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(self.dropout_rate)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)

            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)
            
            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            #return x_reconstruct, F.log_softmax(x, dim=1)
            return x_reconstruct, x
        else:
            #return F.log_softmax(x, dim=1)
            return x
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization, class_weights=None):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp = x_target[:, :, 0]   # (B, N)
                x_target_cnv = x_target[:, :, 1]   # (B, N)

                G = self.num_gene
                M = self.num_mirna
                T = self.num_tf  # può essere 0

                # Slice per blocchi [GENI | miRNA | TF]
                gene_exp  = x_target_exp[:, :G]              # (B, G)
                gene_cnv  = x_target_cnv[:, :G]              # (B, G)

                mirna_exp = x_target_exp[:, G:G+M]           # (B, M)  (cnv è padding)

                tf_exp    = x_target_exp[:, G+M:G+M+T]       # (B, T)
                tf_cnv    = x_target_cnv[:, G+M:G+M+T]       # (B, T)

                # Flatten con decoder_2 (omic_mode 4):
                # [gene_exp, gene_cnv, tf_exp, tf_cnv, mirna_exp]
                x_target_flatten = torch.cat([gene_exp, gene_cnv, tf_exp, tf_cnv, mirna_exp], dim=1)

                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0
        
        if class_weights is not None:
            class_weights = class_weights.to(y.device)  # sicurezza: stesso device di logit
            loss2 = nn.CrossEntropyLoss(weight=class_weights)(y, y_target)
        else:
            loss2 = nn.CrossEntropyLoss()(y, y_target)
        
        loss = 1*loss1 + 1*loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss2_val = 0.2* l2_regularization* l2_loss
            loss += loss2_val
        
        return loss