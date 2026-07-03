import pandas as pd
import torch 
import argparse
import yaml
from tqdm import tqdm
from itertools import combinations
import numpy as np 
from amogel.arm import information_gain , correlation
from amogel.utils import concat_multi_omics_feature
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn.functional import one_hot
import os

def generate_synthetic_graph(data: pd.DataFrame, label: pd.DataFrame, CARs: pd.DataFrame, selected_features_index: list[int] ):
    edge_tensor = torch.zeros((data.shape[1] , data.shape[1]) , dtype=torch.float32)
    info_gain = torch.tensor( list(information_gain(data , label).values()), dtype=torch.float32)
    corr = torch.tensor(list(correlation(data , label).values()), dtype=torch.float32)

    with(tqdm(total = CARs.shape[0])) as pbar:
        pbar.set_description("Generate synthetic information graph")
        
        for _ , row in CARs.iterrows():
            rule = eval(row['Antecedents'])
            
            connected_index = [int(x.split(":")[0]) for x in rule]
            
            if len(connected_index) > 1 : 
                np_combinations = np.array([list(x) for x in combinations(connected_index , 2)]) # shape = [ list of possible combination , 2 ]
                
                # undirected graph
                edge_tensor[np_combinations[:,0] , np_combinations[:,1]] = (info_gain[np_combinations[:,0]] +  info_gain[np_combinations[:,1]] + corr[np_combinations[:,0]] + corr[np_combinations[:,1]])/4
                edge_tensor[np_combinations[:,1] , np_combinations[:,0]] = (info_gain[np_combinations[:,0]] +  info_gain[np_combinations[:,1]] + corr[np_combinations[:,0]] + corr[np_combinations[:,1]])/4
                
            pbar.update(1)
    
    return edge_tensor[selected_features_index][: , selected_features_index]

def generate_kegg_graph(dataset:str , pFilter:float , features_data: pd.DataFrame , selected_features_index: list[int]):
    kegg_df = pd.read_csv(f"data/{dataset}/KEGG_PATHWAY.txt" , sep="\t")[["Genes" , "PValue"]] 
    kegg_df = kegg_df[kegg_df['PValue'] <= pFilter]
    kegg_df['Genes'] = kegg_df['Genes'].apply(lambda x: [int(n) for n in x.split(",")])
    
    available_genes = set(features_data['GeneId'].values)
    edge_tensor = torch.zeros((features_data.shape[0] , features_data.shape[0]) , dtype=torch.float32)
    for gene_list in kegg_df['Genes']: 
        path = set(gene_list)
        
        if len(path.intersection(available_genes)) > 1:
            # get all the features idx 
            gene_loc = features_data[features_data['GeneId'].isin(path.intersection(available_genes))]['FeatureIdx'].tolist()
            np_combinations = np.array([list(x) for x in combinations(gene_loc , 2)]) # shape = [ list of possible combination , 2 ]
            edge_tensor[np_combinations[:,0] , np_combinations[:,1]] = 1
            edge_tensor[np_combinations[:,1] , np_combinations[:,0]] = 1
        
    return edge_tensor[selected_features_index][: , selected_features_index]

def generate_go_graph(dataset:str , pFilter:float , features_data: pd.DataFrame , selected_features_index: list[int]): 
    kegg_df = pd.read_csv(f"data/{dataset}/BP_DIRECT.txt" , sep="\t")[["Genes" , "PValue"]] 
    kegg_df = kegg_df[kegg_df['PValue'] <= pFilter]
    kegg_df['Genes'] = kegg_df['Genes'].apply(lambda x: [int(n) for n in x.split(",")])
    
    available_genes = set(features_data['GeneId'].values)
    edge_tensor = torch.zeros((features_data.shape[0] , features_data.shape[0]) , dtype=torch.float32)
    for gene_list in kegg_df['Genes']: 
        path = set(gene_list)
        
        if len(path.intersection(available_genes)) > 1:
            # get all the features idx 
            gene_loc = features_data[features_data['GeneId'].isin(path.intersection(available_genes))]['FeatureIdx'].tolist()
            np_combinations = np.array([list(x) for x in combinations(gene_loc , 2)]) # shape = [ list of possible combination , 2 ]
            edge_tensor[np_combinations[:,0] , np_combinations[:,1]] = 1
            edge_tensor[np_combinations[:,1] , np_combinations[:,0]] = 1
        
    return edge_tensor[selected_features_index][: , selected_features_index] 

def generate_ppi_graph(filter_combined_score:float , features_data:pd.DataFrame , selected_features_index: list[int]):
    ppi_info_path = "data/PPI/protein_info.parquet.gzip"
    ppi_link_path = "data/PPI/protein_links.parquet.gzip"

    df_protein = pd.read_parquet(ppi_info_path)
    df_protein_link = pd.read_parquet(ppi_link_path)

    df_protein_merged = pd.merge(df_protein_link, df_protein[['#string_protein_id','preferred_name']], left_on="protein1", right_on="#string_protein_id")
    df_protein_merged.rename(columns={"preferred_name":"protein1_name"}, inplace=True)

    df_protein_merged = pd.merge(df_protein_merged, df_protein[['#string_protein_id','preferred_name']], left_on="protein2", right_on="#string_protein_id")
    df_protein_merged.rename(columns={"preferred_name":"protein2_name"}, inplace=True)

    # drop columns
    df_protein_merged.drop(columns=["#string_protein_id_x", "#string_protein_id_y", "protein1" , "protein2"], inplace=True)
    df_protein_merged.head()


    df_protein_merged = df_protein_merged.merge(features_data[['FeatureIdx' , 'GeneSymbol']] , left_on="protein1_name", right_on="GeneSymbol" , how="left")
    df_protein_merged.rename(columns={"FeatureIdx":"gene1_idx"}, inplace=True)

    df_protein_merged = df_protein_merged.merge(features_data[['FeatureIdx' , 'GeneSymbol']] , left_on="protein2_name", right_on="GeneSymbol" , how="left")
    df_protein_merged.rename(columns={"FeatureIdx":"gene2_idx"}, inplace=True)

    #df_protein_merged.drop(columns=["gene_name_x", "gene_name_y"], inplace=True)
        
    # filter rows with only gene1_idx and gene2_idx
    df_filter_protein = df_protein_merged[df_protein_merged['gene1_idx'].notnull()][df_protein_merged['gene2_idx'].notnull()]

    if filter_combined_score > 0:
        df_filter_protein = df_filter_protein[df_filter_protein['combined_score'] >= filter_combined_score]
    
    edge_tensor = torch.zeros((features_data.shape[0] , features_data.shape[0]) , dtype=torch.float32)
    edge_tensor[df_filter_protein['gene1_idx'].values , df_filter_protein['gene2_idx'].values] = torch.tensor(df_filter_protein['combined_score'].values , dtype=torch.float32)
    edge_tensor[df_filter_protein['gene2_idx'].values , df_filter_protein['gene1_idx'].values] = torch.tensor(df_filter_protein['combined_score'].values , dtype=torch.float32)
    edge_tensor = edge_tensor / edge_tensor.max()
    
    return edge_tensor[selected_features_index][: , selected_features_index]

def symmetric_matrix_to_pyg(matrix , node_features , y , edge_threshold=0.0):
    rows, cols = np.triu_indices_from(np.ones((matrix.shape[0] , matrix.shape[1])))
    
    data = matrix[rows, cols] # [number of edges , number of features]
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
        
    indices , values = to_undirected(torch.LongTensor(np.vstack((rows, cols))) , torch.FloatTensor(data) , reduce="mean")
    
    ## Filter the edges with all features is more than 0 
    mask = torch.any(values[:,:1] > edge_threshold , dim=-1)
    indices = indices[:,mask]
    values = values[mask]
    
    return Data(x=node_features , edge_index=indices , edge_attr=values , num_nodes=node_features.shape[0] , y=y , extra_label=torch.arange(node_features.size(0)))


def visualize_graph(edge_tensor: torch.Tensor):
    pass 

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='Generate graph from data')
    parser.add_argument('--dataset', type=str, default="BRCA" , choices=['BRCA' , 'KIPAN'])
    
    args = parser.parse_args()
    
    params = yaml.safe_load(open("params.yaml"))
    
    # load omic data
    omic1_train = pd.read_csv(f"output/features/{args.dataset}/1_tr.csv" , header=None)
    omic2_train = pd.read_csv(f"output/features/{args.dataset}/2_tr.csv" , header=None)
    omic3_train = pd.read_csv(f"output/features/{args.dataset}/3_tr.csv" , header=None)
    omic1_test = pd.read_csv(f"output/features/{args.dataset}/1_te.csv" , header=None)
    omic2_test = pd.read_csv(f"output/features/{args.dataset}/2_te.csv" , header=None)
    omic3_test = pd.read_csv(f"output/features/{args.dataset}/3_te.csv" , header=None)
    
    # load omic features name
    feature1 = pd.read_csv(f"output/features/{args.dataset}/1_features.csv" , header=None)
    feature2 = pd.read_csv(f"output/features/{args.dataset}/2_features.csv" , header=None)
    feature3 = pd.read_csv(f"output/features/{args.dataset}/3_features.csv" , header=None)
    
    # load label
    label_train = pd.read_csv(f"output/features/{args.dataset}/label_tr.csv" , header=None)
    label_test = pd.read_csv(f"output/features/{args.dataset}/label_te.csv" , header=None)
    
    # feature concatenation
    omic_data_train , features_data = concat_multi_omics_feature(args.dataset , [omic1_train , omic2_train , omic3_train] , [feature1 , feature2 , feature3])
    omic_data_test , _ = concat_multi_omics_feature(args.dataset , [omic1_test , omic2_test , omic3_test] , [feature1 , feature2 , feature3])
    
    # load CARs and selected features
    CARs_df = pd.read_csv(f"output/feature_selection/{args.dataset}/topk_CARs.csv")
    selected_features = pd.read_csv(f"output/feature_selection/{args.dataset}/selected_features.csv")
    selected_features_idx = selected_features['FeatureIdx'].values.tolist()
    selected_features_idx.sort()
    
    # generate synthetic information edge
    information_edges = generate_synthetic_graph(
        omic_data_train , label_train , CARs_df , selected_features_idx
    )
    info_mean = information_edges.mean()
    
    # generate ppi edge 
    ppi_edges = generate_ppi_graph(
        params['graph'][args.dataset]['ppi_score_filter'] , features_data , selected_features_idx
    ) 
    ppi_edges = info_mean * ppi_edges
    
    # generate kegg edge
    kegg_edges = generate_kegg_graph(
        args.dataset , params['graph'][args.dataset]['p_filter'] , features_data , selected_features_idx
    )
    kegg_edges = info_mean * kegg_edges
    
    # generate go edge
    go_edges = generate_go_graph(
        args.dataset , params['graph'][args.dataset]['p_filter'] , features_data , selected_features_idx
    )
    go_edges = info_mean * go_edges
    
    # stack all edges
    stack_edges = torch.stack([information_edges , ppi_edges , kegg_edges , go_edges] , dim=-1)
    
    # Generate graph data
    training_graphs = []
    with tqdm(total=omic_data_train.shape[0]) as pbar:
        pbar.set_description("Generate train graph data")
        omics_data = omic_data_train.values
        # loop by row 
        for i in range(omics_data.shape[0]):
            data = one_hot(torch.tensor(omics_data[i] , dtype=torch.long) , num_classes=3).float()
            label = torch.tensor(label_train.iloc[i,0] , dtype=torch.long)
            graph = symmetric_matrix_to_pyg(stack_edges , data , label , edge_threshold=params['graph'][args.dataset]['information_filter'])
            training_graphs.append(graph)
            pbar.update(1)
    
    test_graphs = []
    with tqdm(total=omic_data_test.shape[0]) as pbar:
        pbar.set_description("Generate test graph data")
        omics_data = omic_data_test.values
        # loop by row 
        for i in range(omics_data.shape[0]):
            data = one_hot(torch.tensor(omics_data[i], dtype=torch.long) , num_classes=3).float()
            label = torch.tensor(label_test.iloc[i,0] , dtype=torch.long)
            graph = symmetric_matrix_to_pyg(stack_edges , data , label , edge_threshold=params['graph'][args.dataset]['information_filter'])
            test_graphs.append(graph)
            pbar.update(1)
    
    # Visualize graph
    # TO-DO
    
    # Save graph data 
    os.makedirs(f"output/graph/{args.dataset}" , exist_ok=True)
    torch.save(training_graphs , f"output/graph/{args.dataset}/train.pt")
    torch.save(test_graphs , f"output/graph/{args.dataset}/test.pt")
    
