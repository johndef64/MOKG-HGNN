import pandas as pd 
from sklearn.feature_selection import mutual_info_classif
from amogel.utils import convert_feature_name_to_david_symbol
from fim import ista
from tqdm import tqdm 
from typing import Tuple
import math
import argparse
import os
import yaml

def information_gain(data: pd.DataFrame , label: pd.DataFrame) -> dict: 
    """
    This function calculates the information gain (IG) of each feature with the target in the dataset, it is used in preprocessing
    of the dataset, which is rule ranking and feature selection. It also removes redundant features in the dataset which has an information
    gain lower than 0.

    Args:
        data (pd.DataFrame): The dataset which has been preprocessed to contain only categorical attributes
        label (pd.DataFrame): The index of the target in the dataset

    Returns:
        dict: A dictionary where the key is the index of the column and the value is the information gain.
    """
    
    information_gain = mutual_info_classif(data , label.iloc[: , 0].values , discrete_features=True)
    return { column_name : information_gain[i] for i , column_name  in enumerate(data.columns) }
    
def correlation(data: pd.DataFrame , label: pd.DataFrame) -> dict: 
    """
    This function calculates how correlated the attribute is to the class column. It is used together with information gain for rule
    ranking and feature selection

    Args:
        data (pd.DataFrame): The dataset which has been preprocessed to contain only categorical attributes
        class_column (int): The index of the target in the dataset

    Returns:
        list : a list containing the correlation value of each attribute to the class column
    """
    
    data_copy = pd.concat([data ,label ] , axis= 1)
    class_columns = data_copy.columns[-1]
    corr = data_copy.corr()[class_columns].abs()
    corr.fillna(0 , inplace=True)
    
    
    return { i : corr.iloc[i, 0] for i  in data.columns.to_list() }

def generate_associative_classification(data: pd.DataFrame , label: pd.DataFrame , min_rule_per_class=1000 , min_confidence=0): 
    
    # Seperate the data into classes 
    unique_classes = label.iloc[: , 0].unique() 
    
    # Generate list of transactions per class 
    transactions = {}
    for class_ in unique_classes: 
        data_class = data[label.iloc[: , 0] == class_].copy()
        transactions[class_] = []
        for _ , row in data_class.iterrows(): 
            transactions[class_].append(set([f"{i}:{val}" for i , val in enumerate(row.values)]))
    
    CARs = []
    # Generate rules for each class
    with tqdm(total=len(unique_classes)) as pbar:
        for class_ in unique_classes: 
            pbar.set_description(f"Generating rules for class {class_}")
            data_class = data[label.iloc[: , 0] == class_].copy() # index of label == index of data 
            
            # Generate rules
            min_support = -(len(transactions[class_]))
            rule_count = 0 
            
            while rule_count < min_rule_per_class and min_support < - 10: 
                itemsets = ista(transactions[class_] , target='c' , supp=min_support , report='a')
                rule_count = len(itemsets)
                min_support += 1
                
            print(f"Class {class_} has {rule_count} rules generated")
            
            if rule_count == 0:
                raise ValueError("No rules generated for the class. Please check the data")
            
            sub_CARs = []
            for itemset in itemsets: # iterate each rule (itemset)
                antecedents , upper_support = itemset 
                lower_support = data_class.shape[0]
                support = upper_support / lower_support
                
                # measure confidence 
                match_transaction_within_class = 0 
                match_transaction_outside_class = 0
                
                for _labels , _transactions in transactions.items():
                    if _labels == class_: 
                        match_transaction_within_class += sum([1 for x in _transactions if set(antecedents).issubset(x)])
                    else: 
                        match_transaction_outside_class += sum([1 for x in _transactions if set(antecedents).issubset(x)])
                
                if match_transaction_within_class == 0: 
                    raise ValueError("No transaction found in the class. Please check the data")
                confidence = match_transaction_within_class / (match_transaction_within_class + match_transaction_outside_class)
                
                
                if confidence >= min_confidence: 
                    sub_CARs.append((antecedents , class_ , support , confidence))
            
            print(f"Class {class_} has {len(sub_CARs)} rules generated with confidence >= {min_confidence}")
            CARs.extend(sub_CARs)
            pbar.update(1)
            
    return CARs

def rules_ranking(data:pd.DataFrame , label:pd.DataFrame , CARs: list)-> pd.DataFrame: 
    """ 
    This function ranks the class association rules based on mutual information and correlation
    """
    
    mutual_info = information_gain(data , label)
    corr = correlation(data , label)
    all_CARs = []
    for rule in CARs: 
        antecedents , _class , support , confidence = rule 
        antecedents_index = [int(x.split(":")[0]) for x in antecedents]
        
        avg_mutual_info = sum([mutual_info[i] for i in antecedents_index]) / len(antecedents_index)
        avg_corr = sum([corr[i] for i in antecedents_index]) / len(antecedents_index)
            
        try: 
            interestingness = math.log2(avg_mutual_info) + math.log2(avg_corr) + math.log2(float(confidence)+0.0000001)
            all_CARs.append((antecedents , _class , support , confidence , interestingness))
        except: 
            print(f"Error in calculating interestingness for rule {rule} | avg_ig: {avg_mutual_info} | avg_corr: {avg_corr} | confidence: {confidence}")
            raise ValueError("Error in calculating interestingness")
    
    return pd.DataFrame(all_CARs , columns=["Antecedents" , "Class" , "Support" , "Confidence" , "Interestingness"])

def feature_selection(CARs_data: pd.DataFrame , features_data: pd.DataFrame , topk: int = 100) -> Tuple[pd.DataFrame , list[int] , list[str]] :
    """ 
    This function selects the top k features from the class association
    """
    
    topk_CARs = CARs_data.groupby("Class").apply(lambda x: x.nlargest(topk , "Interestingness")).reset_index(drop=True)
    
    features = set()
    for _ , row in topk_CARs.iterrows(): 
        features = features.union(set([int(x.split(":")[0]) for x in tuple(row["Antecedents"])]))
    features = list(features)
    
    return topk_CARs , list(features) , features_data.iloc[features , 0].to_list()

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description="Generate Class Association")
    parser.add_argument("--dataset" , type=str , default="BRCA" , choices=["BRCA" , "KIPAN"])
    
    args = parser.parse_args()
    
    
    input_dir = f"./output/features/{args.dataset}"
    params = yaml.safe_load(open(f"params.yaml"))
    
    omic1 = pd.read_csv(f'{input_dir}/1_tr.csv' , header=None)
    omic2 = pd.read_csv(f'{input_dir}/2_tr.csv' , header=None)
    omic3 = pd.read_csv(f'{input_dir}/3_tr.csv' , header=None)
    feature1 = pd.read_csv(f'{input_dir}/1_features.csv' , header=None)
    feature1['type'] = "mRNA"
    feature2 = pd.read_csv(f'{input_dir}/2_features.csv' , header=None)
    feature2['type'] = "DNA"
    feature3 = pd.read_csv(f'{input_dir}/3_features.csv' , header=None)
    feature3['type'] = "miRNA"
    label = pd.read_csv(f'{input_dir}/label_tr.csv' , header=None)

    # combined the omic data and features data
    omic = pd.concat([omic1 , omic2 , omic3] , axis=1)
    omic.columns = range(omic.shape[1])
    features = pd.concat([feature1 , feature2 , feature3] , axis=0)
    features.reset_index(drop=True , inplace=True)
    
    
    CARs = generate_associative_classification(omic , label , 1000 , params['arm'][args.dataset]['min_confidence'])
    CARs_df = rules_ranking(omic , label , CARs)
    
    topk_CARs , feature_idx , feature_name = feature_selection(CARs_df , features , 1000)
    print("Number of selected features: " , len(feature_name))
    
    output_dir = f"./output/feature_selection/{args.dataset}"
    os.makedirs(output_dir , exist_ok=True)
    
    CARs_df.to_csv(f"{output_dir}/CARs.csv" , index=False)
    topk_CARs.to_csv(f"{output_dir}/topk_CARs.csv" , index=False)
    
    features['GeneSymbol'] = features[0].apply(lambda x: convert_feature_name_to_david_symbol(x))
    
    # selected features df 
    selected_features_df = features.iloc[feature_idx].copy()
    selected_features_df['FeatureIdx'] = feature_idx
    selected_features_df.to_csv(f"{output_dir}/selected_features.csv" , index=False)
    
    with open(f"{output_dir}/feature_summary.txt" , "w") as f:
        f.write("====== Selected Features Summary ======\n")
        f.write(selected_features_df.groupby("type").size().to_string())
        
        f.write("\n\n")
        f.write(f"Original number of features (mRNA): {omic1.shape[1]}\n")
        f.write(f"Original number of features (DNA): {omic2.shape[1]}\n")
        f.write(f"Original number of features (miRNA): {omic3.shape[1]}\n")
        f.write(f"Total number of selected features: {len(feature_name)}\n")
        
        f.write("\n\n") 
        total_features = omic1.shape[1] + omic2.shape[1] + omic3.shape[1]
        f.write(f"Distribution by type before feature selection: {omic1.shape[1]/total_features:.2f} mRNA | {omic2.shape[1]/total_features:.2f} DNA | {omic3.shape[1]/total_features:.2f} miRNA\n")
        f.write(f"Distribution by type after feature selection: {selected_features_df.groupby('type').size()['mRNA']/len(feature_name):.2f} mRNA | {selected_features_df.groupby('type').size()['DNA']/len(feature_name):.2f} DNA | {selected_features_df.groupby('type').size()['miRNA']/len(feature_name):.2f} miRNA\n")
    