import pandas as pd 
from typing import List , Tuple 

def convert_features_name_to_david_symbol(feature_names: list[str]) -> list[str]: 
    david_symbol = []
    for name in feature_names:
        if "hsa" in name:
            david_symbol.append("".join(name.split("-")[1:]))
        else:
            david_symbol.append(name.split("|")[0])
            
    return david_symbol

def convert_feature_name_to_david_symbol(feature_name:str) -> str: 
    if "hsa" in feature_name:
        return "".join(feature_name.split("-")[1:])
    else:
        return feature_name.split("|")[0]
    
def concat_multi_omics_feature(dataset:str , omics: List[pd.DataFrame] , features: List[pd.DataFrame], omics_type:list[str] = ['mRNA' , 'DNA' , 'miRNA']) -> Tuple[pd.DataFrame , pd.DataFrame]: 
    
    if len(omics) != len(features):
        raise ValueError("The length of omics and features must be same")
    
    for index , omic in enumerate(omics):
        if omic.shape[1] != features[index].shape[0]:
            raise ValueError(f"The number of features in omics[{index}] and features[{index}] must be same")
    
    omic_data = pd.concat(omics , axis=1)
    omic_data.columns = range(omic_data.shape[1])
    
    for i , type in enumerate(omics_type):
        features[i]['type'] = type
    
    # Features Conversion 
    feature_conversion = pd.read_csv(f"data/{dataset}/featname_conversion.csv")
    
    features_data = pd.concat(features , axis=0)
    features_data.reset_index(drop=True , inplace=True)
    features_data['GeneSymbol'] = features_data[0].apply(lambda x: convert_feature_name_to_david_symbol(x))
    features_data['FeatureIdx'] = range(features_data.shape[0])
    features_data = features_data.merge(feature_conversion , left_on="GeneSymbol" , right_on="gene" , how="left")
    features_data.drop(columns=["gene"] , inplace=True)
    features_data.rename(columns={"id":"GeneId"} , inplace=True)
    
    return omic_data , features_data