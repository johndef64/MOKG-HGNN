
import pandas as pd 
import warnings 
from sklearn.feature_selection import f_classif , SelectKBest 
from sklearn.model_selection import train_test_split , StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer , LabelEncoder
from typing  import Union , List
import seaborn as sns 
import matplotlib.pyplot as plt
import yaml
import os
import argparse

warnings.filterwarnings("ignore")

class DataPreprocessing:
    
    def __init__(self):
        pass 
    
    def load_data(self , dataset: str , config: dict) -> tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame]: 
        omic1 = pd.read_csv(config['mRNA'] , sep="\t" , header=None , low_memory=False) # feature x sample
        omic2 = pd.read_csv(config['DNA'] , sep="\t" , header=None , low_memory=False) # feature x sample
        omic3 = pd.read_csv(config['miRNA'] , sep="\t" , header=None) # feature x sample
        label = pd.read_csv(config['label'] , sep="\t" , header=None) # feature x sample
        
        # omic1.set_index(omic1.columns[0] , inplace=True)
        # omic2.set_index(omic2.columns[0] , inplace=True)
        # omic3.set_index(omic3.columns[0] , inplace=True)
        # label.set_index(label.columns[0] , inplace=True)
        
        return omic1 , omic2 , omic3 , label
    
    def data_cleaning(self, omic1:pd.DataFrame , omic2: pd.DataFrame , omic3: pd.DataFrame , label: pd.DataFrame , label_column: str):
        
        omic_1_duplicated_feature = self.__calculate_duplicated_columns__(omic1)
        omic_2_duplicated_feature = self.__calculate_duplicated_columns__(omic2)
        omic_3_duplicated_feature = self.__calculate_duplicated_columns__(omic3)
        print(f"Number of duplicated feature in [omic1 , omic2 , omic3]: {omic_1_duplicated_feature}  , {omic_2_duplicated_feature} , {omic_3_duplicated_feature}")
        
        # clean omic1 
        omic1.iloc[0 , :] = omic1.iloc[0 , :].apply(lambda x : "-".join(x.split("-")[0:3]))
        omic1 = omic1.T 
        omic1.columns = omic1.iloc[0]
        omic1.drop(omic1.index[0] , inplace=True)
        omic1.rename({omic1.columns[0] : "sample"} , axis=1 , inplace=True)
        omic1[omic1.columns[1:]] = omic1[omic1.columns[1:]].astype(float)
        missing_columns = omic1.columns[omic1.isnull().any()].tolist()
        print(f"Number of missing columns in omic1: {len(missing_columns)}")
        omic1.drop(missing_columns , axis=1 , inplace=True)
        origin_length = omic1.shape[0]
        omic1 = omic1.groupby("sample").mean().reset_index()
        print(f"Number of duplicated sample in omic1: {origin_length - omic1.shape[0]}") 
        omic1.set_index("sample" , inplace=True)   
        
        # clean omic2 
        omic2.iloc[0 , :] = omic2.iloc[0 , :].apply(lambda x : "-".join(x.split("-")[0:3]))
        omic2 = omic2.T 
        omic2.columns = omic2.iloc[0]
        omic2.drop(omic2.index[0] , inplace=True)
        omic2.drop(["Composite Element REF"] , axis = 1, inplace=True)
        omic2.rename({omic2.columns[0] : "sample"} , axis=1 , inplace=True)
        omic2[omic2.columns[1:]] = omic2[omic2.columns[1:]].astype(float)
        missing_columns = omic2.columns[omic2.isnull().any()].tolist()
        print(f"Number of missing columns in omic2: {len(missing_columns)}")
        omic2.drop(missing_columns , axis=1 , inplace=True)
        origin_length = omic2.shape[0]
        omic2 = omic2.groupby("sample").mean().reset_index()
        print(f"Number of duplicated sample in omic2: {origin_length - omic2.shape[0]}") 
        omic2.set_index("sample" , inplace=True)
        # omic2.fillna(omic2.mean() , inplace=True)
        
        # clean omic3
        omic3.iloc[0 , :] = omic3.iloc[0 , :].apply(lambda x : "-".join(x.split("-")[0:3]))
        omic3 = omic3.T 
        omic3.columns = omic3.iloc[0]
        omic3.drop(omic3.index[0] , inplace=True)
        omic3.rename({omic3.columns[0] : "sample"} , axis=1 , inplace=True)
        omic3[omic3.columns[1:]] = omic3[omic3.columns[1:]].astype(float)
        missing_columns = omic3.columns[omic3.isnull().any()].tolist()
        omic3.drop(missing_columns , axis=1 , inplace=True)
        print(f"Number of missing columns in omic3: {len(missing_columns)}")
        origin_length = omic3.shape[0]
        omic3 = omic3.groupby("sample").mean().reset_index()
        print(f"Number of duplicated sample in omic3: {origin_length - omic3.shape[0]}") 
        omic3.set_index("sample" , inplace=True)  
        omic3.fillna(omic3.mean() , inplace=True)
        
        # clean label
        label = label.T 
        label.columns = label.iloc[0]
        label.drop(label.index[0] , inplace=True)
        label['bcr_patient_barcode'] = label["bcr_patient_barcode"].apply(lambda x: str.upper(x))
        label.rename({label_column: "label" , "bcr_patient_barcode" : "sample"} , axis=1 , inplace=True)
        label.set_index("sample" , inplace=True)
        label = label[['label']] 
        # remove missing columns
        label = label[~label['label'].isnull()]
        
        # Find common index
        omic1_index = set(omic1.index.to_list())
        omic2_index = set(omic2.index.to_list())
        omic3_index = set(omic3.index.to_list())
        label_index = set(label.index.to_list())
        common_index = list(omic1_index.intersection(omic2_index).intersection(omic3_index).intersection(label_index))
        print(f"Number of common sample: {len(common_index)}")
        
        # Select only common samples
        omic1 = omic1.loc[common_index , : ]
        omic2 = omic2.loc[common_index , : ]
        omic3 = omic3.loc[common_index , : ]
        label = label.loc[common_index , : ]
        
        return omic1 , omic2 , omic3 , label
    
    def feature_selection(self , omic1: pd.DataFrame , omic2: pd.DataFrame , omic3: pd.DataFrame , label: pd.DataFrame , variance_threshold: Union[float , List[float]] , f_annova_threshold: Union[float , List[float]])-> tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame]:
        
        # TODO: Implement variance filtering 
        if isinstance(variance_threshold , float): 
            omic1 = self.__variance_threshold__(omic1 , variance_threshold)
            omic2 = self.__variance_threshold__(omic2 , variance_threshold)
            omic3 = self.__variance_threshold__(omic3 , variance_threshold)
        elif isinstance(variance_threshold , list) and len(variance_threshold) == 3:
            omic1 = self.__variance_threshold__(omic1 , variance_threshold[0])
            omic2 = self.__variance_threshold__(omic2 , variance_threshold[1])
            omic3 = self.__variance_threshold__(omic3 , variance_threshold[2])
        else: 
            raise ValueError("Variance threshold must be float or list of float with length 3")
        
        # TODO: Implement F-Anova test selection
        if isinstance(f_annova_threshold , float) or isinstance(f_annova_threshold, int):
            omic1 = self.__annova_test__(omic1 , label , f_annova_threshold)
            omic2 = self.__annova_test__(omic2 , label , f_annova_threshold)
            omic3 = self.__annova_test__(omic3 , label , f_annova_threshold)
        else: 
            omic1 = self.__annova_test__(omic1 , label , f_annova_threshold[0])
            omic2 = self.__annova_test__(omic2 , label , f_annova_threshold[1])
            omic3 = self.__annova_test__(omic3 , label , f_annova_threshold[2])
            
        return omic1 , omic2 , omic3
    
    def __variance_threshold__(self , omic: pd.DataFrame , threshold: float): 
        omic_variance = omic.var()
        selected_features = omic_variance[omic_variance >= threshold].index.to_list()
        return omic[selected_features]
    
    def __annova_test__(self , omic: pd.DataFrame , label: pd.DataFrame , threshold: float): 
        
        sel = SelectKBest(f_classif , k=threshold)
        sel.fit_transform(omic , label)
        feature_mask = sel.get_feature_names_out()
        
        return omic[feature_mask]
    
    def annova_test_analysis(self , omic: pd.DataFrame , label: pd.DataFrame , range: range): 
        
        for i in range:
            sel = SelectKBest(f_classif , k=i)
            features = sel.fit_transform(omic , label)
            
            # get the variance of first principle component
            pca = PCA(n_components=2)
            pca.fit(features)
            
            scores = pca.explained_variance_ratio_
            print(f"{i} | 1st PC var: {scores[0]:.3f} | Shape: {features.shape} | 2nd PC var: {scores[1]:.3f} | Sum of variance: {scores[0] + scores[1]}")
    
    def split_data(self , 
            omic1: pd.DataFrame, 
            omic2: pd.DataFrame,
            omic3: pd.DataFrame,
            label: pd.DataFrame,
            split_test_ratio:float = 0.3,
            stratafied: bool = False, 
            random_state:int = 42,
        ) -> tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame]:
        
        if not stratafied:
            train_index , test_index = train_test_split(label.index , test_size=split_test_ratio , random_state=random_state)
        else: 
            sss = StratifiedShuffleSplit(n_splits=1 , test_size=split_test_ratio , random_state=random_state)
            for train_index , test_index in sss.split(omic1 , label):
                train_index = label.iloc[train_index , :].index 
                test_index = label.iloc[test_index , :].index
                pass
        
        return omic1.loc[train_index , :] , omic2.loc[train_index , :] , omic3.loc[train_index , :] , label.loc[train_index , :] , omic1.loc[test_index , :] , omic2.loc[test_index , :] , omic3.loc[test_index , :] , label.loc[test_index , :]
    
    def generate_train_test_label_distribution(self , train_label: pd.DataFrame , test_label: pd.DataFrame , save_fig: bool = False): 
        
        #fig , ax = plt.subplots(1 , 1 , figsize=(10, 10))
        
        train_label_copy = train_label.copy()
        train_label_copy['type'] = "Train"
        test_label_copy = test_label.copy()
        test_label_copy['type'] = "Test"
        
        label = pd.concat([ train_label_copy , test_label_copy ] , axis=0)
        sns.displot(label , x="label" , hue="type"  , stat='density' , common_norm=False) 
        #plt.show()
        
    def discretize_data_and_encode_label(self ,
        omic1_train: pd.DataFrame ,
        omic2_train: pd.DataFrame ,
        omic3_train: pd.DataFrame ,
        label_train: pd.DataFrame ,
        omic1_test: pd.DataFrame ,
        omic2_test: pd.DataFrame ,
        omic3_test: pd.DataFrame ,
        label_test: pd.DataFrame
    ) -> tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame , KBinsDiscretizer , LabelEncoder]:
        
        omic1_train , omic1_test , discretizer_omic1 = self.__discretized_data__(omic1_train , omic1_test)
        omic2_train , omic2_test , discretizer_omic2 = self.__discretized_data__(omic2_train , omic2_test)
        omic3_train , omic3_test , discretizer_omic3 = self.__discretized_data__(omic3_train , omic3_test)
        
        label_train , label_test = self.__encode_label__(label_train , label_test)
        
        return omic1_train , omic2_train , omic3_train , label_train , omic1_test , omic2_test , omic3_test , label_test , discretizer_omic1 , discretizer_omic2 , discretizer_omic3
    
    def __discretized_data__(self , 
            train_data: pd.DataFrame , 
            test_data: pd.DataFrame ,
        ) -> tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame]:
        
        ## discretize data to low , medium and high gene expression 
        discretizer = KBinsDiscretizer(n_bins=3 , encode='ordinal' , strategy='quantile')
        discretizer.fit(train_data)
        train_data = pd.DataFrame(discretizer.transform(train_data) , columns=train_data.columns , index=train_data.index)
        test_data  = pd.DataFrame(discretizer.transform(test_data) , columns=test_data.columns , index=test_data.index)
        
        
        return train_data , test_data , discretizer
    
    def __encode_label__(self , train_label: pd.DataFrame , test_label: pd.DataFrame) -> tuple[pd.DataFrame , pd.DataFrame]: 
        
        enc = LabelEncoder()
        enc.fit(train_label['label'])
        
        train_label['label'] = enc.transform(train_label['label']) 
        test_label['label'] = enc.transform(test_label['label'])
        
        return train_label , test_label
    
    def __calculate_duplicated_columns__(self , data:pd.DataFrame , row=False): 
        if row: 
            return data.T[0].duplicated().sum()
        else: 
            return data[0].duplicated().sum()
        
    def __mean_duplicated_columns__(self , data:pd.DataFrame): 
        raise NotImplementedError("This method is not implemented yet")
        
    def save_data(self):
        raise NotImplementedError("This method is not implemented yet")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument("--dataset" , type=str , default="BRCA" , choices=["BRCA" , "KIPAN"])
    
    args = parser.parse_args()
    
    config = yaml.safe_load(open('config.yaml'))
    params = yaml.safe_load(open('params.yaml'))
    preprocessing = DataPreprocessing()
    
    omic1 , omic2 , omic3 , label = preprocessing.load_data(args.dataset , config['datasets'][args.dataset])
    omic1_clean , omic2_clean , omic3_clean , label_clean = preprocessing.data_cleaning(omic1 , omic2 , omic3 , label , config['datasets'][args.dataset]['label_column'])
    omic1_features , omic2_features , omic3_features = preprocessing.feature_selection(omic1_clean , omic2_clean , omic3_clean , label_clean , params['data_preprocessing'][args.dataset]['variance'] , params['data_preprocessing'][args.dataset]['annova_f'])
    
    ## split the data into train and test
    omic1_train  , omic2_train , omic3_train , label_train , omic1_test , omic2_test , omic3_test , label_test = preprocessing.split_data(
        omic1_features , omic2_features , omic3_features , label_clean , 0.3 , stratafied=True)

    ## discretize the omic data and encode label 
    omic1_train , omic2_train , omic3_train , label_train , omic1_test , omic2_test , omic3_test , label_test , _ , _ , _ = preprocessing.discretize_data_and_encode_label(omic1_train , omic2_train , omic3_train , label_train , omic1_test , omic2_test , omic3_test , label_test)
    
    output_dir = f"output/features/{args.dataset}"
    
    os.makedirs(output_dir , exist_ok=True)
    omic1_train.to_csv(f"{output_dir}/1_tr.csv" , index=False , header=False)
    omic2_train.to_csv(f"{output_dir}/2_tr.csv" , index=False , header=False)
    omic3_train.to_csv(f"{output_dir}/3_tr.csv" , index=False , header=False)
    label_train.to_csv(f"{output_dir}/label_tr.csv" , index=False , header=False)
    
    omic1_test.to_csv(f"{output_dir}/1_te.csv" , index=False , header=False)
    omic2_test.to_csv(f"{output_dir}/2_te.csv" , index=False , header=False)
    omic3_test.to_csv(f"{output_dir}/3_te.csv" , index=False , header=False)
    label_test.to_csv(f"{output_dir}/label_te.csv" , index=False , header=False)
    
    pd.DataFrame(omic1_train.columns).to_csv(f"{output_dir}/1_features.csv" , index=False , header=False)
    pd.DataFrame(omic2_train.columns).to_csv(f"{output_dir}/2_features.csv" , index=False , header=False)
    pd.DataFrame(omic3_train.columns).to_csv(f"{output_dir}/3_features.csv" , index=False , header=False)
    
    
    
