from amogel.data_preprocessing import DataPreprocessing
import yaml
import pandas as pd

def test_data_preprocessing():
    
    config = yaml.safe_load(open('config.yaml'))
    preprocessing = DataPreprocessing()
    
    omic1 , omic2 , omic3 , label = preprocessing.load_data('BRCA' , config['datasets']['BRCA'])
    
    assert isinstance(omic1 , pd.DataFrame)
    assert isinstance(omic2 , pd.DataFrame)
    assert isinstance(omic3 , pd.DataFrame)
    assert isinstance(label , pd.DataFrame)
    assert omic1.shape[1] > 1
    assert omic2.shape[1] > 1
    assert omic3.shape[1] > 1
    
    omic1_clean , omic2_clean , omic3_clean , label_clean = preprocessing.data_cleaning(omic1 , omic2 , omic3 , label)
    
    assert omic1_clean.shape[0] == omic2_clean.shape[0] == omic3_clean.shape[0] == label_clean.shape[0] 
    assert omic1_clean.isnull().sum().sum() == 0
    assert omic2_clean.isnull().sum().sum() == 0
    assert omic3_clean.isnull().sum().sum() == 0
    assert label_clean.isnull().sum().sum() == 0
    
    # check the index of dataframes all must be same 
    assert all([x[0] == x[1] for x in zip(omic1_clean.index.to_list() , label_clean.index.to_list()) ]) == True 
    assert all([x[0] == x[1] for x in zip(omic2_clean.index.to_list() , label_clean.index.to_list()) ]) == True
    assert all([x[0] == x[1] for x in zip(omic3_clean.index.to_list() , label_clean.index.to_list()) ]) == True
    
    # calculate the duplicated features 
    omic1_features_t1 , omic2_features_t1 , omic3_features_t1 = preprocessing.feature_selection(omic1_clean , omic2_clean , omic3_clean , label_clean , 0.01 , 1000 )
    omic1_features_t2 , omic2_features_t2 , omic3_features_t2 = preprocessing.feature_selection(omic1_clean , omic2_clean , omic3_clean , label_clean , [0.01 , 0.01 , 0.01] , 1000)
    
    assert omic1_features_t1.shape[1] <= 1000 
    assert omic2_features_t1.shape[1] <= 1000
    assert omic3_features_t1.shape[1] <= 1000
    assert omic1_features_t2.shape[1] <= 1000
    assert omic2_features_t2.shape[1] <= 1000
    assert omic3_features_t2.shape[1] <= 1000
    
    assert omic1_features_t1.shape[1] == omic1_features_t2.shape[1]
    assert omic2_features_t1.shape[1] == omic2_features_t2.shape[1]
    assert omic3_features_t1.shape[1] == omic3_features_t2.shape[1]
    
    ## split the data into train and test
    omic1_train  , omic2_train , omic3_train , label_train , omic1_test , omic2_test , omic3_test , label_test = preprocessing.split_data(omic1_features_t1 , omic2_features_t1 , omic3_features_t1 , label_clean , 0.2 , stratafied=True)

    ## discretize the omic data and encode label 
    omic1_train , omic2_train , omic3_train , label_train , omic1_test , omic2_test , omic3_test , label_test , _ , _ , _ = preprocessing.discretize_data_and_encode_label(omic1_train , omic2_train , omic3_train , label_train , omic1_test , omic2_test , omic3_test , label_test)
    
    assert omic1_train.iloc[: , 0].unique().shape[0] == 3 # [0,1,2]
    assert omic2_train.iloc[: , 0].unique().shape[0] == 3
    assert omic3_train.iloc[: , 0].unique().shape[0] == 3
    assert omic1_test.iloc[: , 0].unique().shape[0] == 3
    assert omic2_test.iloc[: , 0].unique().shape[0] == 3
    assert omic3_test.iloc[: , 0].unique().shape[0] == 3