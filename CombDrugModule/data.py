import os, csv
import numpy as np
import pandas as pd
from sklearn import model_selection, impute, preprocessing, feature_selection


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD_DIR = os.path.join(BASE_DIR, "CombDrugModule/")
DATA_DIR = os.path.join(MOD_DIR, "Data/")
preTRAIN_DIR = os.path.join(MOD_DIR, "preTrainModel/ALL/")
TMP_DIR = os.path.join(BASE_DIR, "tmp/")

def checkFrameNAINF(data):
    """
    Find whether have nan value or infinity value from pandas
    Caution: All columns must be floate values
    """
    
    nan_col = data.columns.to_series()[np.isnan(data).any()]
    print("There are {} columns  where nan value are present".format(len(nan_col)))
    nan_row = data.index[np.isnan(data).any(1)]
    print("There are {} rows  where nan value are present".format(len(nan_row)))
    
    inf_col = data.columns.to_series()[np.isinf(data).any()]
    print("There are {} columns  where infinity value are present".format(len(inf_col)))
    inf_row = data.index[np.isinf(data).any(1)]
    print("There are {} rows  where infinity value are present".format(len(inf_row)))
    
    return nan_col, nan_row, inf_col, inf_row

def getkFoldIndex(score_name, num_splits, num_stop, random_num=0):
    """
    Get K Fold, train and test index
    """
    drug_file_dir = "/home/jingbozhou/Project/CombDrug/rawData/"
    
    drug_index_arr = pd.read_csv(os.path.join(drug_file_dir, 
                                              "three_database_comb_{}.tsv".format(score_name)), 
                                 sep="\t", index_col="DrugComb_id").index.values
    
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=random_num)
    
    num_round = 1
    for train_index, test_index in kf.split(drug_index_arr):
        if num_round == num_stop:
            break
        else:
            num_round += 1
            
    return drug_index_arr[train_index], drug_index_arr[test_index]



def scaleRawData(data, scale_file_path, feature_list=None, feature_prefix='feature_', impute_mode="median", fill_value=0, scale_mode="Standard"):
    """
    Impute and scale test data
    """
    raw_data = pd.read_csv(scale_file_path, sep="\t")
    
    if feature_list:
        feature_columns = [x for x in feature_list if x.startswith(feature_prefix)]
    else:
        # feature_columns = raw_data.filter(like=feature_prefix).columns
        feature_columns = raw_data.filter(regex='^{}'.format(feature_prefix), axis=1).columns
    
    impute_tool = impute.SimpleImputer(strategy=impute_mode, fill_value=fill_value).fit(raw_data.loc[:, feature_columns])
    data.loc[:, feature_columns] = impute_tool.transform(data.loc[:, feature_columns])
    raw_data.loc[:, feature_columns] = impute_tool.transform(raw_data.loc[:, feature_columns])
    
    data_scale = data.copy()
    if scale_mode == "Normalize":
        scale_tool = preprocessing.Normalizer().fit(raw_data.loc[:, feature_columns])
    elif scale_mode == "Standard":
        scale_tool = preprocessing.StandardScaler().fit(raw_data.loc[:, feature_columns])
    
    data_scale.loc[:, feature_columns] = scale_tool.transform(data.loc[:, feature_columns])
    
    return data_scale

def rmLowVarAndHighCor(data, feature_prefix='feature_', var_threshold=0.01, cor_threshods=0.8):
    
    data_feat = data.filter(regex='^{}'.format(feature_prefix), axis=1)
    
    # remove quasi-constant features
    # 0.01 indicates 99% of observations approximately
    var_sel = feature_selection.VarianceThreshold(threshold=var_threshold)
    var_sel.fit(data_feat)
    
    feat_keep = data_feat.columns[var_sel.get_support()]
    data_feat = data_feat[feat_keep]
    
    # drop highly correlated features
    corr_mat = data_feat.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > cor_threshods)]
    
    feat_keep = [column for column in data_feat.columns if column not in to_drop]
    return feat_keep
