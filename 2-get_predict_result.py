#!/home/jingbozhou/anaconda3/envs/drugCombPro/bin/python
import os, sys, logging, pickle, random, subprocess, joblib
import numpy as np
import pandas as pd

import pubchempy as pcp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "CombDrugModule/Data/")

sys.path.append(BASE_DIR)
import CombDrugModule

os.environ['OPENBLAS_NUM_THREADS'] = '2'

preTrain_path = DATA_DIR

def getDrugSmile(drug_name):
    c = pcp.get_compounds(drug_name, "name")
    if len(c) > 0:
        com_flag = c[0]
    else:
        item = drug_name.split(" (")[0]
        c = pcp.get_compounds(item, "name")
        if len(c) == 0:
            raise ValueError("We cannot find {} pubmed cid and corresponding SMILES!".format(drug_name))
        else:
            com_flag = c[0]
            
    return [drug_name, str(com_flag.cid), str(com_flag.canonical_smiles)]
            
def predFourScoreResult(input_data, in_type, pre_num):
    
    data_for_train = input_data.copy()
    if in_type == "Row":
        data_for_train = data_for_train.rename(columns={'Drug_1':'Drug_row', 
                                                        'Drug_2':'Drug_col',
                                                        'Drug_1_cid':'Drug_row_cid', 
                                                        'Drug_2_cid':"Drug_col_cid"})
        logging.info("Concat first result ...")
    elif in_type == "Col":
        data_for_train = data_for_train.rename(columns={'Drug_2':'Drug_row', 
                                                        'Drug_1':'Drug_col',
                                                        'Drug_2_cid':'Drug_row_cid', 
                                                        'Drug_1_cid':"Drug_col_cid"})
        logging.info("Concat second result ...")
    
    ## Get all features columns
    with open(os.path.join(preTrain_path,
                           "HSA_RandomForest_feat_name"), 'rb') as f:
        feat_cols = pickle.load(f)

    ## Concat features
    logging.info("Concat all features ...")
    for func_name in [CombDrugModule.getInfomaxFeatureNew,
                      CombDrugModule.getMordredFeatureNew,
                      CombDrugModule.getRDKitFeatureNew,
                      CombDrugModule.getgeneRawFeatureNew,
                      CombDrugModule.getMutFeatureNew,
                      CombDrugModule.getCNVFeatureNew]:
        feature_df = func_name(data_for_train, pre_num, features=feat_cols)
        data_for_train = pd.concat([data_for_train, 
                                    feature_df.reindex(data_for_train.index)], 
                                   axis=1, sort=False)
    
    del feat_cols
    
    ## Begin predict
    for item in ["ZIP_XGBoost", "Bliss_XGBoost", 
                 "HSA_RandomForest", "Loewe_RandomForest"]:
        score_name, reg_name = item.split("_")
        logging.info("{} start...".format(score_name))
        
        ## Get features columns
        with open(os.path.join(preTrain_path,
                               "{}_{}_feat_name".format(score_name, reg_name)), 'rb') as f:
            feat_cols = pickle.load(f)
            
        ## Get raw test feature columns
        raw_data = data_for_train[feat_cols].copy()
        
        ## load scale model
        scale_tool = joblib.load(os.path.join(preTrain_path,
                                              "{}_{}_scale_model".format(score_name, reg_name)))
        
        logging.info("Start impute and Scale test...")
        raw_data.loc[:, feat_cols] = scale_tool.transform(raw_data.loc[:, feat_cols])
        
        test_data_index = raw_data.index
        
        ## Get X_test
        X_test  = raw_data.filter(regex="^{}".format('feature_'), axis=1).values
        #logging.info("X shape for {} is {}".format(score_name, X_test.shape))
        
        del raw_data
        
        logging.info("Load train model...")
        ## load train model
        model = joblib.load(os.path.join(preTrain_path,
                                         "{}_{}_train_model".format(score_name, reg_name)))
        
        ## Test model
        logging.info('Test {}...'.format(reg_name))
        y_pred = model.predict(X_test)
        
        ## Get predict data
        pred_list = []
        for x, y in zip(test_data_index, y_pred):
            pred_list.append([x, y])
            
        pred_df = pd.DataFrame(pred_list, columns=["DrugComb_id", "pred_{}".format(score_name)])
        pred_df = pred_df.set_index("DrugComb_id")
        
        del pred_list
        
        data_for_train = pd.concat([data_for_train, pred_df], axis=1, sort=False)
        
        del feat_cols, pred_df
        
    data_for_train = data_for_train[["Drug_row", "Drug_col", 
                                     "Cell_line_name", 
                                     "pred_ZIP", "pred_Bliss", 
                                     "pred_HSA", "pred_Loewe"]]
    
    data_for_train.to_csv(os.path.join(CombDrugModule.TMP_DIR,
                                       "Drug_{}_{}".format(in_type, pre_num)),
                                       sep="\t",  float_format='%.3f', index=0)
    return data_for_train    

input_file = sys.argv[1]
output_file = sys.argv[2]

## Get input data
input_data = pd.read_csv(input_file)

## Get unique drug list
drug_list = list(set(input_data["Drug_1"].unique()).union(set(input_data["Drug_2"].unique())))
drug_name2cid = {}
drug_cid2smile = {}
for item in drug_list:
    drug_name_list = getDrugSmile(item)
    drug_name2cid[item] = drug_name_list[1]
    drug_cid2smile[drug_name_list[1]] = drug_name_list[2]
    
input_data["Drug_1_cid"] = input_data["Drug_1"].map(drug_name2cid)
input_data["Drug_2_cid"] = input_data["Drug_2"].map(drug_name2cid)

with open(os.path.join(CombDrugModule.DATA_DIR, "cell_line_map.pickle"), "rb") as f:
    name_map_d = pickle.load(f)
    
input_data["DepMap_ID"] = input_data["Cell_line_name"].map(name_map_d)


# Write cid dict to a dataframe
cid_df = pd.DataFrame.from_dict(drug_cid2smile, orient='index', columns=["canonical_smiles"])

# Get a ranodm file prefix number
pre_num = random.randint(1, 9999)

# Get a new random number if file already exist
while os.path.isfile(os.path.join(CombDrugModule.TMP_DIR,
                                  "cid2smiles_{}.pickle".format(pre_num))):
    pre_num = random.randint(1, 9999)

# Write cid and corresponding smiles to a file
with open(os.path.join(CombDrugModule.TMP_DIR,
                       "cid2smiles_{}.pickle".format(pre_num)), "wb") as f:
    pickle.dump(cid_df, f)
    
    
# Run feature 
logging.info("Starting get Infomax features...")
cmd = "bash {} {}".format(os.path.join(CombDrugModule.MOD_DIR, "runInfomax.sh"),
                          pre_num)
subprocess.call(cmd, shell=True)
    
logging.info("Starting get Mordred features...")
CombDrugModule.runForMordred(pre_num)
    
logging.info("Starting get RDKit features...")
CombDrugModule.runForRDKit(pre_num)

logging.info("Drug features Done...")

logging.info("Starting get gene expression features...")
CombDrugModule.runGeneNew(list(input_data["DepMap_ID"].unique()), pre_num)
    
logging.info("Starting get Mutation and CNV features...")
CombDrugModule.runMutNew(list(input_data["DepMap_ID"].unique()), pre_num)
CombDrugModule.runCNVNew(list(input_data["DepMap_ID"].unique()), pre_num)


data_row = predFourScoreResult(input_data, "Row", pre_num)
data_col = predFourScoreResult(input_data, "Col", pre_num)

logging.info("Concat the data...")
data_concat = pd.concat([data_row, data_col[["pred_ZIP", "pred_Bliss", "pred_HSA", "pred_Loewe"]]], axis=1)

logging.info("Write result...")
for item in ["ZIP", "Bliss", "HSA", "Loewe"]:
    input_data[item] = data_concat["pred_{}".format(item)].mean(axis=1)
    
input_data.to_csv(output_file, sep=",",  float_format='%.3f', index=0)
