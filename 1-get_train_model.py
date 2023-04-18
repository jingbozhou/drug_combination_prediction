#!/home/jingbozhou/anaconda3/envs/drugCombPro/bin/python
import os, sys, logging, pickle, random, subprocess, joblib
import multiprocessing
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "CombDrugModule/Data/")

sys.path.append(BASE_DIR)
import CombDrugModule


def getBestParams(score_name, reg_name):
    
    para_feat_name = "{}_{}".format(score_name, reg_name)
    ## ZIP XGBoost
    if para_feat_name == "ZIP_XGBoost":
        return {'colsample_bylevel': 1.0, 'max_depth': 6, 'n_estimators': 2000, 'subsample': 1.0}
    ## Bliss XGBoost
    elif para_feat_name == "Bliss_XGBoost":
        return {'colsample_bylevel': 1.0, 'max_depth': 6, 'n_estimators': 2000, 'subsample': 1.0}
    ## HSA RandomForest
    elif para_feat_name == "HSA_RandomForest":
        return {'max_features': 0.3, 'n_estimators': 3000}
    ## Loewe RandomForest
    elif para_feat_name == "Loewe_RandomForest":
        return {'max_features': 0.3, 'n_estimators': 3000}

def run2SaveTrainModel(type_feat_name):
    
    score_name, reg_name = type_feat_name.split("_")
    
    logging.info("Start {} {}...".format(score_name, reg_name))
    
    ## Get features columns
    with open(os.path.join(DATA_DIR,
                           "{}_{}_feat_name".format(score_name, reg_name)), 'rb') as f:
        feat_cols = pickle.load(f)
        
    ## Get train stand data
    logging.info("Load train data...")
    train_data = pd.read_pickle(os.path.join(DATA_DIR, "train.pickle"), 
                                compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})

    ## Check whether features are same
    if len(set(feat_cols).difference([x for x in train_data.columns if x.startswith("feature_")])) > 0:
        raise ValueError("There is an error in {}".format("type_feat_name"))
    
    train_data = train_data[feat_cols+["score_{}".format(score_name)]]
    
    ## Get X_train, y_train
    X_train, y_train = CombDrugModule.getXy(train_data, feature_prefix='feature_',
                                            y_col_name="score_{}".format(score_name))  
    del train_data
        
    ## Run regressor
    logging.info('Train {} start...'.format(reg_name))
    # Get best params
    regressor_param = getBestParams(score_name, reg_name)
    # Get regressor
    model = CombDrugModule.getRegressor(reg_name,
                                        regressor_params=regressor_param)
    ## Train model
    logging.info('Train...')
    model.fit(X_train, y_train)     
    
    ## Save model
    logging.info('Save {} model...'.format(reg_name))
    joblib.dump(model, os.path.join(DATA_DIR,
                                    "{}_{}_train_model".format(score_name,
                                                               reg_name)))

    logging.info("Done {} {}...".format(score_name, reg_name))
    
if __name__ == "__main__":
    for item in ["ZIP_XGBoost", "Bliss_XGBoost", 
                 "HSA_RandomForest", "Loewe_RandomForest"]:
        run2SaveTrainModel(item)
    
    logging.info("ALL DONE")