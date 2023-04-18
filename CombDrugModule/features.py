import sys, os, csv, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit.Chem.Descriptors import _descList
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from mordred import Calculator, descriptors, is_missing

from .data import TMP_DIR, DATA_DIR

def runForMordred(random_num):
    
    with open(os.path.join(TMP_DIR, "cid2smiles_{}.pickle".format(random_num)), "rb") as f:
        cid_can = pickle.load(f)
        
    cid_can = cid_can["canonical_smiles"].to_dict()
    
    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors, ignore_3D=False)
    
    # calculate multiple molecule
    cid_list = [x for x in cid_can]
    mols = [Chem.MolFromSmiles(cid_can[cid]) for cid in cid_list]
    
    # as pandas
    df = calc.pandas(mols)
    
    df["Drug_cid"] = cid_list
    df = df.set_index("Drug_cid")
    df.index.name = ""
    
    for ind in df.index:
        for col in df.columns:
            if is_missing(df.loc[ind, col]):
                df.loc[ind, col] = 0
    
    with open(os.path.join(TMP_DIR, "drug_smiles_Mordred_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(df, f)
        
        

def runForRDKit(random_num):
    
    with open(os.path.join(TMP_DIR, "cid2smiles_{}.pickle".format(random_num)), "rb") as f:
        cid_can = pickle.load(f)
    
    cid_can = cid_can["canonical_smiles"].to_dict()

    num_mols = len(cid_can)
    maccs_name = ["MACCS_{}".format(i) for i in range(1, 167)]
    estate_name = ["EState_{}".format(i) for i in range(1, 80)]
    descriptor_list = _descList
    descriptor_names = [descriptor_list[i][0] for i in range(len(descriptor_list))]
    mdc = MolecularDescriptorCalculator(simpleList=descriptor_names)
    
    raw_df = pd.DataFrame(0, index=list(cid_can.keys()), columns=maccs_name+estate_name+descriptor_names)
    for index in raw_df.index:
        m = Chem.MolFromSmiles(cid_can[index])
        raw_df.loc[index, maccs_name] = np.array(MACCSkeys.GenMACCSKeys(m))[1:]
        raw_df.loc[index, estate_name] = FingerprintMol(m)[0]
        raw_df.loc[index, descriptor_names] = np.array(mdc.CalcDescriptors(m))
    
    with open(os.path.join(TMP_DIR, "drug_smiles_RDKit_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(raw_df, f)

def runGeneNew(input_list, random_num):
    gene_expression = pd.read_pickle(os.path.join(DATA_DIR, "gene_expression.pickle"),
                                     compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
    
    with open(os.path.join(TMP_DIR, "drug_smiles_gene_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(gene_expression.loc[input_list], f)
        
def runMutNew(input_list, random_num):
    gene_mut = pd.read_pickle(os.path.join(DATA_DIR, "gene_mut.pickle"),
                                     compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
    
    with open(os.path.join(TMP_DIR, "drug_smiles_mut_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(gene_mut.loc[input_list], f)
        
def runCNVNew(input_list, random_num):
    gene_cnv = pd.read_pickle(os.path.join(DATA_DIR, "gene_cnv.pickle"),
                              compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
    
    with open(os.path.join(TMP_DIR, "drug_smiles_cnv_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(gene_cnv.loc[input_list], f)

def runGeneRaw(input_list, random_num):
    
    gene_dir = "/home/jingbozhou/Project/CombDrug/geneFeature/"
    gene_expression = pd.read_csv(os.path.join(gene_dir, "gene_expression.tsv"),
                                  sep="\t", index_col=0)
    
    with open(os.path.join(TMP_DIR, "drug_smiles_gene_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(gene_expression.loc[input_list], f)
        
def runGeneAll(input_list, random_num):
    
    gene_dir = "/home/jingbozhou/Project/CombDrug/geneFeature/"
    gene_expression = pd.read_csv(os.path.join(gene_dir, "all_gene_expression_sub.tsv"),
                                  sep="\t", index_col=0)
    
    with open(os.path.join(TMP_DIR, "drug_smiles_gene_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(gene_expression.loc[input_list], f)
        
def runMut(input_list, random_num):
    mut_dir = "/home/jingbozhou/Project/CombDrug/mutFeature/"
    gene_mut = pd.read_csv(os.path.join(mut_dir, "gene_mut.tsv"),
                           sep="\t", index_col=0)
    
    with open(os.path.join(TMP_DIR, "drug_smiles_mut_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(gene_mut.loc[input_list], f)
        
def runCNV(input_list, random_num):
    mut_dir = "/home/jingbozhou/Project/CombDrug/mutFeature/"
    gene_cnv = pd.read_csv(os.path.join(mut_dir, "gene_cnv.tsv"),
                           sep="\t", index_col=0)
    
    with open(os.path.join(TMP_DIR, "drug_smiles_cnv_{}.pickle".format(random_num)), 'wb') as f:
        pickle.dump(gene_cnv.loc[input_list], f)
        
def getInfomaxFeature(raw_drug_df, random_num):
    # Load infomax features
    with open(os.path.join(TMP_DIR, "drug_smiles_Infomax_{}.pickle".format(random_num)), 
              'rb') as f:
        drug_infomax = pickle.load(f)
    
    # Get columns name
    cols_name = ["DrugComb_id"]+\
    ["feature_row_Infomax_{}".format(x) for x in range(1, drug_infomax.shape[1]+1)]+\
    ["feature_col_Infomax_{}".format(x) for x in range(1, drug_infomax.shape[1]+1)]
    # Get values
    write_list = []
    for index in raw_drug_df.index:
        drug_row_cid = str(raw_drug_df.loc[index, "Drug_row_cid"])
        drug_col_cid = str(raw_drug_df.loc[index, "Drug_col_cid"])

        row_infomax = [x for x in drug_infomax.loc[drug_row_cid].values]
        col_infomax = [x for x in drug_infomax.loc[drug_col_cid].values]
        
        write_list.append([index]+row_infomax+col_infomax)
    
    infomax_df = pd.DataFrame(write_list, columns=cols_name)
    infomax_df.set_index("DrugComb_id", inplace=True)
    
    return infomax_df

def getMordredFeature(raw_drug_df, random_num):
    # Load Mordred features
    with open(os.path.join(TMP_DIR, "drug_smiles_Mordred_{}.pickle".format(random_num)), 'rb') as f:
        drug_Mordred = pickle.load(f)
    
    # Get columns name
    cols_name = ["DrugComb_id"]+\
    ["feature_row_Mordred_{}".format(x) for x in drug_Mordred.columns]+\
    ["feature_col_Mordred_{}".format(x) for x in drug_Mordred.columns]
    # Get values
    write_list = []
    for index in raw_drug_df.index:
        drug_row_cid = str(raw_drug_df.loc[index, "Drug_row_cid"])
        drug_col_cid = str(raw_drug_df.loc[index, "Drug_col_cid"])
    
        row_Mordred = [x for x in drug_Mordred.loc[drug_row_cid].values]
        col_Mordred = [x for x in drug_Mordred.loc[drug_col_cid].values]
        
        write_list.append([index]+row_Mordred+col_Mordred)
        
    Mordred_df = pd.DataFrame(write_list, columns=cols_name)
    Mordred_df.set_index("DrugComb_id", inplace=True)
    
    return Mordred_df

def getRDKitFeature(raw_drug_df, random_num):
    # Load RDKit features
    with open(os.path.join(TMP_DIR, "drug_smiles_RDKit_{}.pickle".format(random_num)), 'rb') as f:
        drug_RDKit = pickle.load(f)
    
    # Get columns name
    cols_name = ["DrugComb_id"]+\
    ["feature_row_RDKit_{}".format(x) for x in drug_RDKit.columns]+\
    ["feature_col_RDKit_{}".format(x) for x in drug_RDKit.columns]
    # Get values
    write_list = []
    for index in raw_drug_df.index:
        drug_row_cid = str(raw_drug_df.loc[index, "Drug_row_cid"])
        drug_col_cid = str(raw_drug_df.loc[index, "Drug_col_cid"])

        row_RDKit = [x for x in drug_RDKit.loc[drug_row_cid].values]
        col_RDKit = [x for x in drug_RDKit.loc[drug_col_cid].values]
        
        write_list.append([index]+row_RDKit+col_RDKit)
    
    RDKit_df = pd.DataFrame(write_list, columns=cols_name)
    RDKit_df.set_index("DrugComb_id", inplace=True)
    
    return RDKit_df

def getgeneRawFeature(raw_drug_df, random_num):
    # Load gene expression data
    with open(os.path.join(TMP_DIR, "drug_smiles_gene_{}.pickle".format(random_num)), 'rb') as f:
        gene_expression = pickle.load(f)
    
    # Get columns name
    cols_name = ["DrugComb_id"] +\
    ["feature_gene_{}".format(x) for x in gene_expression.columns]
    # Get values
    write_list = []
    for index in raw_drug_df.index:
        cell_line_id = raw_drug_df.loc[index, "DepMap_ID"]
        cell_line_value = [x for x in gene_expression.loc[cell_line_id].values]
        write_list.append([index]+cell_line_value)
        
    gene_df = pd.DataFrame(write_list, columns=cols_name)
    gene_df.set_index("DrugComb_id", inplace=True)
    
    return gene_df

def getMutFeature(raw_drug_df, random_num):
    # Load gene mutation data
    with open(os.path.join(TMP_DIR, "drug_smiles_mut_{}.pickle".format(random_num)), 'rb') as f:
        gene_mut = pickle.load(f)
    
    # Get columns name
    cols_name = ["DrugComb_id"] +\
    ["feature_mut_{}".format(x) for x in gene_mut.columns]
    # Get values
    write_list = []
    for index in raw_drug_df.index:
        cell_line_id = raw_drug_df.loc[index, "DepMap_ID"]
        cell_line_value = [x for x in gene_mut.loc[cell_line_id].values]
        write_list.append([index]+cell_line_value)
        
    mut_df = pd.DataFrame(write_list, columns=cols_name)
    mut_df.set_index("DrugComb_id", inplace=True)
    
    return mut_df

def getCNVFeature(raw_drug_df, random_num):
    # Load gene cnv data
    with open(os.path.join(TMP_DIR, "drug_smiles_cnv_{}.pickle".format(random_num)), 'rb') as f:
        gene_cnv = pickle.load(f)
    
    # Get columns name
    cols_name = ["DrugComb_id"] +\
    ["feature_cnv_{}".format(x) for x in gene_cnv.columns]
    # Get values
    write_list = []
    for index in raw_drug_df.index:
        cell_line_id = raw_drug_df.loc[index, "DepMap_ID"]
        cell_line_value = [x for x in gene_cnv.loc[cell_line_id].values]
        write_list.append([index]+cell_line_value)
        
    cnv_df = pd.DataFrame(write_list, columns=cols_name)
    cnv_df.set_index("DrugComb_id", inplace=True)
    
    return cnv_df

def getInfomaxFeatureNew(raw_drug_df, random_num, features=False):
    # Load infomax features
    with open(os.path.join(TMP_DIR, "drug_smiles_Infomax_{}.pickle".format(random_num)), 
              'rb') as f:
        drug_infomax = pickle.load(f)
    
    if features:
        ## Run select features
        row_feats = [x for x in features if x.startswith("feature_row_Infomax")]
        col_feats = [x for x in features if x.startswith("feature_col_Infomax")]
        
    else:
        ## Run all features
        row_feats = ["feature_row_Infomax_{}".format(x) for x in range(1, drug_infomax.shape[1]+1)]
        col_feats = ["feature_col_Infomax_{}".format(x) for x in range(1, drug_infomax.shape[1]+1)]

    ## Get values
    write_list = []
    for index in raw_drug_df.index:
        drug_row_cid = str(raw_drug_df.loc[index, "Drug_row_cid"])
        drug_col_cid = str(raw_drug_df.loc[index, "Drug_col_cid"])

        row_infomax = [x for x in drug_infomax.loc[drug_row_cid, [x[12:] for x in row_feats]].values]
        col_infomax = [x for x in drug_infomax.loc[drug_col_cid, [x[12:] for x in col_feats]].values]
        
        write_list.append([index]+row_infomax+col_infomax)
    
    ## Get columns name
    cols_name = ["DrugComb_id"] + row_feats + col_feats

    infomax_df = pd.DataFrame(write_list, columns=cols_name)
    
    del write_list
    
    infomax_df.set_index("DrugComb_id", inplace=True)
    
    return infomax_df

def getMordredFeatureNew(raw_drug_df, random_num, features=False):
    # Load Mordred features
    with open(os.path.join(TMP_DIR, "drug_smiles_Mordred_{}.pickle".format(random_num)), 'rb') as f:
        drug_Mordred = pickle.load(f)
    
    if features:
        ## Run select features
        row_feats = [x for x in features if x.startswith("feature_row_Mordred")]
        col_feats = [x for x in features if x.startswith("feature_col_Mordred")]
        
    else:
        ## Run all features
        row_feats = ["feature_row_Mordred_{}".format(x) for x in drug_Mordred.columns]
        col_feats = ["feature_col_Mordred_{}".format(x) for x in drug_Mordred.columns]
    
    ## Get values
    write_list = []
    for index in raw_drug_df.index:
        drug_row_cid = str(raw_drug_df.loc[index, "Drug_row_cid"])
        drug_col_cid = str(raw_drug_df.loc[index, "Drug_col_cid"])
    
        row_Mordred = [x for x in drug_Mordred.loc[drug_row_cid, [x[20:] for x in row_feats]].values]
        col_Mordred = [x for x in drug_Mordred.loc[drug_col_cid, [x[20:] for x in col_feats]].values]
        
        write_list.append([index]+row_Mordred+col_Mordred)
        
    ## Get columns name
    cols_name = ["DrugComb_id"] + row_feats + col_feats
    
    Mordred_df = pd.DataFrame(write_list, columns=cols_name)
    
    del write_list
    
    Mordred_df.set_index("DrugComb_id", inplace=True)
    
    return Mordred_df

def getRDKitFeatureNew(raw_drug_df, random_num, features=False):
    # Load RDKit features
    with open(os.path.join(TMP_DIR, "drug_smiles_RDKit_{}.pickle".format(random_num)), 'rb') as f:
        drug_RDKit = pickle.load(f)
        
    if features:
        ## Run select features
        row_feats = [x for x in features if x.startswith("feature_row_RDKit")]
        col_feats = [x for x in features if x.startswith("feature_col_RDKit")]
    else:
        ## Run all features
        row_feats = ["feature_row_RDKit_{}".format(x) for x in drug_RDKit.columns]
        col_feats = ["feature_col_RDKit_{}".format(x) for x in drug_RDKit.columns]

    ## Get values
    write_list = []
    for index in raw_drug_df.index:
        drug_row_cid = str(raw_drug_df.loc[index, "Drug_row_cid"])
        drug_col_cid = str(raw_drug_df.loc[index, "Drug_col_cid"])

        row_RDKit = [x for x in drug_RDKit.loc[drug_row_cid, [x[18:] for x in row_feats]].values]
        col_RDKit = [x for x in drug_RDKit.loc[drug_col_cid, [x[18:] for x in col_feats]].values]
        
        write_list.append([index]+row_RDKit+col_RDKit)
    
    ## Get columns name
    cols_name = ["DrugComb_id"]+ row_feats + col_feats
    
    RDKit_df = pd.DataFrame(write_list, columns=cols_name)
    
    del write_list 
    
    RDKit_df.set_index("DrugComb_id", inplace=True)
    
    return RDKit_df

def getgeneRawFeatureNew(raw_drug_df, random_num, features=False):
    # Load gene expression data
    with open(os.path.join(TMP_DIR, "drug_smiles_gene_{}.pickle".format(random_num)), 'rb') as f:
        gene_expression = pickle.load(f)
        
    if features:
        ## Run select features
        gene_feats = [x for x in features if x.startswith("feature_gene")]
        
    else:
        ## Run all features
        gene_feats = ["feature_gene_{}".format(x) for x in gene_expression.columns]
        
    ## Get values
    write_list = []
    for index in raw_drug_df.index:
        cell_line_id = raw_drug_df.loc[index, "DepMap_ID"]
        cell_line_value = [x for x in gene_expression.loc[cell_line_id, [x[13:] for x in gene_feats]].values]
        write_list.append([index]+cell_line_value)
        
    # Get columns name
    cols_name = ["DrugComb_id"] + gene_feats
        
    gene_df = pd.DataFrame(write_list, columns=cols_name)
    
    del write_list
    
    gene_df.set_index("DrugComb_id", inplace=True)
    
    return gene_df

def getMutFeatureNew(raw_drug_df, random_num, features=False):
    # Load gene mutation data
    with open(os.path.join(TMP_DIR, "drug_smiles_mut_{}.pickle".format(random_num)), 'rb') as f:
        gene_mut = pickle.load(f)
    
    if features:
        ## Run select features
        mut_feats = [x for x in features if x.startswith("feature_mut")]
    else:
        ## Run all features
        mut_feats = ["feature_mut_{}".format(x) for x in gene_mut.columns]
    
    ## Get values
    write_list = []
    for index in raw_drug_df.index:
        cell_line_id = raw_drug_df.loc[index, "DepMap_ID"]
        cell_line_value = [x for x in gene_mut.loc[cell_line_id, [x[12:] for x in mut_feats]].values]
        write_list.append([index]+cell_line_value)
        
    ## Get columns name
    cols_name = ["DrugComb_id"] + mut_feats
    
    mut_df = pd.DataFrame(write_list, columns=cols_name)
    
    del write_list
    
    mut_df.set_index("DrugComb_id", inplace=True)
    
    return mut_df

def getCNVFeatureNew(raw_drug_df, random_num, features=False):
    # Load gene cnv data
    with open(os.path.join(TMP_DIR, "drug_smiles_cnv_{}.pickle".format(random_num)), 'rb') as f:
        gene_cnv = pickle.load(f)
    
    if features:
        ## Run select features
        cnv_feats = [x for x in features if x.startswith("feature_cnv")]
    else:
        ## Run all features
        cnv_feats = ["feature_cnv_{}".format(x) for x in gene_cnv.columns]
    
    ## Get values
    write_list = []
    for index in raw_drug_df.index:
        cell_line_id = raw_drug_df.loc[index, "DepMap_ID"]
        cell_line_value = [x for x in gene_cnv.loc[cell_line_id, [x[12:] for x in cnv_feats]].values]
        write_list.append([index]+cell_line_value)
        
    # Get columns name
    cols_name = ["DrugComb_id"] + cnv_feats
    
    cnv_df = pd.DataFrame(write_list, columns=cols_name)
    
    del write_list
    
    cnv_df.set_index("DrugComb_id", inplace=True)
    
    return cnv_df

