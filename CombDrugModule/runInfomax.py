import dgl
from dgllife.model import load_pretrained
from rdkit import Chem
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from torch.utils.data import DataLoader
from dgl.nn.pytorch.glob import AvgPooling
from collections import defaultdict
from dgllife.model.model_zoo import *
from dgllife.model.pretrain.property_prediction import create_property_model
    
import numpy as np
import pandas as pd
import os, pickle, sys
    
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD_DIR = os.path.join(BASE_DIR, "CombDrugModule/")
TMP_DIR = os.path.join(BASE_DIR, "tmp/")

def collate(gs):
    return dgl.batch(gs)


model = create_property_model("gin_supervised_infomax")
checkpoint = torch.load(os.path.join(MOD_DIR, "gin_supervised_infomax_pre_trained.pth"), 
                        map_location='cpu')
try:
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    model.load_state_dict(checkpoint)
# model = load_pretrained('gin_supervised_infomax') # contextpred infomax edgepred masking
model.to('cpu')
model.eval()

with open(os.path.join(TMP_DIR, "cid2smiles_{}.pickle".format(sys.argv[1])), "rb") as f:
    cid_can = pickle.load(f)

graphs = []
for smi in cid_can["canonical_smiles"].values:
    mol = Chem.MolFromSmiles(smi)
    g = mol_to_bigraph(mol, add_self_loop=True,
                       node_featurizer=PretrainAtomFeaturizer(),
                       edge_featurizer=PretrainBondFeaturizer(),
                       canonical_atom_order=True)
    graphs.append(g)
    
data_loader = DataLoader(graphs, batch_size=256, collate_fn=collate, shuffle=False)

readout = AvgPooling()

mol_emb = []
for batch_id, bg in enumerate(data_loader):
    bg = bg.to('cpu')
    nfeats = [bg.ndata.pop('atomic_number').to('cpu'),
              bg.ndata.pop('chirality_type').to('cpu')]
    efeats = [bg.edata.pop('bond_type').to('cpu'),
              bg.edata.pop('bond_direction_type').to('cpu')]
    with torch.no_grad():
        node_repr = model(bg, nfeats, efeats)
    mol_emb.append(readout(bg, node_repr))

mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()

fps_infomax_can = pd.DataFrame(data=mol_emb, index=cid_can.index)
fps_infomax_can.columns = ["Infomax_{}".format(x) for x in range(1, 301)]

with open(os.path.join(TMP_DIR, "drug_smiles_Infomax_{}.pickle".format(sys.argv[1])), 'wb') as f:
    pickle.dump(fps_infomax_can, f)
