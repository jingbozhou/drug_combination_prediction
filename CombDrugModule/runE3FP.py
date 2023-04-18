import os, pickle, sys
import numpy as np
import pandas as pd
from itertools import islice
from e3fp.conformer.util import smiles_to_dict
from e3fp.config.params import default_params, read_params
from e3fp.pipeline import fprints_from_smiles, confs_from_smiles, params_to_dicts
from python_utilities.parallel import Parallelizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD_DIR = os.path.join(BASE_DIR, "CombDrugModule/")
TMP_DIR = os.path.join(BASE_DIR, "tmp/")

confgen_params, fprint_params = params_to_dicts(default_params)
del confgen_params['protonate']
del confgen_params['standardise']

fprint_params['include_disconnected'] = True
fprint_params['stereo'] = False
fprint_params['first'] = 1
confgen_params['first'] = 20

kwargs = {"confgen_params": confgen_params, 
          "fprint_params": fprint_params
         }

with open(os.path.join(TMP_DIR, "cid2smiles_{}.pickle".format(sys.argv[1])), "rb") as f:
    cid_can = pickle.load(f)
    
cid_can = cid_can["canonical_smiles"].to_dict()
        
parallelizer = Parallelizer(parallel_mode='processes', num_proc = int(sys.argv[2]))

smiles_iter = ((smiles, name) for name, smiles in cid_can.items())
fprints_list = parallelizer.run(fprints_from_smiles, smiles_iter, kwargs=kwargs)

# they are all equal. so take first element
len_bitness = fprints_list[0][0][0].bits

df = pd.DataFrame(np.zeros((len(fprints_list), len_bitness), 
                           dtype=np.int16), 
                  index=list(cid_can.keys()), 
                  columns=["E3FP_{}".format(x) for x in range(1, len_bitness+1)])


for item in fprints_list:
    cid_name = item[1][1]
    if item[0]:
        df.loc[cid_name] = item[0][0].to_vector(sparse=False, dtype=np.ndarray)

with open(os.path.join(TMP_DIR, "drug_smiles_E3FP_{}.pickle".format(sys.argv[1])), 'wb') as f:
    pickle.dump(df, f)

