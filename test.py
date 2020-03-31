import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from scipy import stats
from flask import Flask, render_template, request, json

from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools

vae = VAEUtils(directory='/project/myflask/zinc_properties')

smiles_1 = mu.canon_smiles('CSCC(=O)NNC(=O)c1c(C)oc(C)c1C')
X_1 = vae.smiles_to_hot(smiles_1, canonize_smiles=True) # input hot
z_1 = vae.encode(X_1) # latent
y_1 = vae.predict_prop_Z(z_1)[0] # properties
print(y_1)
