import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from flask import Flask, render_template, request, json

from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools

app = Flask(__name__)
vae = VAEUtils(directory='/project/myflask/zinc_properties')
graph = tf.get_default_graph()

@app.route('/')
def initPage():
    # path
    path = '/datasets/DTS2020000038/data/'
    file_name = 'd01_te.dat'
    score_name = 'Fault1_SPE_Score.npy'
    xai_name = 'Fault1_SPE_Contribution.npy'

    # data load
    data = np.loadtxt(path + file_name)
    score = np.load(path + score_name)
    xai = np.load(path + xai_name)

    idx = np.hstack((np.arange(22), np.arange(41, 52)))
    data = json.dumps(stats.zscore(data[:, idx]).T.tolist())
    score = json.dumps(np.log(score).tolist())
    xai = json.dumps(xai.tolist())

    return render_template('index.html', data=data, score=score, xai=xai)

@app.route('/mi')
def viewMI():
    return render_template('mi.html')

@app.route('/mi/generate', methods=['POST'])
def generateMI():
    global graph
    with graph.as_default():
        # input smiles
        smiles_1 = mu.canon_smiles(request.form['smiles'])

        X_1 = vae.smiles_to_hot(smiles_1, canonize_smiles=True) # input hot
        z_1 = vae.encode(X_1) # latent
        y_1 = vae.predict_prop_Z(z_1)[0] # properties

        # generate molecules
        df = vae.z_to_smiles(z_1, decode_attempts=80, noise_norm=5)

        return json.dumps({'status': 1, 'pred': y_1.tolist(), 'smiles': df['smiles'].tolist(),
                           'dist': df['distance'].tolist(), 'freq': df['frequency'].tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=9101)