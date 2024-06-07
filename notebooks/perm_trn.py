# Files needed
# - Embeddings
# - CSV that contains:
#   - ID (group), features (y)
# - Config

# Packages needed to install: 
# pandas, numpy, torch, sklearn, itertools, joblib
# Python packages:
# pickle, json

import pandas as pd
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import itertools
from joblib import Parallel, delayed

import pickle
import json
import sys

# print(sys.argv, flush=True)
config_path = sys.argv[1]
print('Using config file', sys.argv[1])

# Config parameters
with open(config_path, 'r') as f:
    config = json.load(f)

np.random.seed(config['seed'])

write_dir = '/data/MLDSST/xinaw/impactme/results/raw/perm_trn'

from pathlib import Path
Path(write_dir).mkdir(parents=True, exist_ok=True)

###############################################################
# 0 Functions
###############################################################

from lr import fit_lr

# Helper function to pass to joblib.Parallel
# Parallelizing over params = (k_fold, C)

def train(params):
    y = lr['y']
    folds = lr['folds']
    i = params[0] # outer fold
    j = params[1] # inner fold
    
    outer_trn = [x for j, x in enumerate(folds) if j != i]
    outer_tst = folds[i]

    # Perform outer CV if j == -1
    if j == -1:
        outer_trn = np.hstack(outer_trn)
        df = fit_lr(
            embeds_np[outer_trn], np.random.permutation(y[outer_trn]), 
            embeds_np[outer_tst], y[outer_tst], 
            max_iter=config['max_iter'],
            seed=config['seed'],
            C=params[2]
        )
    else:
        inner_trn = [x for k, x in enumerate(outer_trn) if k != j]
        inner_trn = np.hstack(inner_trn)
        inner_tst = outer_trn[j]

        df = fit_lr(
            embeds_np[inner_trn], np.random.permutation(y[inner_trn]), 
            embeds_np[inner_tst], y[inner_tst], 
            max_iter=config['max_iter'],
            seed=config['seed'],
            C=params[2]
        )

    df['inner_fold'] = j
    df['outer_fold'] = i

    return df

###############################################################
# 1 Read data
###############################################################

# Store results here
lr = {}

# Read annotations
anno = pd.read_csv(f'/data/MLDSST/xinaw/impactme/data/trn/{config["data"]}.tsv', sep='\t')

# Read in y, ID, outputs
lr['y'] = anno[config['feature']].values
lr['subject'] = anno['subject'].values

# Read in embeddings
with open(
    f'/data/MLDSST/xinaw/impactme/data/trn/{config["model_f"]}-{config["embed"]}-{config["data"]}.t', 'rb'
) as f:
    embeds = torch.load(f, map_location=torch.device('cpu'))

embeds_np = embeds.cpu().numpy()

###############################################################
# 2 Training
###############################################################

# Make k-fold splits
sgkfold = StratifiedGroupKFold(
    n_splits=config['k_fold'], shuffle=False
)

y = lr['y']
lr['folds'] = [
    tst for trn, tst in sgkfold.split(np.zeros(len(y)), y, groups=lr['subject'])
]

# Parallelize over outer fold, inner fold, and C
params = list(
    itertools.product(
        range(config['k_fold']), # outer_fold
        range(-1, config['k_fold'] - 1), # inner_fold
        config['C_list']
    )
)

out = Parallel(n_jobs=config['n_parallel'])(
    delayed(train)(i) for i in params
)

###############################################################
# 3 Post-processing
###############################################################

df = pd.concat(out)
df.insert(0, 'data', config['data'])
df.insert(0, 'feature', config['feature'])
lr['outputs'] = df

# Save results
# Output should contain, in order, model_f, data, and feature
with open(f'{write_dir}/{config["model_f"]}-{config["data"]}-{config["feature"]}.pkl', 'wb') as f:
    pickle.dump(lr, f)
