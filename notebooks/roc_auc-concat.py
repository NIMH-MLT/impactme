# Simple variant on ROC AUC that handles outputs from best_C.py

import pandas as pd
import numpy as np

# ROC AUC calculations
from sklearn.metrics import roc_auc_score

# File handling
import json
import sys

# Additional handling of C values
import itertools

from pathlib import Path

# For records
import time

###############################################################
# 0 Read the data
###############################################################

start_time = time.time()

#==============================================================
# 0.1 Read config params
#==============================================================

# print(sys.argv, flush=True)
config_path = sys.argv[1]
print('Using config file', sys.argv[1])

# Config parameters
with open(config_path, 'r') as f:
    config = json.load(f)

models = config['models']
data = config['data']
feature = config['feature']
trn_types = config['trn_types']

#==============================================================
# 0.2 Read concatenated probabilities
#==============================================================

concat = pd.read_csv('/data/MLDSST/xinaw/impactme/results/tuned/concat.csv')
concat = concat[
    (concat['data'] == data) &
    (concat['feature'] == feature)
]

###############################################################
# 1 Calculate the ROC AUC
###############################################################

params = list(itertools.product(models, trn_types))
roc_auc_df = pd.DataFrame(
    list(itertools.product(models, trn_types)), 
    columns = ['model', 'trn_type']
)
roc_auc_df['data'] = data
roc_auc_df['feature'] = feature
roc_aucs = []

for model, trn_type in params:
    df = concat[
        (concat['model'] == model) &
        (concat['trn_type'] == trn_type)
    ]

    try:
        roc_auc = roc_auc_score(df['y_true'], df['y_prob'])
    except:
        # NaN is harder to accidentally include than 0
        roc_auc = np.nan

    roc_aucs.append(roc_auc)

roc_auc_df['roc_auc'] = roc_aucs

write_dir = Path('/data/MLDSST/xinaw/impactme/results/tuned/roc_auc-concat')
Path(write_dir).mkdir(parents=True, exist_ok=True)
roc_auc_df.to_csv(write_dir / f'{data}-{feature}.csv', index=False)

end_time = time.time()
print(f'{end_time - start_time} s elapsed.')