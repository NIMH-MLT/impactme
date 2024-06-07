import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import itertools
from tqdm import tqdm
from os.path import exists

models = [
    # 'bert_base_uncased', 
    # 'roberta',
    # 'mental_bert', 
    'mental_longformer'
]
datas = [
    'all', 
    'pt_noshort', 
    'pt_noshort_speakerturns',
    'speaker_turns',
    'turns'
]
# A8 commented out because it throws errors for any combination of folds
features = [
    'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', # 'a8', 
    # 'b1', 'b2', 'b3', 
    # 'c1', 'c2', 'c3', 'c4', 
    # 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
    # 'e1', 'e2', 'e3', 'e4', # 'e5', 
    # 'g1', 'g2', 
    # 'f1', 'f2', 'f3', 'f',
    'a', 'b', 'c', 'd', 'e', 'g', 
    'any'
]
k_fold = 4
C_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100., 500., 1000., 5000.]

results_dir = 'results/roc_auc'

params = list(
    itertools.product(
        models,
        datas, 
        features,
        range(k_fold),
        C_list
    )
)

print('Starting calculations...')

for model in models:
    print(f'Starting calculations for {model}.')

    for data in datas:
        t = tqdm(total = len(features) * k_fold * len(C_list), desc=data)

        for feature in features:
            
            if exists(f'{results_dir}/{model}-{data}-{feature}.csv'):
                t.update(k_fold * len(C_list))
                continue

            try:
                with open(f'results/raw/{model}-{data}-{feature}.pkl', 'rb') as f:
                    res = pickle.load(f)
            except:
                print(f'Cannot read {model} {data} {feature}')
                t.update(k_fold * len(C_list))
                continue

            params = list(
                itertools.product(
                    range(k_fold),
                    C_list
                )
            )

            outer_list = []

            for p in params:

                t.update(1)

                i = p[0] # outer fold
                C = p[1]

                df = res['outputs']
                res_list = []

                out_df = df[df['outer_fold'] == i]
                out_df = out_df[out_df['C'] == C]
                        
                inner_list = []

                for j in range(k_fold):
                    in_df = out_df[out_df['inner_fold'] == j - 1]
                    try:
                        roc_auc = roc_auc_score(in_df['y_true'], in_df['y_prob'])
                    except:
                        roc_auc = np.nan

                    inner_list.append(
                        pd.DataFrame({
                            'inner_fold': [j - 1], 
                            'roc_auc': [roc_auc]        
                        })
                    )

                df = pd.concat(inner_list, ignore_index=True)
                df['outer_fold'] = i
                df['C'] = C

                outer_list.append(df)

            outer_df = pd.concat(outer_list, ignore_index=True)
            outer_df.to_csv(f'{results_dir}/{model}-{data}-{feature}.csv', index=False)