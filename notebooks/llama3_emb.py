#####################
#####################
## This file contains the code 
## to do the embedding using LLaMA3
#####################
#####################


import torch
import transformers
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from transformers import LlamaTokenizer, LlamaModel
from transformers import PreTrainedTokenizerFast, LlamaTokenizerFast
import numpy as np
import os


DATA = [
    'all',
    # 'pt_noshort',
    # 'turns'
]
index_data = 0

path_dataset = '/Datasets/impactme/trn/new_parsed/' ## Change the path to your own dataset

text_anno = pd.read_csv(f'{path_dataset}{DATA[index_data]}.tsv', sep='\t')
sentences = text_anno.text.to_list()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Configure more GPUs if models larger than 8B will be used.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#
device = torch.device('cuda')
#
print('We will use the GPU:', os.environ["CUDA_VISIBLE_DEVICES"])
print('+++++++++++++++')


## LLaMA3
path_models = '/lscratch2/llama3_hf/'  ## Change the path to your own llama3 models

## Models
str_model = "Meta-Llama-3-8B"
# Model available: Meta-Llama-3-8B,  Meta-Llama-3-8B-Instruct,   Meta-Llama-3-70B,   Meta-Llama-3-70B-Instruct


name_model = path_models + str_model
print('\n+++++++++++++++')
print(name_model)
print('+++++++++++++++\n')


llama3_tokenizer_fast = PreTrainedTokenizerFast.from_pretrained(name_model)
llama3_model = LlamaModel.from_pretrained(name_model, torch_dtype=torch.float16, device_map='auto')


llama3_model.to(device)

list_data = []
for sent in sentences:
    inputs = llama3_tokenizer_fast(sent, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = llama3_model(**inputs, return_dict=True)#[0]
        version1 = outputs.last_hidden_state[:,].to('cpu').detach().numpy().mean(axis=1)[0]
        list_data.append(version1.tolist())

df = pd.DataFrame(data=np.array(list_data))
df.to_csv(f"{str_model}_{DATA[index_data]}.csv", index=False, header=None)


print('+++++++++++++++')
print('The end')
print('+++++++++++++++')