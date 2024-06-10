import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from transformers import LlamaTokenizer, LlamaModel
import numpy as np
import os

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download


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


path_models = "/lscratch/llama2/llama/"  ## Change the path to your own llama 2 models

str_model = "llama-2-7b-hf" #
## List of possible llama 2 models: 
#list_llama_models = [
    #'llama-2-7b-hf',
    #'llama-2-13b-hf',
    #'llama-2-70b-hf',
    #'llama-2-7b-chat-hf',
    #'llama-2-13b-chat-hf',
    #'llama-2-70b-chat-hf'
#]
name_model = path_models + str_model
print(name_model)


llama2_tokenizer = LlamaTokenizer.from_pretrained(name_model)
llama2_model = LlamaModel.from_pretrained(name_model, device_map='auto')

llama2_model.to(device)

list_data = []
for sent in sentences:
    inputs = llama2_tokenizer(sent, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = llama2_model(**inputs, return_dict=True)#[0]
        version1 = outputs.last_hidden_state[:,].to('cpu').detach().numpy().mean(axis=1)[0]
        print(np.shape(version1))
        list_data.append(version1.tolist())

df = pd.DataFrame(data=np.array(list_data))
df.to_csv(f"{DATA[index_data]}_{str_model}.csv", index=False, header=None)


print('+++++++++++++++')
print('The end')
print('+++++++++++++++')