# Generate embeddings from HF models
# Used to preload embeddings for use in baseline logisitic regression models

# Recommended to run in technetium rather than curium (runs much faster)

from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd

#######################################################################
# 0 Parameters
#######################################################################

# Data parameters
embed = 'last_avg'
datas = [
    'all', 
    'pt_noshort', 
    # 'pt_noshort_speakerturns',
    # 'speaker_turns',
    'turns'
]
splits = [
    'trn', 
    # 'tst'
]
text_col = 'text' # The utterance column

# AutoModel compatible
models = [
    'bert-base-uncased',
    'roberta-base',
    'mental/mental-bert-base-uncased'
]

fnames = [
    'bert_base_uncased',
    'roberta',
    'mental_bert'
]

#######################################################################
# 1 Functions
#######################################################################

# Returns the mean or sum of the last hidden state
def get_embeds(sent, avg=True):
    encoded = tokenizer.__call__(sent, return_tensors="pt", truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    embedding = outputs.last_hidden_state[0]
    
    if avg: 
        return torch.mean(embedding, 0)
    else:
        return torch.sum(embedding, 0)

#######################################################################
# 2 Generate embeds
#######################################################################

# Look for 00B for longformer compatible models
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Loop over models
for i, m in enumerate(models):
    fname = fnames[i]
    model = AutoModel.from_pretrained(m).to(device)
    tokenizer = AutoTokenizer.from_pretrained(m)

    print(f'Loading {m}...')

    # Loop over data subsets and stages
    for t in splits:
        for d in datas:
            print(f'Processing {t}, {d}...')
            df = pd.read_csv(f'data/{t}/{d}.tsv', sep='\t')
            embeds = [get_embeds(s) for s in df[text_col].values]
            embeds = torch.stack(embeds)

            with open(f'data/{t}/{fname}-{embed}-{d}.t','wb') as f:
                torch.save(embeds, f)
            
            print(f'Saved {fname}-{embed}-{d}.t')