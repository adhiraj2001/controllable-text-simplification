#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq Text Summarization Model

# In[1]:


import csv
import string

import numpy as np
import pandas as pd

from tqdm import tqdm


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


import wandb
# wandb.login(key='913841cb22c908099db4951c258f4242c1d1b7aa')

import os
os.environ['WANDB_API_KEY'] = '913841cb22c908099db4951c258f4242c1d1b7aa'
os.environ['WANDB_SILENT'] = 'true'

## To Avoid deadlocks ?
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# In[ ]:


abs_root = '/ssd_scratch/cvit/adhiraj_deshmukh'
abs_code = f'{abs_root}/ANLP-Project'
abs_data = f'{abs_code}/data'


# In[ ]:





# ## 1. Import Dataset

# In[ ]:


colnames = ['source', 'target']

train = pd.read_csv(f"{abs_data}/train.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)
val = pd.read_csv(f"{abs_data}/valid.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)


# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(train, test_size=0.2, random_state=42)
#train, val = train_test_split(train, test_size=0.2, random_state=42)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)


# In[ ]:


#save train, val, test
train.to_csv(f'{abs_data}/train.csv', index=False)
val.to_csv(f'{abs_data}/val.csv', index=False)
test.to_csv(f'{abs_data}/test.csv', index=False)


# In[ ]:





# In[ ]:


from datasets import load_dataset

# train = load_dataset('csv', data_files=f'{abs_data}/train.csv',cache_dir=f'{abs_root}/bart_cnn_data')
# val = load_dataset('csv', data_files=f'{abs_data}/val.csv',cache_dir=f'{abs_root}/bart_cnn_data')
# test = load_dataset('csv', data_files=f'{abs_data}/test.csv',cache_dir=f'{abs_root}/bart_cnn_data')

train = load_dataset('csv', data_files=f'{abs_data}/train.csv')
val = load_dataset('csv', data_files=f'{abs_data}/val.csv')
test = load_dataset('csv', data_files=f'{abs_data}/test.csv')


# In[ ]:





# In[ ]:


train["validation"] = val["train"]
train["test"] = test["train"]


# In[ ]:


train["train"] = train["train"].shuffle().select(range(100000))
train["validation"] = train["validation"].shuffle().select(range(1000))
train["test"] = train["test"].shuffle().select(range(1000))


# In[ ]:





# ### 2. Tokenize and Load Data

# In[ ]:


from transformers import T5TokenizerFast # 6x Speedup

tokenizer = T5TokenizerFast.from_pretrained('t5-small', cache_dir=f'{abs_root}/hf_cache')


# In[ ]:


prefix = "Summarize:"

max_input_length = 512
max_target_length = 64
batch_size = 8 # [4, 8, 16]


# In[ ]:


import nltk

def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
    
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned


# In[ ]:


# Define the function to make the correct data structure
def process_data_to_model_inputs(batch):
    inputs = [f'{prefix} \"{clean_text(text)}\"' for text in batch["source"]]
    
    model_inputs = tokenizer(inputs, padding="max_length", max_length=max_input_length, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        outputs = [clean_text(text) for text in batch["target"]]
        model_outputs = tokenizer(outputs, padding="max_length", max_length=max_target_length, truncation=True)
    
    batch["input_ids"] = model_inputs.input_ids
    batch["attention_mask"] = model_inputs.attention_mask
    
    batch["decoder_input_ids"] = model_outputs.input_ids
    batch["decoder_attention_mask"] = model_outputs.attention_mask
    
    batch["labels"] = model_outputs.input_ids.copy()
    
    # We have to make sure that the PAD token is ignored for calculating the loss
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    
    return batch


# In[ ]:


# Map the function to both train/validation sets.
train = train.map(
    process_data_to_model_inputs, 
    batched=True,
    remove_columns=["source", "target"], 
    batch_size = 1024,
)


# In[ ]:


# Convert the Dataset to PyTorch tensor with the expected columns
train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                           "decoder_attention_mask", "labels"],
)


# In[ ]:


from torch.utils.data import DataLoader

# Make the iterative object that does batching using the DataLoader
train_dl = DataLoader(train["train"], batch_size=batch_size, shuffle=True)
val_dl = DataLoader(train["validation"], batch_size=batch_size, shuffle=True)


# In[ ]:





# ### 3. Load Pre-trained Model

# In[ ]:


from transformers import T5ForConditionalGeneration
import torch

# Load the model
model = T5ForConditionalGeneration.from_pretrained(f"t5-small", cache_dir=f'{abs_root}/hf_cache')


# In[ ]:


# model.resize_token_embeddings(len(tokenizer))


# In[ ]:


# # Split model's components
# the_encoder = model.get_encoder()
# the_decoder = model.get_decoder()

# last_linear_layer = model.lm_head


# In[ ]:


# # Freeze the first n-2 layers
# for i in range(len(model.encoder.layers) - 2):
#     for param in model.encoder.layers[i].parameters():
#         param.requires_grad = False

# for i in range(len(model.decoder.layers) - 2):
#     for param in model.decoder.layers[i].parameters():
#         param.requires_grad = False


# In[ ]:


# Multi-GPU Batching
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.", flush=True)
    model = nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids=[0, 1])

model = model.to(device)
print(model)


# ### 4. Loss Function and Optimizer

# In[ ]:


from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_scheduler

num_epochs = 10 # [3, 10]
num_training_steps = num_epochs * len(train_dl)

learning_rate = 1e-3 # [5e-5, 5e-4]
lr_scheduler_type = "linear" 
# lr_scheduler_type = "reduce_lr_on_plateau"

warmup_steps = int(0.1 *  num_training_steps)

## The loss function
loss_fct =  nn.CrossEntropyLoss(ignore_index=-100)

## The optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

lr_scheduler = get_scheduler (
    lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    # num_training_steps=num_training_steps
    num_training_steps=num_training_steps - warmup_steps
)


# In[ ]:





# ### 6. Training Loop

# In[ ]:


wandb.init(
    project="ANLP-Project",
    name="t5-small",
    config={
        "architecture": "T5",
        "dataset": "Wiki-Auto",
        "batch_size": batch_size,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        # # "gamma": GAMMA,
        # # "step_size": STEP_SIZE
        # "factor": FACTOR,
        # "patience": PATIENCE,
        # "log_step": LOG_STEP
    }
)


# In[ ]:


## Reminder: 
## This process can (and should be!) be done by 
## calling the model(**batch) to get the lm_head_output directly

curr_steps = 0

best_valid_loss = float('inf')
best_epoch = 0

for epoch in tqdm(range(num_epochs)):

    training_loss = 0.0
    validation_loss = 0.0
    
    model.train()
    
    for batch in train_dl:
        curr_steps += 1
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        
        ## Compute the loss
        # loss = loss_fct(outputs.view(-1, model.config.vocab_size),
        #               batch['labels'].view(-1))
        loss = torch.mean(outputs.loss)
        training_loss += loss.item()
        
        wandb.log({
          'steps/step': curr_steps,
          'steps/epoch': epoch,
          'steps/loss': loss.item(),
          'steps/lr': float(lr_scheduler.get_last_lr()[0]),
        })
        
        loss.backward() # Update the weights
        optimizer.step() # Notify optimizer that a batch is done.
        optimizer.zero_grad() # Reset the optimer

        # lr_scheduler.step(loss.item()) # Notify the scheduler that a ...
        lr_scheduler.step() # Notify the scheduler that a ...

    model.eval()
    for batch in val_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
            
        with torch.no_grad():
            outputs = model(**batch)
        
        loss = torch.mean(outputs.loss)
        
        validation_loss += loss.item()
    
    training_loss = training_loss / len(train_dl)
    validation_loss = validation_loss / len(val_dl)
    
    print("Epoch {}:\tTraining Loss {:.2f}\t/\tValidation Loss {:.2f}".format(epoch+1, training_loss, validation_loss))
    
    ## Saving Best model
    if best_valid_loss >= validation_loss:
        best_valid_loss = validation_loss
        best_epoch = epoch + 1
        
        if hasattr(model, 'module'):
            model.module.save_pretrained(f"{abs_root}/t5-small-best")
        else:
            model.save_pretrained(f"{abs_root}/t5-small-best")
    
    wandb.log({
        'epochs/epoch': epoch,
        'epochs/train_loss': training_loss,
        'epochs/val_loss': validation_loss,
    })


# In[ ]:





# ### 7. Save Model

# In[ ]:


if hasattr(model, 'module'):
    model.module.save_pretrained(f"{abs_root}/t5-small-final")
else:
    model.save_pretrained(f"{abs_root}/t5-small-final")


# In[ ]:




