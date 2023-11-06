#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq Text Summarization Model

# In[ ]:


import csv
import json
import yaml
import string

import numpy as np
import pandas as pd
from scipy import stats

import nltk

from tqdm import tqdm

import wandb
# wandb.login(key='913841cb22c908099db4951c258f4242c1d1b7aa')

import os
os.environ['WANDB_API_KEY'] = '913841cb22c908099db4951c258f4242c1d1b7aa'
os.environ['WANDB_SILENT'] = 'true'


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## To avoid Cuda out of Memory Error (if doesn't work, try reducing batch size)
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# fine tune mt5 on dataset
from datasets import load_dataset

from transformers import  AutoTokenizer, BertGenerationDecoder, BertGenerationEncoder,BertGenerationConfig,EncoderDecoderModel
from transformers import XLMTokenizer, XLMForSequenceClassification

from transformers import MT5ForConditionalGeneration, MT5Tokenizer, BertTokenizer, BertModel
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import pipeline

from simpletransformers.t5 import T5Model, T5Args

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

import klib


# ### 1. Import Dataset

# In[ ]:


# import datasets

# # Load 1% of the training/validation sets.
# train_data      = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[0:1%]")
# validation_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[0:1%]")


# In[ ]:


abs_root = '/ssd_scratch/cvit/adhiraj_deshmukh'
abs_code = f'{abs_root}/ANLP-Project'
abs_data = f'{abs_code}/data'


# In[ ]:


#load dataset
colnames = ['source', 'target']

# input_file = "train.tsv"
input_file = f"{abs_data}/train.tsv"
train = pd.read_csv(input_file, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)
val = pd.read_csv(f"{abs_data}/valid.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)


# In[ ]:


#data cleaning 

train=klib.data_cleaning(train)
val=klib.data_cleaning(val)


# In[ ]:


#split train, val, test
# convert df  so that it can be used by transformers

train, test = train_test_split(train, test_size=0.2, random_state=42)
#train, val = train_test_split(train, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)


# In[ ]:


#print lens
print(len(train))
print(len(val))
print(len(test))


# In[ ]:


#save train, val, test
train.to_csv(f'{abs_data}/train.csv', index=False)
val.to_csv(f'{abs_data}/val.csv', index=False)
test.to_csv(f'{abs_data}/test.csv', index=False)


# In[ ]:


# %%
train.columns


# In[ ]:





# ### 2. Tokenize and Load Data

# In[ ]:


from transformers import BartTokenizer
from torch.utils.data import DataLoader

prefix = "Summarize: "

max_input_length = 512
max_target_length = 64
batch_size = 16 # [4, 16]


# In[ ]:


## Load the BART's pre-trained Tokenizer

# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir=f'{abs_root}/bart_base')


# In[ ]:


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
    inputs = [prefix + f"\"clean_text(text)\"" for text in batch["source"]]
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


train = load_dataset('csv', data_files=f'{abs_data}/train.csv',cache_dir=f'{abs_root}/t5_data')
val = load_dataset('csv', data_files=f'{abs_data}/val.csv',cache_dir=f'{abs_root}/t5_data')
test = load_dataset('csv', data_files=f'{abs_data}/test.csv',cache_dir=f'{abs_root}/t5_data')


# In[ ]:


train["validation"] = val["train"]
train["test"] = test["train"]


# In[ ]:


train["train"] = train["train"].shuffle().select(range(100000))
train["validation"] = train["validation"].shuffle().select(range(1000))
train["test"] = train["test"].shuffle().select(range(1000))


# In[ ]:


# Map the function to both train/validation sets.
train = train.map(
    process_data_to_model_inputs, 
    batched=True,
    # remove_columns=["source", "target"], batch_size = batch_size
    remove_columns=["source", "target"], batch_size = 256
)


# In[ ]:


# Convert the Dataset to PyTorch tensor with the expected columns
train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                           "decoder_attention_mask", "labels"],
)


# In[ ]:


# Make the iterative object that does batching using the DataLoader
train_dl = DataLoader(train["train"], batch_size=batch_size, shuffle=True)
val_dl = DataLoader(train["validation"], batch_size=batch_size, shuffle=True)


# In[ ]:





# ### 3. Load Pre-trained Model

# In[ ]:


from transformers import BartForConditionalGeneration
import torch

# Load the model
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", cache_dir=f'{abs_root}/bart_base')
model = model.to(device)

# Split model's components
the_encoder = model.get_encoder()
the_decoder = model.get_decoder()

last_linear_layer = model.lm_head


# In[ ]:


# Not sure why this is done ?

model.resize_token_embeddings(len(tokenizer))


# ### 4. Loss Function and Optimizer

# In[ ]:


from torch.nn import CrossEntropyLoss
from transformers import AdamW
from transformers import get_scheduler

num_epochs = 3 # [3, 10]
num_training_steps = num_epochs * len(train_dl)

learning_rate = 5e-5 # [5e-5, 5e-4]
lr_scheduler_type = "linear"

warmup_steps = 0
# optim = "adafactor"
# weight_decay = 0.01

## The loss function
loss_fct =  nn.CrossEntropyLoss(ignore_index=-100)

## The optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
# optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

lr_scheduler = get_scheduler (
    lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)
# lr_scheduler = AdafactorSchedule(optimizer)


# ### 5. Freeze Layer for Finetuning

# In[ ]:


# Freeze the first n-2 layers
for i in range(len(model.model.encoder.layers) - 2):
    for param in model.model.encoder.layers[i].parameters():
        param.requires_grad = False

for i in range(len(model.model.decoder.layers) - 2):
    for param in model.model.decoder.layers[i].parameters():
        param.requires_grad = False


# In[ ]:





# ### 6. Training Loop

# In[ ]:


wandb.init(
    project="ANLP-Project",
    # name="BART Finetune on Wiki-Auto",
    config={
        "architecture": "BART-Checkpoint",
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

for epoch in tqdm(range(num_epochs)):

    training_loss = 0
    validation_loss = 0
    
    model.train()
    
    for batch in train_dl:
        curr_steps += 1
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Get the "input's representation"
        encoder_output = the_encoder(input_ids = batch['input_ids'],
                                   attention_mask = batch['attention_mask'])
        
        # Pass the representation + the target summary to the decoder
        decoder_output = the_decoder(input_ids=batch['decoder_input_ids'],
                                   attention_mask=batch['decoder_attention_mask'],
                                   encoder_hidden_states=encoder_output[0],
                                   encoder_attention_mask=batch['attention_mask'])
        
        # Use the last linear layer to predict the next token
        decoder_output = decoder_output.last_hidden_state
        lm_head_output = last_linear_layer(decoder_output)
        
        # Compute the loss
        loss = loss_fct(lm_head_output.view(-1, model.config.vocab_size),
                      batch['labels'].view(-1))
        
        training_loss += loss.item()
        
        wandb.log({
          'steps/step': curr_steps,
          'steps/epoch': epoch,
          'steps/loss': loss.item(),
        })
        
        loss.backward() # Update the weights
        optimizer.step() # Notify optimizer that a batch is done.
        lr_scheduler.step() # Notify the scheduler that a ...
        optimizer.zero_grad() # Reset the optimer


    model.eval()
    for batch in val_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
            
        with torch.no_grad():
            outputs = model(**batch)
        
        loss = outputs.loss
        
        validation_loss += loss.item()
    
    training_loss = training_loss / len(train["train"] )
    validation_loss = validation_loss / len(val["train"])
    
    print("Epoch {}:\tTraining Loss {:.2f}\t/\tValidation Loss {:.2f}".format(epoch+1, training_loss, validation_loss))
    
    wandb.log({
        'epochs/epoch': epoch,
        'epochs/train_loss': training_loss,
        'epochs/val_loss': validation_loss,
    })


# In[ ]:





# ### 7. Save Model

# In[ ]:


model.save_pretrained(f"{abs_root}/new_mbart_final")


# In[ ]:




