# %%
import numpy
# fine tune mt5 on dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, BartTokenizer, BertModel
from datasets import load_dataset
from transformers import  BartForConditionalGeneration
from transformers import XLMTokenizer, XLMForSequenceClassification

from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from simpletransformers.t5 import T5Model, T5Args
from transformers import pipeline
#import train split
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import torch
import torch.nn as nn
from google.transliteration import transliterate_word
import klib
import os
import csv

# %%
torch.cuda.is_available()

# %%


# %%
#load dataset

colnames = ['source', 'target']
input_file = "train.tsv"
train = pd.read_csv(input_file, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)
val = pd.read_csv("valid.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)

# %%
#data cleaning 

train=klib.data_cleaning(train)
val=klib.data_cleaning(val)

# %%
#split train, val, test
# convert df  so that it can be used by transformers


train, test = train_test_split(train, test_size=0.2, random_state=42)
#train, val = train_test_split(train, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

#print lens
print(len(train))
print(len(val))
print(len(test))

#save train, val, test
train.to_csv('/ssd_scratch/cvit/aparna/train.csv', index=False)
val.to_csv('/ssd_scratch/cvit/aparna/val.csv', index=False)
test.to_csv('/ssd_scratch/cvit/aparna/test.csv', index=False)


# %%
train.columns

# %%
#tokenize
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',cahe_dir='/ssd_scratch/cvit/aparna/bart_base')

# Load pre-trained XLM model
#tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
# tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<pad>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['</s>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<unk>']})



# %%
train['source']

# %%
# Print the original source.
print(' Original: ', train['source'][0])

# Print the tweet split into tokens.
print('Tokenized: ', tokenizer.tokenize(train['source'][0]))

# Print the tweet mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train['source'][0])))

# %%
maxlen = 512
def tokenize_df(df):
    input = tokenizer(df['source'], padding='max_length', truncation=True, return_tensors="pt", max_length=maxlen)
    target= tokenizer(df['target'], padding='max_length', truncation=True, return_tensors="pt", max_length=maxlen)
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    target_ids = target['input_ids']
    target_attention_mask = target['attention_mask']
    decoder_input_ids = target_ids.clone()
    #convert to tensors
    input_ids = torch.tensor(input_ids).squeeze()
    attention_mask = torch.tensor(attention_mask).squeeze()
    target_ids = torch.tensor(target_ids).squeeze()
    target_attention_mask = torch.tensor(target_attention_mask).squeeze()
   # decoder_input_ids = torch.tensor(decoder_input_ids)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': target_ids,
        #'decoder_input_ids': decoder_input_ids,
        #'decoder_attention_mask': target_attention_mask
    }

train = load_dataset('csv', data_files='/ssd_scratch/cvit/aparna/train.csv',cache_dir='/ssd_scratch/cvit/aparna/bart_data')
val = load_dataset('csv', data_files='/ssd_scratch/cvit/aparna/val.csv',cache_dir='/ssd_scratch/cvit/aparna/bart_data')
test = load_dataset('csv', data_files='/ssd_scratch/cvit/aparna/test.csv',cache_dir='/ssd_scratch/cvit/aparna/bart_data')
train = train.map(tokenize_df, batched=True, batch_size=128,remove_columns=['source','target'])
val = val.map(tokenize_df, batched=True, batch_size=128,remove_columns=['source','target'])
test = test.map(tokenize_df, batched=True, batch_size=128,remove_columns=['source','target'])


# %%
train
#get sample 
sample = train['train'][0]
sample
#print shapes
print(len(sample['input_ids']))
print(len(sample['attention_mask']))

os.environ["WANDB_DISABLED"] = "false"

#model = BertGenerationEncoder.from_pretrained("bert-base-multilingual-cased")


# encoder = BertGenerationEncoder.from_pretrained("bert-base-multilingual-cased", bos_token_id=101, eos_token_id=102)
# # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
# decoder = BertGenerationDecoder.from_pretrained(
#     "bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
# )
# model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model =  BartForConditionalGeneration.from_pretrained('bert-base-multilingual-cased')

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base",cache_dir='/ssd_scratch/cvit/aparna/bart_base').to(device)

# model.encoder.resize_token_embeddings(len(tokenizer))
# model.decoder.resize_token_embeddings(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))


# %%


#training args


training_args = Seq2SeqTrainingArguments(
  output_dir = "/ssd_scratch/cvit/aparna/mbart_simplification",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  warmup_steps = 90,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size =2,
  per_device_eval_batch_size = 1,
  gradient_accumulation_steps = 16,
  evaluation_strategy = "steps",
  eval_steps = 100,
  predict_with_generate=True,
  generation_max_length = 128,
  save_steps = 500,
  logging_steps = 10,
  push_to_hub = False
)


#trainer
trainer = Seq2SeqTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train["train"],         # training dataset
    eval_dataset=val["train"],             # evaluation dataset
    tokenizer=tokenizer,               # tokenizer
    #data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), # data collator
    
)

#train
trainer.train()

#save model
trainer.save_model("/ssd_scratch/cvit/aparna/mbart_simplification_final")

# %%
