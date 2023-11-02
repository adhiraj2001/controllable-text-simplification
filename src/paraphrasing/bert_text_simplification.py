# %%
import numpy
# fine tune mt5 on dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, BertTokenizer, BertModel
from datasets import load_dataset
from transformers import  AutoTokenizer, BertGenerationDecoder, BertGenerationEncoder,BertGenerationConfig,EncoderDecoderModel
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
# from google.transliteration import transliterate_word
import klib
import os
import csv

# %%
torch.cuda.is_available()

# %%

abs_root = '/ssd_scratch/cvit/adhiraj_deshmukh'
abs_code = f'{abs_root}/ANLP-Project'
abs_data = f'{abs_code}/data'

# %%
#load dataset

colnames = ['source', 'target']
# input_file = "train.tsv"
input_file = f"{abs_data}/train.tsv"
train = pd.read_csv(input_file, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)
val = pd.read_csv(f"{abs_data}/valid.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)

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
train.to_csv(f'{abs_data}/train.csv', index=False)
val.to_csv(f'{abs_data}/val.csv', index=False)
test.to_csv(f'{abs_data}/test.csv', index=False)


# %%
train.columns

# %%
#tokenize
tokenizer = AutoTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',cache_dir='f{abs_root}/bert_google_encoder')

# Load pre-trained XLM model
#tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<pad>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['</s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<unk>']})

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

train = load_dataset('csv', data_files=f'{abs_data}/train.csv',cache_dir=f'{abs_root}/bert_data')
val = load_dataset('csv', data_files=f'{abs_data}/val.csv',cache_dir=f'{abs_root}/bert_data')
test = load_dataset('csv', data_files=f'{abs_data}/test.csv',cache_dir=f'{abs_root}/bert_data')
                                          
                                          
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

encoder = BertGenerationEncoder.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder",cache_dir=f'{abs_root}/bert_google_encoder')

config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
config.is_decoder = True
decoder = BertGenerationDecoder.from_pretrained(
    "google/bert_for_seq_generation_L-24_bbc_encoder", config=config,cache_dir=f'{abs_root}/bert_google_decoder'
)


model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# model.encoder.resize_token_embeddings(len(tokenizer))
# model.decoder.resize_token_embeddings(len(tokenizer))


# %%


#training args


training_args = Seq2SeqTrainingArguments(
  output_dir = f"{abs_root}/bert_simplification_google",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  # warmup_steps = 90,
  warmup_steps = 0,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size =2,
  per_device_eval_batch_size = 1,
  gradient_accumulation_steps = 16,
  evaluation_strategy = "steps",
  eval_steps = 1,
  # predict_with_generate=True,
  predict_with_generate=False,
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
   # data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), # data collator
    
)

#train
trainer.train()

#save model
trainer.save_model(f"{abs_root}/google_bert_simplification_final")

# %%

