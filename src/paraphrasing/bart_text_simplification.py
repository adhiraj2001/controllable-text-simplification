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
# from google.transliteration import transliterate_word
import klib
import os
import csv

from tqdm import tqdm
import nltk
nltk.download('punkt')

import string

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
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir=f'{abs_root}/bart_base')

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

prefix = "summarize: "
max_input_length = 512
max_target_length = 64

def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_titles)
  return text_cleaned

def preprocess_data(examples):
  texts_cleaned = [clean_text(text) for text in examples["source"]]
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, padding="max_length", max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["target"], padding="max_length", max_length=max_target_length, 
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  model_inputs["decoder_input_ids"] = labels["input_ids"] 
  model_inputs["decoder_attention_mask"] = labels["attention_mask"]
  return model_inputs

train = load_dataset('csv', data_files=f'{abs_data}/train.csv',cache_dir=f'{abs_root}/t5_data')
val = load_dataset('csv', data_files=f'{abs_data}/val.csv',cache_dir=f'{abs_root}/t5_data')
test = load_dataset('csv', data_files=f'{abs_data}/test.csv',cache_dir=f'{abs_root}/t5_data')

train["validation"] = val["train"]
train["test"] = test["train"]

# medium_datasets["train"] = medium_datasets["train"].shuffle().select(range(100000))
# medium_datasets["validation"] = medium_datasets["validation"].shuffle().select(range(1000))
# medium_datasets["test"] = medium_datasets["test"].shuffle().select(range(1000))

train["train"] = train["train"].shuffle().select(range(100000))
train["validation"] = train["validation"].shuffle().select(range(1000))
train["test"] = train["test"].shuffle().select(range(1000))
print(train)
train= train.map(
    preprocess_data, 
    batched=True,
    remove_columns=["source", "target"], batch_size=128
)


print(train)

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

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", cache_dir=f'{abs_root}/bart_base').to(device)

# model.encoder.resize_token_embeddings(len(tokenizer))
# model.decoder.resize_token_embeddings(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))


# %%


#training args

training_args = Seq2SeqTrainingArguments(
  output_dir = f"{abs_root}/mbart_simplification",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  # warmup_steps = 90,
  warmup_steps = 0,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size = 2,
  per_device_eval_batch_size = 1,
  gradient_accumulation_steps = 16,
  evaluation_strategy = "steps",
  # eval_steps = 100,
  eval_steps = 500,
  # predict_with_generate=True,
  predict_with_generate=False,
  generation_max_length = 128,
  save_steps = 500,
  logging_steps = 10,
  push_to_hub = False
)

print(f'encoder: {model.model.encoder}')
print()

print(f'decoder: {model.model.decoder}')
print()

# for i, param in enumerate(model.model.encoder.parameters()):
#     if i < (len(model.model.encoder.layer) - 2):
#         param.requires_grad = False
#
# for i, param in enumerate(model.model.decoder.parameters()):
#     if i < (len(model.model.decoder.layer) - 2):
#         param.requires_grad = False


# Freeze the first n-2 layers
for i in range(len(model.model.encoder.layers) - 2):
    for param in model.model.encoder.layers[i].parameters():
        param.requires_grad = False

for i in range(len(model.model.decoder.layers) - 2):
    for param in model.model.decoder.layers[i].parameters():
        param.requires_grad = False

#trainer
trainer = Seq2SeqTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train["train"],         # training dataset
    eval_dataset=train["validation"],             # evaluation dataset
    tokenizer=tokenizer,               # tokenizer
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), # data collator
)

#train
trainer.train()

#save model
trainer.save_model(f"{abs_root}/mbart_simplification_final")

# %%
