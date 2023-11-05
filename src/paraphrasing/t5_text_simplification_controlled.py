# %%
import numpy
# fine tune mt5 on dataset
from transformers  import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
import nltk
nltk.download('punkt')
import string
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

#import train split
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import torch
import torch.nn as nn
import klib
import os
import csv

torch.cuda.is_available()



train = pd.read_csv("../../data/10/train_with_parameters.csv")
val = pd.read_csv("../../data/10/val_with_parameters.csv")
# %%

train=klib.data_cleaning(train)
val=klib.data_cleaning(val)


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


train.columns

#tokenize
tokenizer =  T5Tokenizer.from_pretrained("t5-small",cahe_dir='/ssd_scratch/cvit/aparna/t5_small')

# Load pre-trained XLM model
#tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<pad>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['</s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<unk>']})

for i in range(0,11):
  tokenizer.add_special_tokens({'additional_special_tokens': ['<copy_{}>'.format(i*0.1)]})
  tokenizer.add_special_tokens({'additional_special_tokens': ['<levsim_{}>'.format(i*0.1)]})
  tokenizer.add_special_tokens({'additional_special_tokens': ['<cratio_{}>'.format(i*0.1)]})





# %%
train['source']

# %%
# Print the original source.
print(' Original: ', train['source'][0])

# Print the tweet split into tokens.
print('Tokenized: ', tokenizer.tokenize(train['source'][0]))

# Print the tweet mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train['source'][0])))


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
  texts_with_params = []
  for text, ls,cp,comp in zip(texts_cleaned, examples["lavenstein_similarity"],examples["copy_ratio"],examples["compression_ratio"]):
    texts_with_params.append("<copy_{}> <levsim_{}> <cratio_{}> ".format(cp,ls,comp) + text)
  
     
  inputs = [prefix + text for text in texts_with_params]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["target"], max_length=max_target_length, 
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

train = load_dataset('csv', data_files='/ssd_scratch/cvit/aparna/train.csv',cache_dir='/ssd_scratch/cvit/aparna/t5_data_controlled')
val = load_dataset('csv', data_files='/ssd_scratch/cvit/aparna/val.csv',cache_dir='/ssd_scratch/cvit/aparna/t5_data_controlled')
test = load_dataset('csv', data_files='/ssd_scratch/cvit/aparna/test.csv',cache_dir='/ssd_scratch/cvit/aparna/t5_data_controlled')

train["validation"] = val["train"]
train["test"] = test["train"]

# medium_datasets["train"] = medium_datasets["train"].shuffle().select(range(100000))
# medium_datasets["validation"] = medium_datasets["validation"].shuffle().select(range(1000))
# medium_datasets["test"] = medium_datasets["test"].shuffle().select(range(1000))

train["train"] = train["train"].shuffle().select(range(300000))
train["validation"] = train["validation"].shuffle().select(range(10000))
train["test"] = train["test"].shuffle().select(range(10000))
print(train)
train= train.map(
    preprocess_data, 
    batched=True,
    remove_columns=["source", "target"], batch_size=128
)


print(train)
#get sample 
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

model =  T5ForConditionalGeneration.from_pretrained("t5-small",cache_dir='/ssd_scratch/cvit/aparna/t5_small').to(device)

# model.encoder.resize_token_embeddings(len(tokenizer))
# model.decoder.resize_token_embeddings(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))



for param in model.parameters():
     param.requires_grad = False

num_encoder_layers = len(model.encoder.block)
# print(num_encoder_layers)
# and Un-Freeze lower 4 layers of encoder 

for param in model.encoder.block[5].parameters():
    param.requires_grad = True

for param in model.decoder.block[5].parameters():
    param.requires_grad = True
for name, param in model.named_parameters():
    print(name,param.requires_grad)


#training args
#data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
  output_dir = "/ssd_scratch/cvit/aparna/t5_simplification_controlled",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  #warmup_steps = 90,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size =46,
  per_device_eval_batch_size =4,
  gradient_accumulation_steps = 16,
  evaluation_strategy = "steps",
  eval_steps = 100,
  #predict_with_generate=True,
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
    eval_dataset=train["validation"],             # evaluation dataset
    tokenizer=tokenizer,               # tokenizer
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), # data collator
    
)

trainer.train()


#save model
trainer.save_model("/ssd_scratch/cvit/aparna/t5_simplification_controlled_final")

#save tokenizer
tokenizer.save_pretrained("/ssd_scratch/cvit/aparna/t5_simplification_controlled_final_tokenizer")

# %%
