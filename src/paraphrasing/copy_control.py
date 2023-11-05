# %%
import numpy
# fine tune mt5 on dataset
from transformers  import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, AdafactorSchedule
#import train split
import pandas as pd
import nltk
# nltk.download('punkt')
import string
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import torch
import torch.nn as nn
import klib
import os
import csv
import  wandb
from binner import GaussianBinner1D

wandb.init(project="t5_simplification")
# %%
torch.cuda.is_available()

# %%


# %%
#load dataset

colnames = ['source', 'target']
input_file = "../../data/train.tsv"
train = pd.read_csv(input_file, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)
val = pd.read_csv("../../data/valid.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)

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
train.to_csv('/ssd_scratch/cvit/shreya/train.csv', index=False)
val.to_csv('/ssd_scratch/cvit/shreya/val.csv', index=False)
test.to_csv('/ssd_scratch/cvit/shreya/test.csv', index=False)


# %%
# train.columns

# %%
#tokenize
tokenizer =  T5Tokenizer.from_pretrained("t5-small",cahe_dir='/ssd_scratch/cvit/shreya/t5_small')

# Load pre-trained XLM model
#tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
# tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<pad>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['</s>']})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<unk>']})



# %%
# train['source']

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
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs,padding="max_length", max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["target"], padding="max_length",max_length=max_target_length, 
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  model_inputs["decoder_input_ids"] = labels["input_ids"]
  model_inputs["decoder_attention_mask"] = labels["attention_mask"]
  return model_inputs

train = load_dataset('csv', data_files='/ssd_scratch/cvit/shreya/train.csv',cache_dir='/ssd_scratch/cvit/shreya/t5_data')
val = load_dataset('csv', data_files='/ssd_scratch/cvit/shreya/val.csv',cache_dir='/ssd_scratch/cvit/shreya/t5_data')
test = load_dataset('csv', data_files='/ssd_scratch/cvit/shreya/test.csv',cache_dir='/ssd_scratch/cvit/shreya/t5_data')

train["validation"] = val["train"]
train["test"] = test["train"]

# medium_datasets["train"] = medium_datasets["train"].shuffle().select(range(100000))
# medium_datasets["validation"] = medium_datasets["validation"].shuffle().select(range(1000))
# medium_datasets["test"] = medium_datasets["test"].shuffle().select(range(1000))

train["train"] = train["train"].shuffle().select(range(380000))
train["validation"] = train["validation"].shuffle().select(range(10000))
train["test"] = train["test"].shuffle().select(range(10000))
print(train)
train= train.map(
    preprocess_data, 
    batched=True,
    remove_columns=["source", "target"], batch_size=128
)


print(train)
train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels","decoder_input_ids", "decoder_attention_mask"])
train_dataloader = DataLoader(train["train"], batch_size=4, shuffle=True)
val_dataloader = DataLoader(train["validation"], batch_size=4, shuffle=True)

os.environ["WANDB_DISABLED"] = "false"


bert_dim = 64
cp_value = 0.2
binner = GaussianBinner1D(bert_dim, 1, 0, 1)
copy_vector = binner.generate_vectors(cp_value)
   
#model = BertGenerationEncoder.from_pretrained("bert-base-multilingual-cased")


# encoder = BertGenerationEncoder.from_pretrained("bert-base-multilingual-cased", bos_token_id=101, eos_token_id=102)
# # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
# decoder = BertGenerationDecoder.from_pretrained(
#     "bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
# )
# model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model =  BartForConditionalGeneration.from_pretrained('bert-base-multilingual-cased')

model =  T5ForConditionalGeneration.from_pretrained("t5-small",cache_dir='/ssd_scratch/cvit/shreya/t5_small').to(device)

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

the_encoder = model.get_encoder()
the_decoder = model.get_decoder()
#training args



num_epochs = 10 
loss_fct =  nn.CrossEntropyLoss(ignore_index=-100)
optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

for epoch in range(num_epochs):

    model.train()
    for batch in train_dataloader:
      # print(batch)
      if torch.cuda.is_available():
        batch = {k:v.to('cuda') for k, v in batch.items()}
        #batch = {k: v.to('cuda') for k, v in batch.items()}

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
      last_linear_layer = model.lm_head
      lm_head_output = last_linear_layer(decoder_output)
      
      # Compute the loss
      loss = loss_fct(lm_head_output.view(-1, model.config.vocab_size),
                      batch['labels'].view(-1))
      wandb.log({"train loss":loss.item()})
      loss.backward() # Update the weights
      optimizer.step() # Notify optimizer that a batch is done.
      lr_scheduler.step() # Notify the scheduler that a ...
      optimizer.zero_grad() # Reset the optimer

    model.eval()
    for batch in val_dataloader:
        if torch.cuda.is_available():
            batch = {k: v.to('cuda') for k, v in batch.items()}
                
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        wandb.log({"val loss": loss.item()})
    
    training_loss = training_loss / len( train["train"] )
    validation_loss = validation_loss / len( val["train"])
    print("Epoch {}:\tTraining Loss {:.2f}\t/\tValidation Loss {:.2f}".format(epoch+1, training_loss, validation_loss))
    wandb.log({"Training Loss": training_loss, "Validation Loss": validation_loss})

#save model
model.save_pretrained('/ssd_scratch/cvit/shreya/t5_simplification_custom')
          

          