import argparse
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5Tokenizer, T5Model
import evaluate 

def create_sequence(dataset, cutoff_len=100):
    X = []

    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    for sen in dataset:
        X.append(tokenizer(sen.lower(), return_tensors="pt").input_ids[:cutoff_len])

    X = pad_sequence(X, batch_first=True, padding_value=tokenizer.pad_token_id)

    return X


def load_data(file_name, batch_size=None, num_workers=0):

    comp_txt = open('./dataset/{}_comp.txt'.format(file_name), "r", encoding="utf-8").read().splitlines()
    simp_txt = open('./dataset/{}_simp.txt'.format(file_name), "r", encoding="utf-8").read().splitlines()

    comp_tensors = create_sequence(comp_txt, cutoff_len=100)
    simp_tensors = create_sequence(simp_txt, cutoff_len=100)

    # comp_tensors , simp_tensors = zip(*[(i[0], i[1]) for i in zip(comp_tensors, simp_tensors) if i[0] != i[1]])
    
    data_tensor = TensorDataset(comp_tensors, simp_tensors) 
    data_loader = DataLoader(data_tensor, batch_size=batch_size, num_workers=num_workers)

    return data_loader


def validate(model, valid_dl):

    sari = evaluate.load("sari")
    bleu = evaluate.load("bleu")
    
    val_loss = 0
    val_sari = 0
    val_bleu = 0

    model.eval()

    step_size = 500

    # with torch.inference_mode(): # Had trouble with tensor clone (NaN outputs)
    with torch.no_grad():
        for idx, (X, y) in enumerate(valid_dl):
            
            X = X.to(device)
            y = y.to(device)

            outputs = model(X, y)
            
            print(outputs)

            assert False

            ## For the cross-entropy function to work we flatten them
            y = y.view(-1)
            outputs = outputs.view(-1, outputs.size(-1))

            curr_loss = loss_func(outputs, y).item()
            curr_sari = sari(sources=X, predictions=outputs, references=y)
            curr_bleu = bleu(sources=X, predictions=outputs, references=y)
            
            if idx % step_size == 0:
                print(f"\nstep: {idx:03d} | Loss: {curr_loss:.3f}, Sari: {curr_sari:3f}, Bleu: {curr_bleu:.3f}", flush=True)

            val_loss += loss
            val_sari += curr_sari
            val_bleu += curr_bleu

        val_loss /= len(valid_dl)
        val_sari /= len(valid_dl)
        val_bleu /= len(valid_dl)

    return val_loss, val_sari, val_bleu


def main(args):

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    if torch.cuda.device_count() > 0 and NUM_WORKERS == 0:
        NUM_WORKERS = int(torch.cuda.device_count()) * 4
    

    val_loader = load_data('valid', BATCH_SIZE, NUM_WORKERS)

    model = T5Model.from_pretrained('./weights/t5_model.bin')

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.", flush=True)
        model = nn.DataParallel(model)


    model = model.to(device)

    val_loss, val_sari, val_bleu = validate(model, val_loader)
    
    print()
    print(f"\nValidation | Loss: {val_loss:.3f}, Sari: {val_sari:3f}, Bleu: {val_bleu:.3f}", flush=True)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of Workers between which batch size is divided parallely ?')

    args = parser.parse_args()

    main(args)
