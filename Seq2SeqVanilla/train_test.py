


import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from torchtext.data import TabularDataset
from torchtext.data import Field, BucketIterator
import pandas as pd
import argparse

from data import *
from train_test import *
from lstm_model import *
from rnn_model import *

from gru_model import *
def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg, teacher_forcing_ratio)
        
        output_dim = output.shape[-1]
        
        #flatten output
        output = output[1:].view(-1, output_dim)
        #flatten target
        trg = trg[1:].view(-1)
        
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def predict_word(device, model, source, target, src, trg):
    #print(src, trg)
    src_tensor = source.process([src]).to(device)
    trg_tensor = target.process([trg]).to(device)
    #print(src_tensor)
  
    model.eval()
    with torch.no_grad():
      outputs = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)
    output_ids = outputs[1:].squeeze(1).argmax(1)
    output_tokens = [target.vocab.itos[idx] for idx in output_ids]
    return output_tokens
    

def predict(device, model, iterator, source, target, testfile, correct_path, incorrect_path, save_pred):
    
    src_words_correct = []
    trg_words_correct = []
    pred_words_correct = []
    src_words_incorrect = []
    trg_words_incorrect = []
    pred_words_incorrect = []

    src_vocab = source.vocab
    trg_vocab = target.vocab

    test_data = pd.read_csv(testfile,sep=",")

    src = test_data.iloc[:,0]
    trg = test_data.iloc[:,1]

    num_predictions = len(src)

    correct_predictions = 0

    for s, t  in zip(src, trg):
        pred_word = predict_word(device, model,source, target, s, t)
        #print("pred_",pred_word)

        src_list = tokenize(s)
        trg_list = tokenize(t)

        pred_list = [item for item in pred_word if '<eos>' not in item and '<sos>' not in item]
        
        src_str = "".join(src_list)
        trg_str = "".join(trg_list)
        pred_str = "".join(pred_list)

        #compute accuracy
        if trg_list == pred_list:
            src_words_correct.append(s)
            trg_words_correct.append(t)
            pred_words_correct.append(pred_str)
            correct_predictions+=1
        else:
            src_words_incorrect.append(s)
            trg_words_incorrect.append(t)
            pred_words_incorrect.append(pred_str)
    accuracy = correct_predictions*100.0/num_predictions

    #save predictions
    if save_pred:
      df_correct = pd.DataFrame({"source":src_words_correct, "target":trg_words_correct,\
                                    "prediction": pred_words_correct})
      df_incorrect = pd.DataFrame({"source":src_words_incorrect, "target":trg_words_incorrect,\
                                    "prediction": pred_words_incorrect})
      df_correct.to_csv(correct_path, index=False)
      df_incorrect.to_csv(incorrect_path, index=False)
      
    return accuracy