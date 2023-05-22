


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

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm


from data import *
from gru_model import *

def train(model, iterator, optimizer, criterion, clip,teacher_forcing_ratio):
    """ Train seq2seq model"""
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, att_list = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)



def evaluate(model, iterator, criterion):
    """Evaluate seq2seq model"""
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, att_list = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            
            #output = [(trg len - 1) * batch size, output dim]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)



def showAttention(input_word, output_word, attentions):
    """show attention heatmap"""
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    prop = fm.FontProperties(fname='AnjaliOldLipi-Regular.ttf')
    ax.set_xticklabels(['']+input_word, rotation=90)
    ax.set_yticklabels([''] + output_word,fontproperties=prop)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()



def predict_word(device, model, source, target, src, trg):
    """Predict a word given source word"""
    #print(src, trg)
    src_tensor = source.process([src]).to(device)
    trg_tensor = target.process([trg]).to(device)
    #print(src_tensor)
  
    model.eval()
    with torch.no_grad():
      outputs, att_list = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)
    output_ids = outputs[1:].squeeze(1).argmax(1)
    output_tokens = [target.vocab.itos[idx] for idx in output_ids]
    return output_tokens, att_list
    

def predict(device, model, iterator, source, target, testfile, correct_path, incorrect_path, save_pred, show_att):
    #list to store correctly predicted words, source and target
    src_words_correct = []
    trg_words_correct = []
    pred_words_correct = []
    #lists to store incorrectly predicted words, source, target
    src_words_incorrect = []
    trg_words_incorrect = []
    pred_words_incorrect = []

    src_vocab = source.vocab
    trg_vocab = target.vocab

    #read test data
    test_data = pd.read_csv(testfile,sep=",")
    src = test_data.iloc[:,0]
    trg = test_data.iloc[:,1]
    num_predictions = len(src)
    correct_predictions = 0
    # for each word in test data predict
    for s, t  in zip(src, trg):
        pred_word, att_list = predict_word(device, model,source, target, s, t)
        #print("pred_",pred_word)
        src_list = tokenize(s)
        trg_list = tokenize(t)
        pred_list = [item for item in pred_word if '<eos>' not in item and '<sos>' not in item]
        
        src_str = "".join(src_list)
        trg_str = "".join(trg_list)
        pred_str = "".join(pred_list)

        #show attention matrix
        if show_att:
           att_tensor = torch.stack(att_list)
           att_matrix = att_tensor.numpy()
           att_matrix = np.squeeze(att_matrix)
           #print(src_list, trg_list, att_matrix.shape)
           showAttention(src_list, trg_list, att_matrix)
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
    
    #save the predictions in csv files
    if save_pred:
      df_correct = pd.DataFrame({"source":src_words_correct, "target":trg_words_correct,\
                                    "prediction": pred_words_correct})
      df_incorrect = pd.DataFrame({"source":src_words_incorrect, "target":trg_words_incorrect,\
                                    "prediction": pred_words_incorrect})
      df_correct.to_csv(correct_path, index=False)
      df_incorrect.to_csv(incorrect_path, index=False)
      
    return accuracy