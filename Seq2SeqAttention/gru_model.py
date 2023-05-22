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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from data import *
from train_test import *


def init_weights(m):
    """ Initialize paramaeters"""
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.01, 0.01)

class Encoder_GRU(nn.Module):
    """Encoder for bidirectional GRU"""
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, hidden_dim, bidirectional = True)
        
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        
        #compute embedding
        embedded = self.dropout(self.embedding(src))
        
        # bidirectional rnn layer
        
        outputs, hidden = self.rnn(embedded)

        #concat the outputs from two dimension and then do fc and tanh                
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
                
        return outputs, hidden
    

class Attention(nn.Module):
    """Compute attention weights"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.U = nn.Linear((hidden_dim * 2) + hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outs):
        
        
        batch_size = encoder_outs.shape[1]
        src_len = encoder_outs.shape[0]
        
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outs = encoder_outs.permute(1, 0, 2)    
        
        
        energy = torch.tanh(self.U(torch.cat((hidden, encoder_outs), dim = 2))) 
        
        att_score = self.V(energy).squeeze(2)
        att_weights = torch.softmax(att_score, dim=1)
        
    
        
        return att_weights

class Decoder_GRU(nn.Module):
    """Decoder """
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((hidden_dim * 2) + emb_dim, hidden_dim)
        
        self.fc_out = nn.Linear((hidden_dim * 2) + hidden_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outs):
             
        
        #reshape input
        input = input.unsqueeze(0)
        
        #get embedding
        
        embedded = self.dropout(self.embedding(input))
        
        #find attention weights
        
        att_wts = self.attention(hidden, encoder_outs)
                
        #reshape attention weights
        
        att_wts = att_wts.unsqueeze(1)
        
        #permute output from encoder
        
        encoder_outs = encoder_outs.permute(1, 0, 2)
        
        #find weighted embedding
        
        weighted_embed = torch.bmm(att_wts, encoder_outs)
        
        #permute weighted embeddding
        
        weighted_embed = weighted_embed.permute(1, 0, 2)
        
        #concat embedding and weighted embedding
        
        rnn_input = torch.cat((embedded, weighted_embed), dim = 2)
        
        #pass it to rnn
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_embed = weighted_embed.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted_embed, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), att_wts
    


class Seq2Seq_GRU(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        
        #get encoder output
        encoder_outs, hidden = self.encoder(src)
                
        #first input to the decoder <SOS>
        input = trg[0,:]
        att_list = [] # to store the att_wts of all tokens
        for t in range(1, trg_len):
            
            #collect the att_wts of a the input token
            output, hidden, att_wts = self.decoder(input, hidden, encoder_outs)

            att_list.append(att_wts)
            outputs[t] = output
            
            #randomly decide to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest prediction
            top1 = output.argmax(1) 
            
            #if teacher forcing, use next token as next input else use predicted token
        
            input = trg[t] if teacher_force else top1
        #print(type(att_wts), att_wts.shape)

        return outputs, att_list
        
def create_model_GRU(inp_dim,out_dim, enc_embed_dim,dec_embed_dim, hidden_dim, dropout_prob,\
                 source, target, lr , optm):
  attn = Attention(hidden_dim)            
  enc = Encoder_GRU(inp_dim, enc_embed_dim, hidden_dim,dropout_prob,)
  dec = Decoder_GRU(out_dim, dec_embed_dim, hidden_dim,  dropout_prob, attn)


  model = Seq2Seq_GRU(enc, dec, device).to(device)
  model.apply(init_weights)
  # Define the optimizer
  if optm == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)
  if optm == "nadam":
    optimizer = optim.NAdam(model.parameters(), lr=lr)
  if optm == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
  # CrossEntropyLoss: ignores the padding tokens.
  TARGET_PAD_IDX = target.vocab.stoi[target.pad_token]
  criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

  return model, optimizer, criterion
