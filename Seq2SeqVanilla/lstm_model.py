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
class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout = dropout, \
                           bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        embedded = self.dropout(self.embedding(src))     
        outputs, (hidden, cell) = self.rnn(embedded)
        
        if self.bidirectional:
            hidden_concat  = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden = torch.tanh(self.fc(hidden_concat))        
        
        return hidden, cell
    
class Decoder_LSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout = dropout, bidirectional=self.bidirectional)

        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        if self.bidirectional:
            hidden_concat  = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden = torch.tanh(self.fc_hidden(hidden_concat))
            
        
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell
    

    class Seq2Seq_LSTM(nn.Module):
        def __init__(self, encoder: Encoder_LSTM, decoder: Decoder_LSTM, device: torch.device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device

        def forward(self, src, trg, teacher_forcing_ratio=0.5):

            max_len, batch_size = trg.shape
            trg_vocab_size = self.decoder.output_dim

            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
            
           
            hidden, cell = self.encoder(src)

            batch_size = trg.shape[1]
            trg_len = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
            
            
            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
            
           
            hidden, cell = self.encoder(src)
            
            #first input to the decoder is the <sos> tokens
            input = trg[0,:]
            
            for t in range(1, trg_len):
                
                output, hidden, cell = self.decoder(input, hidden, cell)
                
                
                outputs[t] = output
                
                #decide if we are going to use teacher forcing or not
                teacher_force = random.random() < teacher_forcing_ratio
                
                top1 = output.argmax(1) 
                
                input = trg[t] if teacher_force else top1
            
            return outputs

            


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.01, 0.01)

def create_model_LSTM(inp_dim,out_dim, enc_embed_dim,dec_embed_dim, hidden_dim, num_layers, dropout_prob,\
                bidirectional, source, target, lr , optm):
  enc = Encoder_LSTM(inp_dim, enc_embed_dim, hidden_dim, num_layers, dropout_prob, bidirectional)
  dec = Decoder_LSTM(out_dim, dec_embed_dim, hidden_dim, num_layers, dropout_prob, bidirectional)

  model = Seq2Seq_LSTM(enc, dec, device).to(device)
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


    
        
