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




def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.01, 0.01)

class Encoder_GRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, dropout = dropout, \
                           bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        ##compute embedding
        
        embedded = self.dropout(self.embedding(src))
        
        #rnn layer
        
        outputs, hidden = self.rnn(embedded)
        #if bidirection is True, concat the outputs and pass through another layer
        
        if self.bidirectional:
            hidden_concat  = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden = torch.tanh(self.fc(hidden_concat))
        
        
        return outputs, hidden
    
class Decoder_GRU(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        #embedding
        self.embedding = nn.Embedding(output_dim, emb_dim)
        #rnn 
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, dropout = dropout, bidirectional=self.bidirectional)
        #hidden linear layer
        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        #output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        #dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        
        input = input.unsqueeze(0)
        
        #embedding
        
        embedded = self.dropout(self.embedding(input))
        
        #rnn
                
        output, hidden = self.rnn(embedded, (hidden, cell))
        
        if self.bidirectional:
            hidden_concat  = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden = torch.tanh(self.fc_hidden(hidden_concat))
            
        
        prediction = self.fc_out(output.squeeze(0))

        
        return prediction, hidden
    
class Seq2Seq_GRU(nn.Module):
    def __init__(self, encoder: Encoder_GRU, decoder: Decoder_GRU, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        max_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder's output
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # encoder outputs
        hidden, cell = self.encoder(src)

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        ##first input to the decoder <SOS>
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            outputs[t] = output
            
            #randomly decide to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest prediction
            top1 = output.argmax(1) 
            
            #if teacher forcing, use next token as next input else use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs

        
def create_model_GRU(inp_dim,out_dim, enc_embed_dim,dec_embed_dim, hidden_dim, num_layers, dropout_prob,\
                bidirectional, source, target, lr, optm ):
  enc = Encoder_GRU(inp_dim, enc_embed_dim, hidden_dim, num_layers, dropout_prob, bidirectional)
  dec = Decoder_GRU(out_dim, dec_embed_dim, hidden_dim, num_layers, dropout_prob, bidirectional)

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



    
        
