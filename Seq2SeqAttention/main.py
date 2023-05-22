

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
from gru_model import *





parser = argparse.ArgumentParser("Please enter the hyper parameters:")  
# creating  variables using the add_argument method  
parser.add_argument('-wp', '--wandb_project',  help='wandb project name', default="Seq2Seq")
parser.add_argument('-we', '--wandb_entity', help="wandb entity name", default="deep-learning-assignment")
parser.add_argument("-e", "--epochs", help="Number of epochs to train neural network",  default="5", type=int)
parser.add_argument("-b", "--batch_size", help="batchsize", default="128", type=int)
parser.add_argument("-c", "--clip", help="gradient clipping",  default="0.5", type=float)
parser.add_argument("-o", "--optimizer", help="choices:rmsprop, adam, nadam", default="nadam")
parser.add_argument("-lr", "--learning_rate", help="Learning rate used to optimize model parameters", default="0.001", type=float)
parser.add_argument("-rnn", "--rnn_type", help="Type of RNN choices: GRU",  default="GRU")
parser.add_argument("-d", "--dropout", help="dropout probability", default="0.5",type=float)
parser.add_argument("-tf", "--teacher_forcing_ratio", help="teacher forcing ratio decoder", default="0.5",type=float)
parser.add_argument("-eed", "--enc_emb_dim", help="Embedding dimension of encoder", default="256",type=int)
parser.add_argument("-ded", "--dec_emb_dim", help="Embedding dimension of decoder", default="256",type=int)
parser.add_argument("-hd", "--hidden_dim", help="Hidden dimension of encoder and decoder", default="512",type=int)
parser.add_argument("-bi", "--bidirectional", help="True",  default="True")
parser.add_argument("-nl", "--num_layers", help="Number of layers in RNN",  default="1",type=int)
parser.add_argument("-sa", "--show_attn", help="Show attention maps",  default=False)
parser.add_argument("-sp", "--save_pred", help="Save the predictions",  default=False)
parser.add_argument("-test","--test", help = "choices: True if perform test, else False", default=True)
parser.add_argument("-sweep","--sweep", help = "choices: True if perform wandb sweep, else False", default=True)  


#recieve command line arguments
args = parser.parse_args() 
#define defaults configuration of hyper parameters
config_defaults = { "lr": args.learning_rate,                
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "clip": args.clip,
        "rnn_type" : args.rnn_type,
        "bidirectional" : args.bidirectional,
        "enc_embed_dim":args.enc_emb_dim,
        "hidden_dim": args.hidden_dim,
        "dec_embed_dim":args.dec_emb_dim,
        "dropout_prob": args.dropout,
        "num_layers": args.num_layers,
        "teacher_forcing_ratio" :args.teacher_forcing_ratio,
        "optimizer": args.optimizer
    }



sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'validation loss',
            'goal': 'minimize'
        },
        'parameters':{
            "lr":{
              "values":[0.01, 0.001, 0.0001]
              },
            "batch_size":{
              "values":[16,64,128, 256]
              },
            "epochs":{
              "values":[50]
              },
            "clip":{
            "values":[0.5,1]
            },

            "rnn_type":{
              "values":["GRU"]
              },

            "enc_embed_dim":{
              "values":[64, 128, 256]
              },
            "dec_embed_dim":{
              "values":[32, 64,128, 256, 512]
              },
              

            "hidden_dim":{
              "values":[128, 256,512,1024 ]
              },
                      
            
            "dropout_prob":{
              "values":[0.3, 0.5, 0.7]
              },

            "teacher_forcing_ratio":{
                "values":[0.3, 0.5, 0.7]
            },         

            "optimizer":{
              "values":["adam", "nadam", "rmsprop"]
              }
      }
}

#path for training and testing files
train_path = "mal/mal_train.csv"
valid_path = "mal/mal_valid.csv"
test_path  =  "mal/mal_test.csv"
pred_path = "mal_predict.csv"

#path to store the results
correct_path = "with_attention/correct.csv"
incorrect_path = "with_attention/incorrect.csv"
#correct_path = "with_attention/correct.csv"
#incorrect_path = "with_attention/incorrect.csv"


#set wandb_sweep to True to perform hyper parameter sweep
wandb_sweep = args.sweep
#set project name
project_name =args.wandb_project
#set entity name
entity_name = args.wandb_entity
#set True if testing to be done or not
is_test = args.test
#set True if show heat maps





def transliterate():
  """Perform English to Malayalam Transliteration"""
  #initalize wandb with default configuaration
  wandb.init(config=config_defaults)
  config=wandb.config
  config = config_defaults
    
  
  #set wandb run name
  wandb.run.name = "e_{}_lr_{}_bs_{}_c_{}_rnn_{}_bi_{}_eed_{}_ded_{}_hd_{}_nl_{}_dr_{}_tf_{}_o_{}".format(
  config["epochs"],
  config["lr"],
  config["batch_size"],
  config["clip"],
  config["rnn_type"],
  config["bidirectional"],
  config["enc_embed_dim"],
  config["dec_embed_dim"],
  config["hidden_dim"],
  config["num_layers"],
  config["dropout_prob"],
  config["teacher_forcing_ratio"],
  config["optimizer"]
  ) 
    

  #get datasets, indices and vocabulary  
  train_iterator, valid_iterator, test_iterator, source, target = get_datasets(device, train_path, valid_path, test_path,\
                                                                               config["batch_size"])
  inp_dim = len(source.vocab) #input vocabulary size
  out_dim = len(target.vocab) #output vocabulary size
  if config["rnn_type"]=="GRU":
    model, optimizer, criterion = create_model_GRU(inp_dim,out_dim, config["enc_embed_dim"],\
          config["dec_embed_dim"], config["hidden_dim"], config["dropout_prob"],\
          source, target, config["lr"],config["optimizer"])
  
  for epoch in range(config["epochs"]):
  
      
      
      train_loss = train(model, train_iterator, optimizer, criterion, config["clip"], \
                         config["teacher_forcing_ratio"])
      valid_loss = evaluate(model, valid_iterator, criterion)
      
      print(f'Epoch: {epoch}  Train Loss: {train_loss:.3f}  Val. Loss: {valid_loss:.3f} ')
      wandb.log({'train_loss':train_loss, 'valid_loss': valid_loss})
  

  
  
  valid_acc = predict(device, model, valid_iterator, source, target, valid_path, correct_path,\
                          incorrect_path, args.save_pred, args.show_att)
  print(f'  valid acc:{valid_acc:.3f}' )
  wandb.log({'valid_accuracy':valid_acc})
  if(args.test):
    test_loss = evaluate(model, test_iterator, criterion)
    test_acc = predict(device, model, test_iterator, source, target, test_path, correct_path,\
                          incorrect_path, args.save_pred, args.show_att)
  
    print(f' Test Loss: {test_loss:.3f}' )
    wandb.log({'test_accuracy':test_acc, 'test_loss': test_loss})
    pred_acc =  predict(device, model, test_iterator, source, target, pred_path, correct_path,\
                          incorrect_path, args.save_pred, args.show_att)
  
    wandb.log({ 'test_accuracy':test_acc, 'test_loss': test_loss})



if args.sweep:
    #set project name
    project_name = args.wandb_project
    #set entity name
    entity_name = args.wandb_entity
    #entity_name = "dlresearchdl"
    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep_config,project=project_name, entity=entity_name)
    wandb.agent(sweep_id, function = transliterate, count=100) #for starting the job
else:
   transliterate()