# English-Malayalam Transliteration


<!-- ABOUT THE PROJECT -->
## Description

This directory contains the implementation of Sequence to Sequence Models for Transliteration from English to Malayalam using attention mechanism and without attention. 

## To install

<ul>
  <li>Python3.8</li>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>Wandb</li>
  <li>torchtext==0.6</>
</ul>

## Contents
The directory contains follwing  folders:
<ul>
  <li> Seq2SeqVanilla: contains the files for sequence to sequence translation using RNN, LSTM and GRU. The files are as follows:</li>
      <ul>
              <li> main.py: The main file of the code. You have to run this file to run the model.</li>
              <li> data.py: code for data preprocessing, creating train, valid and test sets</li>
              <li> train_test.py: contains functions for training and testing the model</li>
              <li> rnn_model.py: contain classes of RNN encoder, decoder and functions for creating a sequence to sequence model using RNN</li>
              <li> gru_model.py: contain classes of GRU encoder, decoder and functions for creating a sequence to sequence model using GRU</li>
              <li> lstm_model.py: contain classes of LSTM encoder, decoder and functions for creating a sequence to sequence model using LSTM</li>  
              <li>mal: folder containing trainig, testing and validation files. The datafiles are in csv formats with source and targets.</li>
                          <ul> 
                                   <li> mal_train.csv: training set </li>
                                   <li> mal_test.csv: test set </li>
                                   <li> mal_valid.csv: validation set </li>
                                   <li> malpredict.csv: file containg words to predict and their ground truth</li>
                          </ul>           
        </ul>
  
  <li> Seq2SeqAttention: contains the files for sequence to sequence transliteration using bidirectional GRU.</li>
        <ul>
              <li> main.py: The main file of the code. You have to run this file to run the model.</li>
              <li> data.py: code for data preprocessing, creating train, valid and test sets</li>
              <li> train_test.py: contains functions for training and testing the model</li>
              <li> gru_model.py: contain classes of GRU encoder, decoder,attention and functions for creating a sequence to sequence model using GRU</li>
              <li>mal: folder containing trainig, testing and validation files. The datafiles are in csv formats with source and targets.</li>
                          <ul> 
                                   <li> mal_train.csv: training set </li>
                                   <li> mal_test.csv: test set </li>
                                   <li> mal_valid.csv: validation set </li>
                                   <li> malpredict.csv: file containg words to predict and their ground truth</li>
                          </ul>           
               </ul>
        </ul>
   <li>  Predictions Vanilla : Predictions made by the RNN models without attention</li>. The csv files are  
          <ul>
                  <li> correct.csv</li>
                  <li> incorrect.csv</li>
          </ul>   
   <li>  Predictions Attention:  Predictions made by the RNN models without attention</li>. The csv files are  
          <ul>
                  <li> correct.csv</li>
                  <li> incorrect.csv</li>
          </ul>   
  </ul>  
  
  
     
     
     
## To run the seq2seq model without attention
In main.py the neural network is initialized by the best hyperparameter setting obtained. 
To train the model with the same hyperparamer settings:
```
python main.py 
```
To train the netowork with new hyperparameter settings, the hyperparameters have to passed as command line arguments as shown below

``` 
python  -o adam -lr 0.0002

```
The above arguments will train RNN with , adam optimizer and learning rate 0.0002

### Supported arguments
|Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | seq2seq | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | deep-learning-assignment | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|  `-e`, `--epochs` | 100 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` |512 | Batch size used to train neural network. |
|  `-o`, `--optimizer` | adaam | choices:  [ "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-nl`, `--num_layers` | 2 | Number of hidden layers used in feedforward neural network. | 
| 'd', '--dropout'|0.5 | droput probability|
| 'tf', '--teacher_forcing_ratio'| 0.5|teacher forcing ratio decoder|
|"-eed", "--enc_emb_dim | Embedding dimension of encoder|
|"-ded", "--dec_emb_dim"| Embedding dimension of decoder|
|"-hd", "--hidden_dim"| Hidden dimension of encoder and decoder|
|"-bi", "--bidirectional"|, Choices:True, False|
|"-sa", "--show_attn"|Show attention maps|
|"-sp", "--save_pred"| Save the predictions|
|"-test","--test"|"choices: True if perform test, else False", default=True|
|"-sweep","--sweep"| "choices: True if perform wandb sweep, else False", default=True)  |

## To run the seq2seq modle with attention
In main.py the neural network is initialized by the best hyperparameter setting obtained. 
To train the model with the same hyperparamer settings:
```
python main.py 
```
To train the netowork with new hyperparameter settings, the hyperparameters have to passed as command line arguments as shown below

``` 
python  -o adam -lr 0.0002

```
The above arguments will train RNN with , adam optimizer and learning rate 0.0002

### Supported arguments
|Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | seq2seq | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | deep-learning-assignment | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|  `-e`, `--epochs` | 100 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` |512 | Batch size used to train neural network. | 
|  `-o`, `--optimizer` | adaam | choices:  [ "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-nl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| 'd', '--dropout'|0.5 | droput probability|
| 'tf', '--teacher_forcing_ratio'| 0.5|teacher forcing ratio decoder|
|"-eed", "--enc_emb_dim |256| Embedding dimension of encoder|
|"-ded", "--dec_emb_dim"| 256| Embedding dimension of decoder|
|"-hd", "--hidden_dim"| Hidden dimension of encoder and decoder|
|"-sa", "--show_attn"|Show attention maps|
|"-sp", "--save_pred"| Save the predictions|
|"-test","--test"|"choices: True if perform test, else False", default=True|
|"-sweep","--sweep"| "choices: True if perform wandb sweep, else False", default=True)  |
 ```
 
 
    
  
   
