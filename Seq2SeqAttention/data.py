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




from train_test import *

from gru_model import *


def tokenize(word):
    """Split the word in to list of characters"""
    return list(word)


def get_datasets(device, train_path, valid_path, test_path, batch_size=2):
    """
    Perform train-test-valid split, create vocabulary, chars to indices, indices to char and data iterator
    """

    # Create the pytext's Field
    source_field = Field(tokenize=tokenize,
                    init_token='<sos>', 
                    eos_token='<eos>', 
                    pad_token="<pad>",
                    unk_token="<unk>",
                    lower=False)
    target_field = Field(tokenize=tokenize,
                    init_token='<sos>',
                    eos_token='<eos>',
                    pad_token="<pad>",
                    unk_token="<unk>",
                    )

    # Splits the data in Train, Test and Validation data
    train_set, valid_set, test_set = TabularDataset.splits(
        path="",
        train=train_path,
        validation=valid_path,
        test=test_path,
        format="csv",
        csv_reader_params={"delimiter": ",", "skipinitialspace": True},
        fields=[("src", source_field), ("trg", target_field)],)

    # Build the vocabulary for both the language
    source_field.build_vocab(train_set, min_freq=3)
    target_field.build_vocab(train_set, min_freq=3)

    # Create the Iterator using builtin Bucketing
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_set,\
                                                valid_set, test_set),
                                                batch_size=batch_size,
                                                sort_within_batch=True,
                                                sort_key=lambda x: len(x.src),
                                                device=device)
    return train_iterator, valid_iterator, test_iterator, source_field, target_field