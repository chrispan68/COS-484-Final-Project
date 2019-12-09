from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNN_Model(nn.Module):
    """Basic RNN LSTM classifier

    @param embed_size (int): Size of word embedding
    @param hidden_size (int): Size of hidden vector
    @param vocab (List[str]): list of words
    """

    def __init__(self, embed_size, hidden_size, vocab_len):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_regions = 3
        self.num_time_periods = 2
        self.model_embeddings = nn.Embedding(
            vocab_len, embed_size, padding_idx=0)
        
        #layers
        self.rnn_lstm = nn.LSTM(embed_size , hidden_size, bidirectional=True)
        self.linear_region = nn.Linear(hidden_size , 2 , bias=False)
        self.softmax_region = nn.Softmax(1)
        self.linear_time = nn.Linear(hidden_size , 8 , bias=False)
        self.softmax_time = nn.Softmax(1)
        #

    """Forward props the RNN, returns both the output vector for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a list of lists
    
    """
    def forward(self , input_batch):
        
        return 0
        # default values
    def train(self , train_set , train_input):
        return 0
    def test(self , test_set , test_input):
        return 0
        
