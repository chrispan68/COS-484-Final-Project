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
        self.rnn_lstm = nn.LSTM(embed_size , hidden_size, bidirectional=False , batch_first=True)
        self.linear_region = nn.Linear(hidden_size , 2 , bias=False)
        self.softmax_region = nn.Softmax(1)
        self.linear_time = nn.Linear(hidden_size , 8 , bias=False)
        self.softmax_time = nn.Softmax(1)
        #

    """Forward props the RNN, returns both the output tensor for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a list of lists
    
    """
    def forward(self , input_batch):
        x = torch.tensor(input_batch)
        x = self.model_embeddings(x)
        o , (h_final , c)= self.rnn_lstm(x)
        h_final = torch.squeeze(h_final , dim=0)
        output_region = self.linear_region(h_final)
        output_time = self.linear_time(h_final)

        print(str(output_region.size()))
        print(str(output_time.size()))
        
        return output_region , output_time
        # default values
    def train(self , train_input , train_output_region , train_output_time):
        self.forward(train_input[0])
        return 0
    def test(self , test_input , test_output_region , test_output_time):
        return 0
        
