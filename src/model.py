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

    def __init__(self, embed_size, hidden_size, vocab):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.model_embeddings = nn.Embedding(
            len(vocab), embed_size, padding_idx=0)
        

        # default values
    def train(self , train_set , train_input):
        return 0
    def test(self , test_set , test_input):
        return 0
        
