from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


class RNN_Model(nn.Module):
    """Basic RNN LSTM classifier

    @param embed_size (int): Size of word embedding
    @param hidden_size (int): Size of hidden vector
    @param vocab (List[str]): list of words
    """

    def __init__(self, embed_size, hidden_size, vocab_len , epoch , learning_rate):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_regions = 3
        self.num_time_periods = 2
        self.epoch_size = epoch
        self.model_embeddings = nn.Embedding(
            vocab_len, embed_size, padding_idx=0)
        
        #layers
        self.rnn_lstm = nn.LSTM(embed_size , hidden_size, bidirectional=False)
        self.linear_region = nn.Linear(hidden_size , 2 , bias=False)
        self.linear_time = nn.Linear(hidden_size , 8 , bias=False)

        #functions and optimizers
        self.softmax = nn.Softmax(1)
        self.optimizer = torch.optim.Adam(self.parameters() , lr=learning_rate)
        self.cost = nn.CrossEntropyLoss()

    """Forward props the RNN, returns both the output tensor for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a list of lists
    
    """
    def forward(self , input_batch):
        x = torch.tensor(input_batch).t()
        x = self.model_embeddings(x)
        o , (h_final , c)= self.rnn_lstm(x , (torch.randn(1 , len(input_batch) , self.hidden_size) , torch.randn(1 , len(input_batch) , self.hidden_size)))
        rnn_output = o[3]
        output_region = self.softmax(self.linear_region(rnn_output))
        output_time = self.softmax(self.linear_time(rnn_output))
        return output_region , output_time
        # default values
    def train(self , train_input , train_output_region , train_output_time):
        for epoc in range(0 , self.epoch_size):
            print(epoc)
            for i in range(0 , len(train_input)):
                self.optimizer.zero_grad()
                batch_input = train_input[i]
                prediction_region , prediction_time = self.forward(batch_input)
                batch_output_region = torch.tensor(train_output_region[i])
                batch_output_time = torch.tensor(train_output_time[i])
                loss = self.cost(prediction_region , batch_output_region)
                loss.backward(retain_graph=True)
                loss = self.cost(prediction_time , batch_output_time)
                loss.backward()
                self.optimizer.step()
        return
    def test(self , test_input , test_output_region , test_output_time):

        confusion_time = np.zeros((8 , 8))
        confusion_region = np.zeros((2 , 2))
        for i in range(0 , len(test_input)):
                prediction_region , prediction_time = self.forward([test_input[i]])
                pregion = torch.argmax(prediction_region, dim=1).tolist()[0]
                ptime = torch.argmax(prediction_time, dim=1).tolist()[0]
                confusion_time[ptime][test_output_time[i]] += 1
                confusion_region[pregion][test_output_region[i]] += 1
        print("Confusion matrix for time:")
        print('\n'.join([''.join(['{:6}'.format(item) for item in row]) 
            for row in confusion_time.tolist()]))
        print("Confusion matrix for region:")
        print('\n'.join([''.join(['{:6}'.format(item) for item in row]) 
            for row in confusion_region.tolist()]))
        return
        
