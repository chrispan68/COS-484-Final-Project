from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from torch.utils.data import Dataset, TensorDataset , DataLoader



class RNN_Model(nn.Module):
    """Basic RNN LSTM classifier

    @param embed_size (int): Size of word embedding
    @param hidden_size (int): Size of hidden vector
    @param vocab (List[str]): list of words
    """

    def __init__(self, embed_size, hidden_size, vocab_len , epoch , learning_rate , batch_size):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.epoch_size = epoch
        self.batch_size = batch_size
        self.model_embeddings = nn.Embedding(
            vocab_len, embed_size, padding_idx=0)
        
        #layers
        self.rnn_lstm = nn.LSTM(embed_size , hidden_size, bidirectional=False)
        self.linear_region = nn.Linear(hidden_size , 2 , bias=False)
        self.linear_time = nn.Linear(hidden_size , 4 , bias=False)

        #functions and optimizers
        self.softmax = nn.Softmax(1)
        self.optimizer = torch.optim.Adam(self.parameters() , lr=learning_rate)
        self.cost = nn.CrossEntropyLoss()

    """Forward props the RNN, returns both the output tensor for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a tensor
    
    """
    def forward(self , input_batch):
        x = self.model_embeddings(input_batch).permute(1 , 0 , 2)
        o , _ = self.rnn_lstm(x , (torch.randn(1 , len(input_batch) , self.hidden_size) , torch.randn(1 , len(input_batch) , self.hidden_size)))
        rnn_output = o[-1]
        output_region = self.softmax(self.linear_region(rnn_output))
        output_time = self.softmax(self.linear_time(rnn_output))
        return output_region , output_time
        # default values
    def train(self , train_input , train_output_region , train_output_time):
        train_input = torch.from_numpy(train_input).long()
        train_output_region = torch.from_numpy(train_output_region).long()
        train_output_time = torch.from_numpy(train_output_time).long()
        train_data = TensorDataset(train_input , train_output_region , train_output_time)
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size)
        for epoc in range(0 , self.epoch_size):
            print(epoc)
            totalLossRegion = torch.tensor(0.0)
            totalLossTime = torch.tensor(0.0)
            for train_input , train_output_region , train_output_time in train_loader:
                self.optimizer.zero_grad()
                prediction_region , prediction_time = self.forward(train_input)
                loss = self.cost(prediction_region , train_output_region)
                loss.backward(retain_graph=True)
                totalLossRegion += loss
                loss = self.cost(prediction_time , train_output_time)
                loss.backward()
                totalLossTime += loss
                self.optimizer.step()
            print("Cross Entropy Loss for Region Classification:", totalLossRegion.tolist())
            print("Cross Entropy Loss for Time Period Classification:", totalLossTime.tolist())
        return
    def test(self , test_input , test_output_region , test_output_time):

        confusion_time = np.zeros((4 , 4))
        confusion_region = np.zeros((2 , 2))
        for i in range(0 , len(test_input)):
                prediction_region , prediction_time = self.forward(torch.tensor([test_input[i]]))
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
        
