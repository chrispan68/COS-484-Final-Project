from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from torch.utils.data import Dataset, TensorDataset , DataLoader



class RNN_Multitask(nn.Module):
    """Basic RNN LSTM multiclass classifier

    @param embed_size (int): size of word embedding
    @param hidden_size (int): size of hidden vector
    @param vocab_len (int): size of our vocabulary
    @param epoch (int): number of epocs run through
    @param learning_rate (int): learning rate of optimization
    @param batch_size (int): Number of examples per batch
    @param numRegions (int): Number of total regions we are classifying from
    @param numPeriods (int): Number of total time periods we are classifying from
    """

    def __init__(self, embed_size, hidden_size, vocab_len , epoch , learning_rate , batch_size , numPeriods , numRegions):
        super(RNN_Multitask, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.epoch_size = epoch
        self.batch_size = batch_size
        self.model_embeddings = nn.Embedding(
            vocab_len, embed_size, padding_idx=0)
        self.numPeriods = numPeriods
        self.numRegions = numRegions
        
        #layers
        self.rnn_lstm = nn.LSTM(embed_size , hidden_size, bidirectional=True)
        self.linear_region = nn.Linear(hidden_size*2 , numRegions , bias=False)
        self.linear_time = nn.Linear(hidden_size*2 , numPeriods , bias=False)

        #functions and optimizers
        self.softmax = nn.Softmax(1)
        self.optimizer = torch.optim.Adam(self.parameters() , lr=learning_rate)
        self.cost = nn.CrossEntropyLoss()

    """Forward props the RNN, returns both the output tensor for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a tensor
    
    """
    def forward(self , input_batch):
        x = self.model_embeddings(input_batch).permute(1 , 0 , 2)
        o , _ = self.rnn_lstm(x , (torch.randn(2 , len(input_batch) , self.hidden_size) , torch.randn(2 , len(input_batch) , self.hidden_size)))
        rnn_output = o[-1]
        output_region = self.softmax(self.linear_region(rnn_output))
        output_time = self.softmax(self.linear_time(rnn_output))
        return output_region , output_time
        # default values

class RNN_Singletask(nn.Module):
    """Basic RNN LSTM single task classifier. This model is actually two 
    RNN's since we need to do two problems simultaneously. 

    @param embed_size (int): size of word embedding
    @param hidden_size (int): size of hidden vector
    @param vocab_len (int): size of our vocabulary
    @param epoch (int): number of epocs run through
    @param learning_rate (int): learning rate of optimization
    @param batch_size (int): Number of examples per batch
    @param numRegions (int): Number of total regions we are classifying from
    @param numPeriods (int): Number of total time periods we are classifying from
    """

    def __init__(self, embed_size, hidden_size, vocab_len , epoch , learning_rate , batch_size , numPeriods , numRegions):
        super(RNN_Singletask, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.epoch_size = epoch
        self.batch_size = batch_size
        self.model_embeddings = nn.Embedding(
            vocab_len, embed_size, padding_idx=0)
        self.numPeriods = numPeriods
        self.numRegions = numRegions
        
        #layers
        self.rnn_lstm_region = nn.LSTM(embed_size , hidden_size, bidirectional=True)
        self.rnn_lstm_time = nn.LSTM(embed_size , hidden_size, bidirectional=True)
        self.linear_region = nn.Linear(2*hidden_size , numRegions , bias=False)
        self.linear_time = nn.Linear(2*hidden_size , numPeriods, bias=False)

        #functions and optimizers
        self.softmax = nn.Softmax(1)
        self.optimizer = torch.optim.Adam(self.parameters() , lr=learning_rate)
        self.cost = nn.CrossEntropyLoss()

    """Forward props the RNN, returns both the output tensor for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a tensor
    
    """
    def forward(self , input_batch):
        x = self.model_embeddings(input_batch).permute(1 , 0 , 2)   
        o , _ = self.rnn_lstm_region(x , (torch.randn(2 , len(input_batch) , self.hidden_size) , torch.randn(2 , len(input_batch) , self.hidden_size)))
        output_region = self.softmax(self.linear_region(o[-1]))
        o , _ = self.rnn_lstm_time(x , (torch.randn(2 , len(input_batch) , self.hidden_size) , torch.randn(2 , len(input_batch) , self.hidden_size)))
        output_time = self.softmax(self.linear_time(o[-1]))
        return output_region , output_time
        # default values

"""
This method takes in a model, training input, and training output, and trains the model 
based on its training parameters

@param model (nn.Module): The model being trained
@train_input (numpy array): The training input examples
@train_output_region (numpy array): The training region labels
@train_output_time (numpy array): The training time period labels. 
"""
def train(model , train_input , train_output_region , train_output_time):
    train_input = torch.from_numpy(train_input).long()
    train_output_region = torch.from_numpy(train_output_region).long()
    train_output_time = torch.from_numpy(train_output_time).long()
    train_data = TensorDataset(train_input , train_output_region , train_output_time)
    train_loader = DataLoader(dataset=train_data, batch_size=model.batch_size)
    for epoc in range(0 , model.epoch_size):
        print(epoc)
        totalLossRegion = torch.tensor(0.0)
        totalLossTime = torch.tensor(0.0)
        for train_input , train_output_region , train_output_time in train_loader:
            model.optimizer.zero_grad()
            prediction_region , prediction_time = model.forward(train_input)
            loss = model.cost(prediction_region , train_output_region)
            loss.backward(retain_graph=True)
            totalLossRegion += loss
            loss = model.cost(prediction_time , train_output_time)
            loss.backward()
            totalLossTime += loss
            model.optimizer.step()
        print("Cross Entropy Loss for Region Classification:", totalLossRegion.tolist())
        print("Cross Entropy Loss for Time Period Classification:", totalLossTime.tolist())
    return

"""
This method takes in a model, testing input, and testing output, and computes
the confusion matrix for both the region classification and time preiod
classification problems. 

@param model (nn.Module): The model being tested
@train_input (numpy array): The testing input examples
@train_output_region (numpy array): The testing region labels
@train_output_time (numpy array): The testing time period labels. 
"""
def test(model , test_input , test_output_region , test_output_time):

    confusion_time = np.zeros((model.numPeriods , model.numPeriods))
    confusion_region = np.zeros((model.numRegions , model.numRegions))
    for i in range(0 , len(test_input)):
            prediction_region , prediction_time = model.forward(torch.tensor([test_input[i]]))
            pregion = torch.argmax(prediction_region, dim=1).tolist()[0]
            ptime = torch.argmax(prediction_time, dim=1).tolist()[0]
            confusion_time[ptime][test_output_time[i]] += 1
            confusion_region[pregion][test_output_region[i]] += 1
    print("Confusion matrix for time periods:")
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) 
        for row in confusion_time.tolist()]))
    print("Confusion matrix for region:")
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) 
        for row in confusion_region.tolist()]))
    return
