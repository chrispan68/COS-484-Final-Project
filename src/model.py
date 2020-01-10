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

    def __init__(self, embed_size, hidden_size, vocab_len , epoch , learning_rate , batch_size , numPeriods , numRegions , bidirectional=True , layers=1 , dropout=0):
        super(RNN_Multitask, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.epoch_size = epoch
        self.batch_size = batch_size
        self.model_embeddings = nn.Embedding(
            vocab_len, embed_size, padding_idx=0)
        self.numPeriods = numPeriods
        self.numRegions = numRegions
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.numLayers = layers
        self.numDirections = 2 if self.bidirectional else 1
        #layers
        self.rnn_lstm = nn.LSTM(embed_size , hidden_size, bidirectional=self.bidirectional , num_layers=self.numLayers , dropout=self.dropout)
        self.linear_region = nn.Linear(hidden_size*self.numDirections , numRegions , bias=False)
        self.linear_time = nn.Linear(hidden_size*self.numDirections , numPeriods , bias=False)
        #functions and optimizers
        self.softmax = nn.Softmax(1)
        self.optimizer = torch.optim.Adam(self.parameters() , lr=learning_rate)
        self.cost = nn.CrossEntropyLoss()

    """Forward props the RNN, returns both the output tensor for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a tensor
    
    """
    def forward(self , input_batch):
        x = self.model_embeddings(input_batch).permute(1 , 0 , 2)
        
        o , _ = self.rnn_lstm(x , (torch.randn(self.numDirections * self.numLayers , len(input_batch) , self.hidden_size) , torch.randn(self.numDirections * self.numLayers , len(input_batch) , self.hidden_size)))
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

    def __init__(self, embed_size, hidden_size, vocab_len , epoch , learning_rate , batch_size , numPeriods , numRegions , bidirectional=True , layers=1 , dropout=0):
        super(RNN_Singletask, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.epoch_size = epoch
        self.batch_size = batch_size
        self.model_embeddings = nn.Embedding(
            vocab_len, embed_size, padding_idx=0)
        self.numPeriods = numPeriods
        self.numRegions = numRegions
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.numLayers = layers
        self.numDirections = 2 if self.bidirectional else 1
        
        #layers
        self.rnn_lstm_region = nn.LSTM(embed_size , hidden_size, bidirectional=self.bidirectional , num_layers=self.numLayers , dropout=self.dropout)
        self.rnn_lstm_time = nn.LSTM(embed_size , hidden_size, bidirectional=self.bidirectional , num_layers=self.numLayers , dropout=self.dropout)
        self.linear_region = nn.Linear(self.numDirections*hidden_size , numRegions , bias=False)
        self.linear_time = nn.Linear(self.numDirections*hidden_size , numPeriods, bias=False)

        #functions and optimizers
        self.softmax = nn.Softmax(1)
        self.optimizer = torch.optim.Adam(self.parameters() , lr=learning_rate)
        self.cost = nn.CrossEntropyLoss()

    """Forward props the RNN, returns both the output tensor for the location and the period. 


    @param input_batch, a maxl by batch_size input that is a tensor
    
    """
    def forward(self , input_batch):
        x = self.model_embeddings(input_batch).permute(1 , 0 , 2)   
        o , _ = self.rnn_lstm_region(x , (torch.randn(self.numLayers*self.numDirections , len(input_batch) , self.hidden_size) , torch.randn(self.numLayers*self.numDirections , len(input_batch) , self.hidden_size)))
        output_region = self.softmax(self.linear_region(o[-1]))
        o , _ = self.rnn_lstm_time(x , (torch.randn(self.numLayers*self.numDirections , len(input_batch) , self.hidden_size) , torch.randn(self.numLayers*self.numDirections , len(input_batch) , self.hidden_size)))
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
    
    #Computing precision, recall, f1
    oSumRegion = sum(confusion_region)
    pSumRegion = sum(confusion_region.T)
    trueRegion = np.diagonal(confusion_region)
    oSumTime = sum(confusion_time)
    pSumTime = sum(confusion_time.T)
    trueTime = np.diagonal(confusion_time)
    
    #Precision uses pSum, Recall uses oSum
    precisionRegion = trueRegion / pSumRegion
    recallRegion = trueRegion / oSumRegion
    precisionTime = trueTime / pSumTime
    recallTime = trueTime / oSumTime
    f1Region = (2 * precisionRegion * recallRegion) / (precisionRegion + recallRegion)
    f1Time = (2 * precisionTime * recallTime) / (precisionTime + recallTime)

    #print out precision and recall
    print("Summary statistics for Region Classification:")
    print("Confusion Matrix:")
    print('\n'.join([''.join(['{:7}'.format(item) for item in row]) 
        for row in confusion_region.tolist()]))
    print("Precision:")
    print('\n'.join([''.join(['{:7}'.format(round(item , 4)) for item in precisionRegion])])) 
    print("Recall:")
    print('\n'.join([''.join(['{:7}'.format(round(item , 4)) for item in recallRegion])]))
    print("F1 Score:")
    print('\n'.join([''.join(['{:7}'.format(round(item , 4)) for item in f1Region])]))
    print()
    print("Summary statistics for Time Period Classification:")
    print("Confusion Matrix:")
    print('\n'.join([''.join(['{:7}'.format(round(item , 4)) for item in row]) 
        for row in confusion_time.tolist()]))
    print("Precision:")
    print('\n'.join([''.join(['{:7}'.format(round(item , 4)) for item in precisionTime])])) 
    print("Recall:")
    print('\n'.join([''.join(['{:7}'.format(round(item , 4)) for item in recallTime])]))
    print("F1 Score:")
    print('\n'.join([''.join(['{:7}'.format(round(item , 4)) for item in f1Time])]))


    return
