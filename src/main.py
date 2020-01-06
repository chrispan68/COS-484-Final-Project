import math
import time
import argparse
from model import RNN_Singletask
from model import RNN_Multitask
import numpy as numpy
from typing import List
from tqdm import tqdm
import torch
import torch.nn.utils
import re
from collections import OrderedDict
import random
import pickle

"""
This method validates a dictionary containing a training example, ensuring that it contains 
both a region and a time label. 

@param x, the example being validated.
@return a boolean that indicates whether its accepted or not. 
"""
def validate(x):
    return (not x['period'] == None) and (not x['region'] == None)

"""
This method takes in a list of dictionaries and alters each element's 'long_text' field to 
be a list of all words in a sentence rather than individual sentences.

@param example_set, the list of dictionaries that serve as input to be altered. 
"""
def tokenize(example_set):
    for example in example_set:
        words = []
        for sentence in example["long_text"]:
            words += sentence.split()
            words += "."
        example["long_text"] = words

"""
This method takes in the train_set, a list of dictionaries containing our training examples, 
and extracts out a set of words from which to build a dictionary. 

@param train_set, the list of dictionaries representing the training set
@return vocab, the set of all words encountered in training set. 
"""
def getVocab(train_set):
    vocab = set()
    for train_ex in train_set:
        for word in train_ex['long_text']:
            vocab.add(word)
    
    return vocab

"""
This method takes in both the training set and the test set, and calculates the maximum length
entry found in either and returns it. This is useful for batching. 

@param train_set, the list of dictionaries corresponding to the training set
@param test_set, the list of dictionaries corresponding to the testing set
@return maxL, the length of the longest passage found in either training of testing. 
"""
def getMaxL(train_set , test_set):
    maxL = 0
    for train_ex in train_set:
        maxL = max(maxL , len(train_ex['long_text']))
    for test_ex in test_set:
        maxL = max(maxL , len(test_ex['long_text']))
    return maxL

"""
This method takes in the training set and extracts out a dictionary that is used to convert regions to 
label indices. 

@param train_set, the set of all training examples. 
@return region2id, the dictionary used to convert regions to indices. 
"""
def getRegionLabels(train_set):
    region2id = {}
    numRegions = 0
    for train_ex in train_set:
        if not train_ex['region'] == None:
            if not train_ex['region'] in region2id:
                region2id[train_ex['region']] = numRegions
                numRegions += 1
    return region2id , numRegions

"""
This method takes in the training set and extracts out a dictionary that is used to convert periods to 
label indices. 

@param train_set, the set of all training examples. 
@return period2id, the dictionary used to convert periods to indices. 
"""
def getPeriodLabels(train_set):
    period2id = {}
    numPeriods = 0
    for train_ex in train_set:
        if not train_ex['period'] == None:
            if not train_ex['period'] in period2id:
                period2id[train_ex['period']] = numPeriods
                numPeriods += 1
    return period2id , numPeriods

"""
This is a function that takes in either the training set or the test set. 
It uses a set of examples to create matrices of word indices to represent sentences. 
It pads input and replaces unknown tokens with the unknown word index. 

@param example_set, the list of dictionaries representing the example set
@param word2id, the dictionary that converts words into word indices. 
@param period2id, the dictionary that converts time periods to period indices. 
@param region2id, the dictionary that converts region to region indices. 
@return input, the list of lists containing all the input passages to the RNN
@return output_region, the list of region ids 
@return output_period, the list of period ids
"""
def getInput(example_set , word2id , period2id , region2id , maxL):
    input_vec = []
    output_region = []
    output_period = []

    for example in example_set:
        output_region.append(region2id[example['region']])
        output_period.append(period2id[example['period']])

        #formats via dictionary 
        tmp = []
        tmp.append(word2id["<start>"])
        cnter = 0
        for word in example["long_text"]:
            if(cnter < maxL):
                if word in word2id:
                    tmp.append(word2id[word])
                else:
                    tmp.append(word2id['<unknown>'])
                cnter += 1
        tmp.append(word2id["<end>"])
        for i in range(0 , maxL + 2 - min(maxL , len(example["long_text"]))):
            tmp.append(word2id["<pad>"])
        input_vec.append(tmp)
    return input_vec , output_region , output_period



if __name__ == '__main__':
    assert(torch.__version__ == "1.3.0"), \
        "Please update your installation of PyTorch. " \
        "You have {} and you should have version 1.3.0".format(
            torch.__version__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="basic", type=str,
                        choices=["basic" , "single_task"], help="the model version to be used")
    parser.add_argument("--sourcetest", default=None, type=str,
                        help="source test file, with metadata")
    parser.add_argument("--sourcetrain", default=None, type=str,
                        help="source train file, with metadata")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_size", default=64, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--epoch_size", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--cap", default=float('INF'), type=int)
    args = parser.parse_args()
    
    test_set = pickle.load(open(args.sourcetest , "rb"))
    train_set = pickle.load(open(args.sourcetrain , "rb"))
    test_set = [x for x in test_set if validate(x)]
    train_set = [x for x in train_set if validate(x)]
    tokenize(test_set)
    tokenize(train_set)

    vocab = getVocab(train_set)
    maxL = min(getMaxL(train_set , test_set) , args.cap)

    word2id = {}
    word2id["<pad>"] = 0
    word2id["<start>"] = 1
    word2id["<end>"] = 2
    word2id["<unknown>"] = 3
    vocab_size = 4
    for word in vocab:
        word2id[word] = vocab_size
        vocab_size += 1
    
    region2id , numRegions = getRegionLabels(train_set)
    period2id , numPeriods = getPeriodLabels(train_set)
    
    train_input , train_output_region , train_output_time = getInput(train_set , word2id , period2id , region2id , maxL)
    test_input , test_output_region , test_output_time = getInput(test_set , word2id , period2id , region2id , maxL) 
    
    if args.model_type == "basic":
        model = RNN_Multitask(embed_size=args.embed_size,
                          hidden_size=args.hidden_size,
                          vocab_len=vocab_size,
                          epoch=args.epoch_size,
                          learning_rate=args.lr, 
                          batch_size=args.batch_size,
                          numPeriods=numPeriods,
                          numRegions=numRegions)
    elif args.model_type == "single_task":
        model = RNN_Singletask(embed_size=args.embed_size,
                          hidden_size=args.hidden_size,
                          vocab_len=vocab_size,
                          epoch=args.epoch_size,
                          learning_rate=args.lr, 
                          batch_size=args.batch_size,
                          numPeriods=numPeriods,
                          numRegions=numRegions)

    model.train(numpy.asarray(train_input) , numpy.asarray(train_output_region) , numpy.asarray(train_output_time)) 
    model.test(test_input , test_output_region , test_output_time)