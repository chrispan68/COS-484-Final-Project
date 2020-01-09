import math
import pickle
import time
import argparse
from model import RNN_Model
import numpy as numpy
from typing import List
from tqdm import tqdm
import torch
import torch.nn.utils
import re
from collections import OrderedDict
import random

def validate(x):
    return (not x['period'] == None) and (not x['region'] == None)


if __name__ == '__main__':
    assert(torch.__version__ == "1.3.0"), \
        "Please update your installation of PyTorch. " \
        "You have {} and you should have version 1.3.0".format(
            torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="basic", type=str,
                        choices=["basic"], help="the model version to be used")
    parser.add_argument("--source", default=None, type=str,
                        help="source file, with metadata")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_size", default=64, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--epoch_size", default=50, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    args = parser.parse_args()
    
    
    test_set = pickle.load(open("test_set_titles.pickle" , "rb"))
    train_set = pickle.load(open("train_set_titles.pickle" , "rb"))
    test_set = [x for x in test_set if validate(x)]
    train_set = [x for x in train_set if validate(x)]

    region2id = {}
    region2id["US"] = 0
    region2id["UK"] = 1
    print(train_set[0])
    period2id = {}
    numPeriods = 0
    maxl = 25
    for train_ex in train_set:
        if not train_ex['period'] == None:
            if not train_ex['period'] in period2id:
                period2id[train_ex['period']] = numPeriods
                numPeriods += 1
    for ex in train_set:
        ex["region_id"] = region2id[ex["region"]]
        ex["time_id"] = period2id[ex["period"]]
        ex["words"] = ex["long_text"][0].split()
    for ex in test_set:
        ex["region_id"] = region2id[ex["region"]]
        ex["time_id"] = period2id[ex["period"]]
        ex["words"] = ex["long_text"][0].split()

    vocab = set()
    for ex in train_set:
        for word in ex["words"]:
            vocab.add(word)


    word2ind = {}
    word2ind["<pad>"] = 0
    word2ind["<start>"] = 1
    word2ind["<end>"] = 2
    word2ind["<unknown>"] = 3
    vocab_size = 4
    for word in vocab:
        word2ind[word] = vocab_size
        vocab_size += 1
    
    train_input = [] #Num Examples x maxL 
    train_output_region = [] #Num Examples
    train_output_time = []#Num Examples
    test_input = [] #Num Examples x maxL
    test_output_region = [] #Num Examples
    test_output_time = [] #Num examples
    batch = [] #a temporary list that holds the current batch
    for train_ex in train_set:
        train_output_region.append(train_ex["region_id"])
        train_output_time.append(train_ex["time_id"])

        #formats the input via dictionary 
        tmp = []
        tmp.append(1)
        for word in train_ex["words"]:
            tmp.append(word2ind[word])
        tmp.append(2)
        for i in range(0 , maxl + 2 - len(train_ex["words"])):
            tmp.append(0)
        train_input.append(tmp)
    
    for test_ex in test_set:
        test_output_region.append(test_ex["region_id"])
        test_output_time.append(test_ex["time_id"])

        #formats the input via dictionary
        tmp = []
        tmp.append(1)
        for word in test_ex["words"]:
            if(word in word2ind):
                tmp.append(word2ind[word])
            else:
                tmp.append(3)
        tmp.append(2)
        for i in range(0 , maxl + 2 - len(test_ex["words"])):
            tmp.append(0)
        test_input.append(tmp)
    print(train_set[0])
    print(train_input[0])
    print(train_output_region[0])
    print(train_output_time[0])

    model = RNN_Model(embed_size=args.embed_size,
                      hidden_size=args.hidden_size,
                      vocab_len=vocab_size,
                      epoch=args.epoch_size,
                      learning_rate=args.lr, 
                      batch_size=args.batch_size)
    model.train(numpy.asarray(train_input) , numpy.asarray(train_output_region) , numpy.asarray(train_output_time)) 
    model.test(test_input , test_output_region , test_output_time)

    print("train results:")
    model.test(train_input , train_output_region , train_output_time)
