import math
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


def process_data(fpath: str):
    text = open(fpath, 'r', encoding='utf8')
    text.readline() # do not remove this line

    data = []
    maxl = 0
    vocab = set()
    for line in text:
        new_data = []
        items = re.split('(,)(?=(?:[^"]|"[^"]*")*$)', line)

        try:
            year = int(items[16])
        except ValueError:
            continue

        text = items[14]
        text = text.replace('"', ' ')
        words = text.split()
        maxl = max(maxl , len(words) + 2)
        for word in words:
            vocab.add(word)
        city = items[18].strip("[").strip("]").strip(":")
        city = "".join(city.split())
        new_data = {"id": items[0], "text": text, "year": year, "city": items[18],
                    "region": None, "period": None , "words": words , "region_id": None , "time_id": None}  # [id, text, date, place]
        data.append(new_data)

    # Get region: British, American, or other
    us_list = ["New York", "Chicago", "Philadelphia", "New-York", "N.Y.", "N.H.", "N.J.", "Calif.",
               "N. Y.", "Sacramento", "MN", "Mich", "Washington", "Sacramento", "Pa.", "Columbus",
               "Madison", "Albany", "Ind.", "Tex", "Tenn" "Ohio", "Urbana", "Portland", "Boston",
               "St. Louis", "Cincinnati", "Baltimore", "Washington, D.C.", "Andover", "Richmond",
               "Rochester", "Louisville", "New Haven", "Indianapolis", "Greensboro", "Raleigh",
               "Nashville", "San Francisco", "Los Angeles", "Mass.", "Me.", "New Orleans", "Tenn.",
               "Atlanta", "Minn", "Berkeley", "Kansas", "Syracuse", "Providence", "Lincoln", "Wis.",
               "New-Haven", "Buffalo", "Pittsburgh", "Ill.", "Detroit", "Ann Arbor"]
    uk_list = ["London", "Cambridge", "Oxford", "Glasgow", "Liverpool", "Edinburg",
               "Eng.", "York", "Edinburgh", "Westminster", "Belfast", "Dublin", "Manchester",
               "Abingdon", "Birmingham", "Cork", "Newcastle", "Stratford-upon-Avon", "Perth",
               "Southampton", "Bath", "Eton", "Lond."]
    europe_list = ["Paris", "Berlin", "Leipzig", "Strassburg", "Zurich", "Lund", "Heidelberg",
                   "Madrid", "Bologna", "Göttingen", "Groningen", "Uppsala"]  # do NOT switch order of US/UK checks

    counts = [0, 0, 0, 0]

    for doc in data:
        categorized = False
        for city in us_list:
            if city in doc["city"]:
                doc["region"] = "US"
                doc["region_id"] = 0
                counts[0] += 1
                categorized = True

        if not categorized:
            for city in uk_list:
                if city in doc["city"]:
                    doc["region"] = "UK"
                    doc["region_id"] = 1
                    counts[1] += 1
                    categorized = True

        if not categorized:
            for city in europe_list:
                if city in doc["city"]:
                    doc["region"] = "Europe"
                    doc["region_id"] = 2
                    counts[2] += 1
                    categorized = True

        if not categorized:
            doc["region"] = "UNK"
            counts[3] += 1

    # Get time period counts
    period_counts = OrderedDict({(1550, 1799): [0, 0], (1800, 1842): [0, 0], (1843, 1859): [0, 0],
                                 (1860, 1874): [0, 0], (1875, 1890): [0, 0], (1891, 1901): [0, 0],
                                 (1902, 1912): [0, 0], (1913, 1923): [0, 0]})

    for index, doc in enumerate(data):
        label = -1
        for period in period_counts:
            label += 1
            if doc['year'] >= period[0] and doc['year'] <= period[1]:
                data[index]['period'] = period
                data[index]['time_id'] = label
                if doc['region'] == "US":
                    period_counts[period][0] += 1
                else:
                    period_counts[period][1] += 1
                break

    # Keep only British and American documents; sample evenly from US and UK
    random.shuffle(data)

    uk_data = [x for x in data if x['region'] == "UK"]
    us_data = [x for x in data if x['region'] == "US"][:len(uk_data)]

    split = int(.8*len(uk_data))
    train_set = uk_data[:split] + us_data[:split]
    test_set = uk_data[split:] + us_data[split:]

    random.shuffle(train_set)
    random.shuffle(test_set)

    period_counts = OrderedDict({(1550, 1799): [0, 0], (1800, 1842): [0, 0], (1843, 1859): [0, 0],
                                 (1860, 1874): [0, 0], (1875, 1890): [0, 0], (1891, 1901): [0, 0],
                                 (1902, 1912): [0, 0], (1913, 1923): [0, 0]})
    for doc in train_set:
        for period in period_counts:
            if doc['year'] >= period[0] and doc['year'] <= period[1]:
                if doc['region'] == "US":
                    period_counts[period][0] += 1
                else:
                    period_counts[period][1] += 1
                break
    
    period_counts = OrderedDict({(1550, 1799): [0, 0], (1800, 1842): [0, 0], (1843, 1859): [0, 0],
                                 (1860, 1874): [0, 0], (1875, 1890): [0, 0], (1891, 1901): [0, 0],
                                 (1902, 1912): [0, 0], (1913, 1923): [0, 0]})
    for index, doc in enumerate(test_set):
        for period in period_counts:
            if doc['year'] >= period[0] and doc['year'] <= period[1]:

                if doc['region'] == "US":
                    period_counts[period][0] += 1

                else:
                    period_counts[period][1] += 1
                break

    # Then return val dict and train dict

    return vocab, train_set, test_set, period_counts , maxl


def post_compute(labeled_data, word2ind):
    return labeled_data

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
    
    args = parser.parse_args()
    vocab , train_set , test_set , period_counts , maxl = process_data(args.source)

    word2ind = {}
    word2ind["<pad>"] = 0
    word2ind["<start>"] = 1
    word2ind["<end>"] = 2
    word2ind["<unknown>"] = 3
    vocab_size = 4
    for word in vocab:
        word2ind[word] = vocab_size
        vocab_size += 1
    train_input = [] #(Num Examples / Batch) x Batch x maxL 
    train_output_region = [[[0]*2]*args.batch_size]*(math.ceil(len(train_set) / args.batch_size))#(Num Examples / Batch) x Batch x 2
    train_output_time = [[[0]*8]*args.batch_size]*(math.ceil(len(train_set) / args.batch_size))#(Num Examples / Batch) x Batch x 8
    test_input = [] #(Num Examples / Batch) x Batch x maxL
    test_output_region = [[[0]*2]*args.batch_size]*(math.ceil(len(test_set) / args.batch_size)) #(Num Examples / Batch) x Batch x 2
    test_output_time = [[[0]*8]*args.batch_size]*(math.ceil(len(test_set) / args.batch_size)) #(Num Examples / Batch) x Batch x 8
    batch = [] #a temporary list that holds the current batch
    index = 0
    for train_ex in train_set:
        if(len(batch) == args.batch_size):
            train_input.append(batch)
            batch = []
            index += 1
        train_output_region[index][len(batch)][train_ex["region_id"]] = 1
        train_output_time[index][len(batch)][train_ex["time_id"]] = 1
        tmp = []
        tmp.append(1)
        for word in train_ex["words"]:
            tmp.append(word2ind[word])
        for i in range(0 , maxl - 1 - len(train_ex["words"])):
            tmp.append(0)
        tmp.append(2)
        batch.append(tmp)
    if(len(batch) > 0):
        train_input.append(batch)
    
    batch = [] #a temporary list that holds the current batch
    index = 0
    for test_ex in test_set:
        if(len(batch) == args.batch_size):
            test_input.append(batch)
            batch = []
            index += 1
        test_output_region[index][len(batch)][test_ex["region_id"]] = 1
        test_output_time[index][len(batch)][test_ex["time_id"]] = 1
        tmp = []
        tmp.append(1)
        for word in test_ex["words"]:
            tmp.append(word2ind[word])
        for i in range(0 , maxl - 1 - len(test_ex["words"])):
            tmp.append(0)
        tmp.append(2)
        batch.append(tmp)
    if(len(batch) > 0):
        test_input.append(batch)
    
    model = RNN_Model(embed_size=args.embed_size,
                      hidden_size=args.hidden_size,
                      vocab_len=vocab_size)
    model.train(train_input , train_output_region , train_output_time) 
    model.test(test_input , test_output_region , test_output_time)