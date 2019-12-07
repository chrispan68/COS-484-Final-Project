import math
import time
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
  print(text.readline())  # do not remove this line

  data = []
  ctr = 0
  for line in text:
    new_data = []
    items = re.split('(,)(?=(?:[^"]|"[^"]*")*$)', line)

    try:
      year = int(items[16])
    except ValueError:
      print("NOT INT", items)
      continue

    text = items[14]
    text = text.replace('"', ' ')
    city = items[18].strip("[").strip("]").strip(":")
    city = "".join(city.split())
    new_data = {"id": items[0], "text": text, "year": year, "city": items[18], "region": None, "period": None} # [id, text, date, place] 
    data.append(new_data)


  # Get region: British, American, or other
  us_list = ["New York", "Chicago", "Philadelphia", "New-York", "N.Y.", "N.H.", "N.J.", "Calif.",
            "N. Y.", "Sacramento", "MN", "Mich", "Washington", "Sacramento", "Pa.", "Columbus", 
            "Madison", "Albany", "Ind.", "Tex", "Tenn" "Ohio", "Urbana", "Portland","Boston", 
            "St. Louis", "Cincinnati", "Baltimore", "Washington, D.C.", "Andover", "Richmond", 
            "Rochester", "Louisville", "New Haven", "Indianapolis", "Greensboro", "Raleigh", 
            "Nashville", "San Francisco", "Los Angeles", "Mass.", "Me.", "New Orleans", "Tenn.",
            "Atlanta", "Minn", "Berkeley", "Kansas", "Syracuse", "Providence", "Lincoln", "Wis.", 
            "New-Haven", "Buffalo", "Pittsburgh", "Ill.", "Detroit", "Ann Arbor"]
  uk_list = ["London", "Cambridge", "Oxford", "Glasgow", "Liverpool", "Edinburg", 
            "Eng.", "York", "Edinburgh", "Westminster","Belfast", "Dublin", "Manchester", 
            "Abingdon", "Birmingham", "Cork", "Newcastle", "Stratford-upon-Avon", "Perth", 
            "Southampton", "Bath", "Eton", "Lond."]
  europe_list = ["Paris", "Berlin", "Leipzig", "Strassburg", "Zurich", "Lund", "Heidelberg",
                "Madrid", "Bologna", "Göttingen", "Groningen", "Uppsala"] #do NOT switch order of US/UK checks

  counts = [0,0,0, 0]

  for doc in data:
    categorized = False
    for city in us_list:
      if city in doc["city"]:
        doc["region"] = "US"
        counts[0]+=1 
        categorized = True
    
    if not categorized:
      for city in uk_list:
        if city in doc["city"]:
          doc["region"] = "UK"
          counts[1]+=1 
          categorized = True

    if not categorized:
      for city in europe_list:
        if city in doc["city"]:
          doc["region"] = "Europe"
          counts[2]+=1 
          categorized = True

    if not categorized:
      doc["region"] = "UNK"
      counts[3] += 1

  # Get time period counts
  period_counts = OrderedDict({(1550, 1799): [0, 0], (1800, 1842): [0, 0], (1843, 1859): [0, 0], 
                              (1860, 1874): [0, 0], (1875, 1890): [0, 0], (1891, 1901): [0, 0],
                              (1902, 1912): [0, 0], (1913, 1923): [0, 0]})


  for index, doc in enumerate(data):
    for period in period_counts:
      if doc['year'] >= period[0] and doc['year'] <= period[1]:
        data[index]['period'] = period
        
        if doc['region'] == "US":
          period_counts[period][0] += 1
        else:
          period_counts[period][1] += 1
        break

  print("Overall period counts:", period_counts)

  # Keep only British and American documents; sample evenly from US and UK
  random.shuffle(data)

  uk_data = [x for x in data if x['region']=="UK"]
  us_data = [x for x in data if x['region']=="US"][:len(uk_data)]


  split = int(.8*len(uk_data))
  train_set = uk_data[:split] + us_data[:split]
  test_set = uk_data[split:] + us_data[split:]

  random.shuffle(train_set)
  random.shuffle(test_set)
  print("TRAIN SET", train_set)
  print("TEST SET", test_set)

  print("---INFO---")
  print("Train set data by period ("+ str(len(train_set))+ " total examples):")
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
  print(period_counts)

  print("Test set data by period ("+ str(len(test_set)) + " total examples):")
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

  print(period_counts)

  # Then return val dict and train dict


  return train_set, test_set, period_counts


def post_compute(labeled_data , word2ind):
    
if __name__ == '--main--':
    assert(torch.__version__ == "1.3.0"), \
        "Please update your installation of PyTorch. " \
        "You have {} and you should have version 1.3.0".format(torch.__version__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="basic", type=str, choices=["basic"] , help="the model version to be used")
    parser.add_argument("--source", default=None, type=str, help="source file, with metadata")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_size", default=64, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    
    data , vocab = process_data(args.source)
    word2ind = {}
    for i in range(0 , len(vocab)):
        word2ind[vocab[i]] = i
    
    traindata , devdata = add_labels(data)
    traindata = traindata
    
    model = RNN_Model(embed_size=args.embed_size,
                      hidden_size = args.hidden_size , 
                      vocab)
    model.train(traindata)
    model.test(devdata)