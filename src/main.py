import math
import time
from model import RNN_Model
import numpy as numpy
from typing import List
from tqdm import tqdm
import torch
import torch.nn.utils

def process_data(source: str):
    return list_of_dicts
def add_labels(processed_data: List[dict]):
    return list_of_dicts , another_list_of_dicts
def post_compute()
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
    
    data = process_data(args.source)
    traindata , devdata = add_labels(data)
    
    model = RNN_Model(embed_size=args.embed_size,
                      hidden_size = args.hidden_size)
    model.train(traindata)
    model.test(devdata)