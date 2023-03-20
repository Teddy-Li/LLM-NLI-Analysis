import os
import os.path
from os import path
import openai
import json
import time
import random
import argparse
import spacy
import matplotlib.pyplot as plt

def load_data(path):
    with open(path, "r") as f:
        data = [json.loads(i) for i in f.readlines()]
    return data
    
      
def evaluate(data):
    tp = 0
    for item in data:
        pred = item["predict"]
        labels = item["original"]["obj_aliases"]
        labels.append(item["original"]["obj_label"])
        for token in labels:
            if token in pred:
                tp+=1
                break
    print("\n Precision at 1 on whole dataset:    " + str(tp/len(data)))
        
    if args.use_subset:
        place_of_birth = [i for i in data if i["rel"]=="place_of_birth"]
        date_of_birth = [i for i in data if i["rel"]=="date_of_birth"]
        place_of_death = [i for i in data if i["rel"]=="place_of_death"]
        subsets = {"place_of_birth":place_of_birth, "date_of_birth":date_of_birth, "place_of_death":place_of_death}
        for rel in subsets:
            tp_subset = 0
            for item in subsets[rel]:
                pred = item["predict"]
                labels = item["original"]["obj_aliases"]
                labels.append(item["original"]["obj_label"])
                #debug_find_flag = False
                for token in labels:
                    if token in pred:
                        tp_subset+=1
                        debug_find_flag = True
                        break
                #if not debug_find_flag:
                #    print(pred)
                #    print(labels)
            print("\n Precision at 1 on "+rel+":    " + str(tp_subset/len(subsets[rel])))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="data/LAMA/result_googlere/")
    parser.add_argument('--filename', type=str, default="text-davinci-003_False_False.json")
    parser.add_argument('--use_subset', type=bool, default=False)
    parser.add_argument('--draw_image', type=bool, default=False)
    parser.add_argument('--write_errors', type=bool, default=False)
    args = parser.parse_args()
    
    
    data = load_data(args.results_dir+args.filename)
    evaluate(data)