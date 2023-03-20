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
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, auc

def load_data(path):
    with open(path, "r") as f:
        data = [json.loads(i) for i in f.readlines()]
    return data
    
"""      
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
"""

def evaluate(data):
    def judger(pred: str, label: str) -> bool:
        if label.lower() == "yes":
            if label.lower() in pred.lower():
                return True
            else:
                return False
        if label == "no":
            if label.lower() in pred.lower():
                return True
            else:
                return False
    
    def score_(logprob: float, pred: str):
        if pred is None:
            return False, 0.0
        elif judger(pred, "True"):
            # print("!")
            assert 0 < math.exp(logprob) < 1
            effective_scr = 0.5 + 0.5*math.exp(logprob)
            return True, effective_scr
        elif judger(pred, "False"):
            assert 0 < math.exp(logprob) < 1
            effective_scr = 0.5 - 0.5*math.exp(logprob)
            return False, effective_scr
        else:
            return False, 0.0
    
    # calculate AUC
    y_true = []
    y_pred = []
    print(len(data))
    for i in data:
        label = i["original"]["label"]
        if i["original"]["label"] == "Yes":
            y_true.append(0)
        if i["original"]["label"] == "No":
            y_true.append(1)
        istrue, pred = score_(i["response"]["choices"][0]["logprobs"]["token_logprobs"][0], i["predict"])
        y_pred.append(pred)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred) 
    auc_score = auc(recalls, precisions)
    
    # calculate precision, recall, f-1
    tp, tn, fp, fn = 0,0,0,0
    for item in data:
        label = item["original"]["label"]
        pred = item["predict"]
        if label.lower() == "yes":
            if label.lower() in pred.lower():
                tp += 1
            else:
                fn += 1
        elif label.lower() == "no":
            if label.lower() in pred.lower():
                tn += 1
            else:
               fp += 1
    
    print("AUC : " + str(round(auc_score, 2)))
    print("Precision : " + str(round(100*tp/(tp+fp), 2)))
    print("Recall : " + str(round(100*tp/(tp+fn), 2))) 
    print("F-1 : " + str(round(100*2*tp/ (2*tp + fp + fn), 2)))
    print("Accuracy : " + str(round(100*(tp+tn)/(tp+fp+fn+tn), 2)))
    print("true positives : ", str(tp))
    print("true negatives : ", str(tn))
    print("false positives : ", str(fp))
    print("false negatives : ", str(fn))
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="data/BoolQA/result_boolqa/")
    parser.add_argument('--filename', type=str, default="text-davinci-003_False_False.json")
    parser.add_argument('--use_subset', type=bool, default=False)
    parser.add_argument('--draw_image', type=bool, default=False) # not finished in this demo
    parser.add_argument('--write_errors', type=bool, default=False) # not finished in this demo
    args = parser.parse_args()
    
    data = load_data(args.results_dir+args.filename)
    evaluate(data)