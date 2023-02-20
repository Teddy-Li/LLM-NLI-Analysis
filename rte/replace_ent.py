import spacy
import json
import random
from termcolor import colored, cprint
import wuggy
from random_word import RandomWords
import os
from generate_pseudowords import get_pseudos_from_phrase
import argparse 

        
def load_RTE(files):
    with open(files,"r") as f:
        return json.loads(f.read())

def load_SNLI(files):
    with open(files,"r") as f:
        return [json.loads(line) for line in f.readlines()]


def detect_ents(text):
    document = pipeline(text)
    # Entity text & label extraction
    ents_set = {}
    if args.use_filter: # filter some noise types
        for entity in document.ents:
            if entity.label_ in args.filter_labels:
                continue
            if entity.text not in ents_set:
                ents_set[entity.text] = entity.label_
    else:
        for entity in document.ents:
            if entity.text not in ents_set:
                ents_set[entity.text] = entity.label_
    return ents_set

def construct_corpus_types_set(data, dataset):
    # the input data is a json file with two keys, pos and neg. Both are lists. 
    # {"pos":[{"t":"...", "h":"...."}, ...], "neg":[{"t":"...", "h":"...."}, ...]}
    if dataset == "RTE":
        lines = data["pos"] + data["neg"]
        corpus = ""
        for line in lines:
            corpus += line["t"] + " " + line["h"] + "\n"
        doc = pipeline(corpus)
        for entity in doc.ents:
            if entity.label_ in args.filter_labels:
                continue
            if entity.label_ not in corpus_types_set:
                corpus_types_set[entity.label_] = set()
                corpus_types_set[entity.label_].add(entity.text)
            else:
                corpus_types_set[entity.label_].add(entity.text)
        
    elif dataset == "SNLI":
        corpus = ""
        for line in data:
            corpus += line["sentence1"] + " " + line["sentence2"] + "\n"
    
        for c in corpus.split("\n"):
            doc = pipeline(c)

            for entity in doc.ents:
                if entity.label_ not in corpus_types_set:
                    corpus_types_set[entity.label_] = set()
                    corpus_types_set[entity.label_].add(entity.text)
                else:
                    corpus_types_set[entity.label_].add(entity.text)
    return corpus_types_set

def choose_new_ent(ent, label):
    return random.sample(corpus_types_set[label] - set({ent}), 1)[0]

def check_entity_matching(ent_tokens, dict_, allow_mismatch=1):
    for item in dict_:
        if all(i in ent_tokens for i in dict_[item]) or all(i in dict_[item] for i in ent_tokens):
            return item
        elif len(set(ent_tokens))>=3 and len(set(dict_[item]))>=3 and (len(set(ent_tokens)-set(dict_[item]))<=allow_mismatch or len(set(dict_[item])-set(ent_tokens))<=allow_mismatch): 
        # if it is a long entity, we allow 1 mismatch token
            return item
    return

def type_mapping(label):
    spacy_mapping = {"PERSON":"Person", "NORP":"Nation", "FAC":"Buildings", "ORG":"Organization", "GPE":"Country", "LOC":":Locations","PRODUCT":"Product","WORK_OF_ART":"The Work", "LAW":"Law","LANGUAGE":"Language", "DATE":"Date",
"TIME":"Time", "PERCENT":"Number", "MONEY":"Number", "QUANTITY":"Number", "ORDINAL":"Ordinal", "CARDINAL":"Number"}
    return spacy_mapping[label]

def replace_ent_rte(data, model):
    pos = []
    neg = []
    not_change_count = 0
    count = 0
    
    for p in data["pos"]:
        h = p["h"]
        t = p["t"]
        rp_t = t
        rp_h = h
        cprint(colored("\n"+t, "green", attrs=['bold']))
        cprint(colored(h, "green", attrs=['bold']))
        
        mapping_entities = {}             
        ents_set = detect_ents(t + " " + h)
        ent_to_tokens = {}
        same_types = {}
        for ent in ents_set.keys():
            # split the entity to tokens
            split_ent_ = [i.text.replace(".","") for i in pipeline(ent)]
            if not check_entity_matching(split_ent_, ent_to_tokens): # if the tokens not match the exist entity, add it as a new entity
                ent_to_tokens[ent] = [i.text.replace(".","") for i in pipeline(ent)]
                label = ents_set[ent]
                if model == "pseudos":
                    r_ent, miss_flag = get_pseudos_from_phrase(ent, g, r, args.ncandidates_per_sequence, args.max_search_time_per_sequence) # default ncandidates_per_sequence=1, max_search_time_per_sequence=20
                elif model == "swap":
                    r_ent = choose_new_ent(ent, label)
                elif model == "type":
                    print(same_types)
                    if label not in same_types: # check if the dict has the types, if not add Type.1, else add Type.{n+1}
                        same_types[label] = 1
                    else:
                        same_types[label] += 1
                    r_ent = type_mapping(label) + str(".") + str(same_types[label])
                mapping_entities[ent] = r_ent
                cprint(colored(ent + "--->" + label, "green", attrs=['reverse'])) # print replacement
                cprint(colored(r_ent + "--->" + label, "blue", attrs=['reverse']))
                rp_t = rp_t.replace(ent, r_ent)
                rp_h = rp_h.replace(ent, r_ent)
            else:
                label = ents_set[ent]
                r_ent = mapping_entities[check_entity_matching(split_ent_, ent_to_tokens)] # if the tokens match the exist entity, using the former entity replacement
                cprint(colored(ent + "--->" + label, "red", attrs=['reverse']))
                cprint(colored(r_ent + "--->" + label, "blue", attrs=['reverse']))
                rp_t = rp_t.replace(ent, r_ent)
                rp_h = rp_h.replace(ent, r_ent)    
        if rp_t == t and rp_h == h:
            not_change_count += 1
            continue # if no replacement, remove the sentence.
        cprint(colored(rp_t, "blue", attrs=['bold']))
        cprint(colored(rp_h, "blue", attrs=['bold'])) 
        pos.append({"t":rp_t, "h":rp_h})
        count+=1
        print(count)
    
    for n in data["neg"]:
        h = n["h"]
        t = n["t"]
        rp_t = t
        rp_h = h
        cprint(colored("\n"+t, "green", attrs=['bold']))
        cprint(colored(h, "green", attrs=['bold']))
        
        mapping_entities = {}             
        ents_set = detect_ents(t + " " + h)
        ent_to_tokens = {}
        same_types = {}
        for ent in ents_set.keys():
            # split the entity to tokens
            split_ent_ = [i.text.replace(".","") for i in pipeline(ent)]
            if not check_entity_matching(split_ent_, ent_to_tokens): # if the tokens not match the exist entity, add it as a new entity
                ent_to_tokens[ent] = [i.text.replace(".","") for i in pipeline(ent)]
                label = ents_set[ent]
                if model == "pseudos":
                    r_ent, miss_flag = get_pseudos_from_phrase(ent, g, r, args.ncandidates_per_sequence, args.max_search_time_per_sequence) # default ncandidates_per_sequence=1, max_search_time_per_sequence=20
                elif model == "swap":
                    r_ent = choose_new_ent(ent, label)
                elif model == "type":
                    print(same_types)
                    if label not in same_types: # check if the dict has the types, if not add Type.1, else add Type.{n+1}
                        same_types[label] = 1
                    else:
                        same_types[label] += 1
                    r_ent = type_mapping(label) + str(".") + str(same_types[label])
                mapping_entities[ent] = r_ent
                cprint(colored(ent + "--->" + label, "green", attrs=['reverse']))
                cprint(colored(r_ent + "--->" + label, "blue", attrs=['reverse']))
                rp_t = rp_t.replace(ent, r_ent)
                rp_h = rp_h.replace(ent, r_ent)
            else:
                label = ents_set[ent]
                r_ent = mapping_entities[check_entity_matching(split_ent_, ent_to_tokens)] # if the tokens match the exist entity, using the former entity replacement
                cprint(colored(ent + "--->" + label, "red", attrs=['reverse']))
                cprint(colored(r_ent + "--->" + label, "blue", attrs=['reverse']))
                rp_t = rp_t.replace(ent, r_ent)
                rp_h = rp_h.replace(ent, r_ent)    
        if rp_t == t and rp_h == h:
            not_change_count += 1
            continue # if no replacement, remove the sentence.
        cprint(colored(rp_t, "blue", attrs=['bold']))
        cprint(colored(rp_h, "blue", attrs=['bold'])) 
        neg.append({"t":rp_t, "h":rp_h})
        count+=1
        print(count)
    return pos, neg, not_change_count

def replace_ent_snli(data, model):
    output = []
    count=0
    length = len(data)
    for p in data:
        print(str(count) + "  /  " +str(length))
        h = p["sentence2"]
        t = p["sentence1"]
        mapping_entities = {}            
        ents_set = detect_ents(t + " " + h)
        ent_to_tokens = {}
        for ent in ents_set.keys():
            # split the entity to tokens
            split_ent_ = [i.text.replace(".","") for i in pipeline(ent)]
            if not check_entity_matching(split_ent_, ent_to_tokens): # if the tokens not match the exist entity, add it as a new entity
                ent_to_tokens[ent] = [i.text.replace(".","") for i in pipeline(ent)]
                label = ents_set[ent]
                try:
                    r_ent = choose_new_ent(ent, label)
                except ValueError:
                    continue
                mapping_entities[ent] = r_ent
                t = t.replace(ent, r_ent)
                h = h.replace(ent, r_ent)
            else:
                label = ents_set[ent]
                r_ent = mapping_entities[check_entity_matching(split_ent_, ent_to_tokens)] # if the tokens match the exist entity, using the former entity replacement
                t = t.replace(ent, r_ent)
                h = h.replace(ent, r_ent)    
        count+=1
        output.append({"sentence1":t, "sentence2":h, "gold_label":p["gold_label"]})
    return output       
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RTE') #RTE, SNLI
    parser.add_argument('--data_root', type=str, default="data/rte1_test.json")
    parser.add_argument('--output_path', type=str, default="data/rte1_test.wuggy_ent.json")
    parser.add_argument('--replace_method', type=str, default="swap") # swap, pseudos, type
    parser.add_argument('--ncandidates_per_sequence', type=int, default=1)
    parser.add_argument('--max_search_time_per_sequence', type=int, default=20)
    parser.add_argument('--use_filter', type=bool, default=True)
    parser.add_argument('--filter_labels', type=list, default=["EVENT","DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"])
    args = parser.parse_args()
    
    pipeline = spacy.load("en_core_web_trf")
    print("model load")
    # store all entities of corpus, with the format corpus_types_set[label] = set(entities)
    corpus_types_set = {}
    #filter_labels=["EVENT","DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"] # ignore the types "EVENT"
    
    if args.replace_method == "pseudos":
        g = wuggy.WuggyGenerator()
        g.load('orthographic_english')
        r = RandomWords()
    #  RTE
    if args.dataset == "RTE":
        data = load_RTE(args.data_root)
    
    print("Data readed")    
    corpus_types_set = construct_corpus_types_set(data, args.dataset)
    print("Entity Dict Generated")
    pos, neg, not_change_count = replace_ent_rte(data, args.replace_method)
    
    
    with open(args.output_path, "w") as f:
        json.dump({"pos":pos, "neg":neg}, f)
        f.write("\n")
    
    print("not_change_count :    " + str(not_change_count) +" "+ str(len(pos)) +" "+ str(len(neg)))
    """
    
    #  SNLI
    files = "/disk/scratch_big/liang/OpenAI/openai-quickstart-python/SNLI/snli_1.0/snli_1.0_test.jsonl"
    output_files = "/disk/scratch_big/liang/OpenAI/openai-quickstart-python/SNLI/snli_1.0/snli_1.0_test.r_ent.jsonl"
    data = load_SNLI(files)
    
    construct_corpus_types_set(data, "SNLI")
    output = replace_ent_snli(data)
    print(random.sample(output, 1)[0])
    print(len(output))
    
    with open(output_files, "w") as f:
        json.dump({"pos":pos, "neg":neg}, f)
        f.write("\n") 
    """