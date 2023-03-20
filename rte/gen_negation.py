import openai
from read_data import load_RTE, load_SNLI, construct_corpus_types_set
from negate import Negator
import os
import json
import argparse

#openai.api_key = os.getenv("OPENAI_API_KEY")
start_sequence = "\nA:"
restart_sequence = "\n\nQ: "

def gen_negation(text):
    template = 'What is the negation of ' + '"[P]"' + '?'
    query = restart_sequence + template.replace("[P]", text) + start_sequence
    print(query)
    
    max_length = len(text.split(" "))
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        logprobs=1,
        temperature=0,
        max_tokens=max_length+10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"])
    print(response["choices"][0]["text"])
    return response["choices"][0]["text"], response

def run_on_RTE_pos(dataset, data_root, output_path):
    if dataset == "RTE":
        data = load_RTE(data_root)
        pos = data["pos"]
    
    f = open(output_path, "a")
    global pcount
    for item in pos[pcount:]:
        t = item["t"]
        h = item["h"]
        neg_t, response_t = gen_negation(t)
        neg_h, response_h = gen_negation(h)
        json.dump({"t":t, "neg_t":neg_t, "h":h, "neg_h":neg_h, "response_t":response_t, "response_h":response_h}, f)
        f.write("\n")
        pcount+=1
        print("pcount  :  " + str(pcount))
    f.close()
    return True

def run_on_NLI_pos(dataset, data_root, output_path):
    label = "neutral"
    
    if "NLI" in dataset:
        js = load_SNLI(data_root)
        # only keep the "entail" label directional subset
        #data = [i for i in js if i["gold_label"]=="entailment"] # just keep 200 samples
        #data = [i for i in js if i["gold_label"]=="neutral"]
        data = [i for i in js if i["gold_label"]==label]
    
    f = open(output_path, "a")
    
    stop_num = 0
    global pcount
    for item in data[pcount:-1]:
        t = item["sentence1"]
        h = item["sentence2"]
        neg_t, neg_h = "",""
        if args.use_openai:
            neg_t, response_t = gen_negation(t)
            neg_h, response_h = gen_negation(h)
        else:
            try:
                neg_t = negator.negate_sentence(t, prefer_contractions=False)
            except RuntimeError:
                print(t)
                pass  # skip unsupported sentence
            
            response_t = {}
            try:
                neg_h = negator.negate_sentence(h, prefer_contractions=False)
            except RuntimeError:
                print(h)
                pass  # skip unsupported sentence
            response_h = {}
        pcount += 1
        print("pcount  :  " + str(pcount))
        if neg_t == t or neg_h == h or neg_h=="" or neg_t=="":
            continue
        stop_num += 1
        json.dump({"t":t, "neg_t":neg_t, "h":h, "neg_h":neg_h, "response_t":response_t, "response_h":response_h, "label":label}, f)
        f.write("\n")
        if stop_num > 200:
            break
    f.close()
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RTE') #RTE, NLI
    parser.add_argument('--data_path', type=str, default="data/rte1_test.json")
    parser.add_argument('--gen_neg_path', type=str, default="data/genNEG_rte1_test.json")
    parser.add_argument('--use_openai', type=bool, default=False)
    #parser.add_argument('--replace_method', type=str, default="swap") # swap, pseudos, type
    #parser.add_argument('--ncandidates_per_sequence', type=int, default=1)
    #parser.add_argument('--max_search_time_per_sequence', type=int, default=20)
    #parser.add_argument('--use_filter', type=bool, default=True)
    #parser.add_argument('--filter_labels', type=list, default=["EVENT","DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"])
    args = parser.parse_args()
    
    if not args.use_openai:
        negator = Negator(use_transformers=True, fail_on_unsupported=True)
        
    pcount=0
    end = False
    while not end:
        try:
            if args.dataset == "RTE":
                end = run_on_RTE_pos(args.dataset, args.data_path, args.gen_neg_path)
            elif args.dataset == "NLI":
                end = run_on_NLI_pos(args.dataset, args.data_path, args.gen_neg_path)
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError")
    
    