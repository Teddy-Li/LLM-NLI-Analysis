import os
import openai
import json
import time
import argparse

#openai.api_key = os.getenv("OPENAI_API_KEY")
start_sequence = "\nA:"
restart_sequence = "\n\nQ: "
patterns = [
  "[P], which means that [H]. Is that true or false?", 
  "If [P], then [H]. Is that true or false?",
  "[H] because [P]. Is that true or false?",
  "[P], so [H]. Is that true or false?",
  "It is not the case that [H], let alone [P]. Is that true or false?", 
  "Does '[P]' entail '[H]'?"]

#pcount=96 #pattern2
pcount=0
ncount=0

def preprocess_RTE_datas(path, key):
    # we are using RTE-1 now
    with open(path, "r") as f:
        js = json.loads(f.read())
    pos = []
    neg = []
    pattern = patterns[key]
    for p in js["pos"]:
        t = p["t"][:-1]
        h = p["h"][:-1]
        prompt = restart_sequence + pattern.replace("[P]", t).replace("[H]", h) + start_sequence
        pos.append(prompt)
    for n in js["neg"]:
        t = n["t"][:-1]
        h = n["h"][:-1]
        prompt = restart_sequence + pattern.replace("[P]", t).replace("[H]", h) + start_sequence
        neg.append(prompt)
    return pos,neg

def preprocess_snli_data(path,key):
    # "/disk/scratch_big/liang/OpenAI/openai-quickstart-python/SNLI/snli_1.0/snli_1.0_test.jsonl"
    with open(path, "r") as f:
        js = [json.loads(line) for line in f.readlines()]
    filter_sent = []
    for item in js:
        label = item["gold_label"]
        t = item["sentence1"]
        h = item["sentence2"]
        prompt = restart_sequence + pattern.replace("[P]", t).replace("[H]", h) + "\n(A) entailment" + "\n(B) neutral" + "\n(C) contradiction" + start_sequence
        filter_sent.append({"label": label, "prompt":prompt})
    return filter_sent

# for rte dataset fommat
def run_rte(inPath, outPath, key):
    pos, neg = preprocess_RTE_datas(inPath, key)
    print(len(pos))
    print(len(neg))
    output = []
    pattern = patterns[key]
    f = open(outPath+"."+"p"+str(key),"a")

    #pcount = 96
    global pcount
    global ncount 
    time.sleep(1)
    for pro in pos[pcount:]:
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=pro,
        logprobs=1,
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"])
        
        time.sleep(0.5)
        pcount+=1
        print("pcount  :  " + str(pcount))
        item = {"predict":response["choices"][0]["text"], "label": "True", "model":"text-davinci-003", "prompt":pro, "pattern":pattern, "response":response}
        
        json.dump(item,f)
        f.write("\n")
    
    #ncount = 0
    for pro in neg[ncount:]:
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=pro,
        logprobs=1,
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"])
        
        time.sleep(0.5)
        ncount+=1
        print("ncount  :  " + str(ncount))
        item = {"predict":response["choices"][0]["text"], "label": "False", "model":"text-davinci-003", "prompt":pro, "pattern":pattern, "response":response}
        
        json.dump(item,f)
        f.write("\n")
        
    f.close()
    return True

def only_hypothesis(inPath, outPath):
    with open(inPath, "r") as f:
        js = json.loads(f.read())
    pos = js["pos"]
    neg = js["neg"]
    print(len(pos))
    print(len(neg))
    output = []
    f = open(outPath,"a")

    global pcount
    global ncount 
    time.sleep(1)
    for item in pos[pcount:]:
        pro = restart_sequence + item["h"] + "\n(A) true" + "\n(B) it's impossible to say" + "\n(C) false" + start_sequence
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=pro,
        logprobs=1,
        temperature=0,
        max_tokens=10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"])
        
        time.sleep(0.5)
        pcount+=1
        print("pcount  :  " + str(pcount))
        i = {"predict":response["choices"][0]["text"], "label": "True", "model":"text-davinci-003", "prompt":pro, "response":response}
        
        json.dump(i,f)
        f.write("\n")
    
    for item in neg[ncount:]:
        pro = restart_sequence + item["h"] + "\n(A) true" + "\n(B) it's impossible to say" + "\n(C) false" + start_sequence
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=pro,
        logprobs=1,
        temperature=0,
        max_tokens=10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"])
        
        time.sleep(0.5)
        ncount+=1
        print("ncount  :  " + str(ncount))
        i = {"predict":response["choices"][0]["text"], "label": "False", "model":"text-davinci-003", "prompt":pro, "response":response}
        
        json.dump(i,f)
        f.write("\n")
        
    f.close()
    return True

    
# for snli dataset fommat
def run_snli(inPath, outPath, key):
    pos = preprocess_snli_data(inPath, key)
    print(len(pos))
    output = []
    pattern = patterns[key]
    f = open("p"+str(key)+"_"+outPath,"a")
    
    #pcount = 96
    global pcount
    time.sleep(1)
    for pro in pos[pcount:]:
        text = pro["prompt"]
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text,
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"])
        
        #time.sleep(1.0)
        pcount+=1
        print("pcount  :  " + str(pcount))
        
        item = {"predict":response["choices"][0]["text"], "label": pro["label"], "model":"text-davinci-003", "prompt":pro, "pattern":pattern, "response":response}
        
        json.dump(item,f)
        f.write("\n")
        
    f.close()
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=2) #RTE, SNLI
    args = parser.parse_args()
    
    run_ = ["rte-1", "snli-1.0", "only-hypothesis"][args.run] # choose dataset
    if run_ == "rte-1":
        for pattern_key in [5,1,0,2,3,4]:
            print("processing pattern : " + str(pattern_key))
            print(patterns[pattern_key])
            pcount=0
            ncount=0
            end = False
            while not end:
                try:
                    end = run_rte("data/rte1_test.swap_ent.json.subset", "data/logprobs_RTE1_test.swap_ent.json.subset", pattern_key)
                except openai.error.RateLimitError:
                    print("openai.error.RateLimitError")
    
    if run_ == "only-hypothesis":
        for model_type in ["original", "swap"]:
            print("############ "+model_type+" ############ ")
            pcount=0
            ncount=0
            end = False
            while not end:
                try:
                    end = only_hypothesis("data/rte1_test."+model_type+"_ent.json.subset", "data/only-hypothesis_rte1_test."+model_type+"_ent.json.subset")
                except openai.error.RateLimitError:
                    print("openai.error.RateLimitError")
    
    
    elif run_ == "snli-1.0":
        for pattern_key in [3,4,5]:
            print("processing pattern : " + str(pattern_key))
            print(patterns[pattern_key])
            pcount=0
            end = False
            while not end:
                try:
                    end = run_snli("SNLI/snli_test.r_ent.json", "snli_test.rp_ent.json", pattern_key)
                except openai.error.RateLimitError:
                    print("openai.error.RateLimitError")