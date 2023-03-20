import os
import os.path
from os import path
import openai
import json
import time
import random
import argparse
import spacy
from read_data import load_GoogleRE, load_BoolQA

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nA:"
restart_sequence = "\n\nQ: "
#REPLACEMENT= ["Steve Jobs", "Donald Trump", "Barack Obama", "Michael Jackson", "Taylor Swift", "Rihanna" ,"Bill Gates","Justin Bieber", "Kanye West", "Kim Kardashian", "Oprah Winfrey", "Angelina Jolie", "The Rock", "Brad Pitt", "Cristiano Ronaldo", "Katy Perry", "Lady Gaga", "George Clooney", "Ariana Grande", "Emma Watson", "Jennifer Lopez"] # an entities list rank by frequency

def gen_query_googlere(ent, rel, context=""):
    prompts = {"place_of_birth":"Where was [P] born?", "date_of_birth":"When was [P] born?", "place_of_death":"Where did [P] die?"}
    if args.use_replace:
        rp_ent = random.choice(REPLACEMENT)
        print(type(ent),type(context),type(rp_ent))
        print(ent, rp_ent)
        #rp_ent(ent, text, rp_ent)
        rp_context = replace_ent(ent, context, rp_ent)
        query = restart_sequence + rp_context + " " + prompts[rel].replace("[P]", rp_ent) + start_sequence
    else:
        query = restart_sequence + context + " " + prompts[rel].replace("[P]", ent) + start_sequence
    return query

def replace_ent(entity, text, new_entity):
    document = pipeline(text)
    ents = document.ents
    person = []
    for e in ents:
        if e.label_ == "PERSON":
            person.append(e.text)
    rp_ent = {}
    for p in person:
        if p in entity: # check if match
            rp_ent[p] = new_entity
        elif entity in p:
            rp_ent[p] = new_entity
        else:
            continue
    print(rp_ent)
    for key in rp_ent:
        #print(text)
        text = text.replace(key, new_entity)
    return text

def query_GPT_googlere(data, output):
    if not path.exists(output):
        os.mkdir(output)
    fo =open(output + args.model +"_"+str(args.use_context)+"_"+str(args.use_replace)+".json", "a")
    
    global pcount
    for d in data[pcount:]:
        rel = d["pred"].split("/")[-1]
        context = d["evidences"][0]["snippet"]
        query_ent = d["sub_label"]
        target_ent = d["obj_label"]
        if args.use_context:
            query = gen_query_googlere(query_ent, rel, context)
        else:
            query = gen_query_googlere(query_ent, rel, "")
        
        response = openai.Completion.create(
            model=args.model,
            prompt=query,
            logprobs=1,
            temperature=0,
            max_tokens=20,#len(query)+20,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"])
        
        time.sleep(0.5)
        pcount+=1
        print("pcount  :  " + str(pcount))
        
        i = {"predict":response["choices"][0]["text"], "rel": rel, "model":args.model, "query":query, "response":response, "original":d}
        json.dump(i,fo)
        fo.write("\n")
    fo.close()
    return True

def query_GPT_boolqa(data, output):
    if not path.exists(output):
        os.mkdir(output)
    fo =open(output + args.model +"_"+str(args.use_context)+"_"+str(args.use_replace)+".json", "a")
    
    global pcount
    for d in data[pcount:]:
        rel = d["rel"]
        query = restart_sequence + d["answer"] +" "+ d["question"] +start_sequence
        print(query)
        if len(query.split(" ")) > 4096:
            pcount+=1
            continue
        response = openai.Completion.create(
            model=args.model,
            prompt=query,
            logprobs=1,
            temperature=0,
            max_tokens=1,#len(query)+20,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"])
        
        time.sleep(0.5)
        pcount+=1
        print("pcount  :  " + str(pcount))
        
        i = {"predict":response["choices"][0]["text"], "rel": rel, "model":args.model, "query":query, "response":response, "original":d}
        json.dump(i,fo)
        fo.write("\n")
    fo.close()
    return True
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/LAMA/Google_RE/")
    parser.add_argument("--dataset", type=str, default="googlere")
    parser.add_argument('--output_dir', type=str, default="data/LAMA/result_googlere/")
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--rels', type=list, default=["place_of_birth", "date_of_birth", "place_of_death"])
    parser.add_argument('--use_context', type=bool, default=False)
    parser.add_argument('--use_replace', type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    
    #pipeline = spacy.load("en_core_web_trf")
    #print("------ model load ------")
    
    if args.use_replace:
        with open("data/top40_freq_person.txt", "r") as f:
            REPLACEMENT = [line.strip() for line in f.readlines()]
    
    if args.debug:
        test_new_ent = "Steve Jobs XX"
        #test_entity = "Khatchig Mouradian"
        #text = "Khatchig Mouradian is a journalist, writer and translator born in Lebanon. He was one of the junior editors of the Lebanese-Armenian daily newspaper Aztag from 2000 to 2007, when he moved to Boston and became the editor of the Armenian Weekly. Mouradian holds a B.S. in biology and has studied towards a graduate degree in clinical psychology. He is working towards a PhD in Genocide Studies at Clark University http://www.clarku.edu/departments/holocaust/phd/research.cfm."
        #test_entity = "Art Murakowski"
        #text = "Murakowski died in 1985 at age 60 at his home in Hammond, Indiana. He was survived by his wife, Lucille Murakowski, three sons, and three daughters."
        test_entity = "Anna Eliza Bray"
        text = "A year or two after Stothard died, Anna Eliza married Edward Atkyns Bray, the vicar of Tavistock. She then began writing novels, and from 1826 to 1874, produced at least a dozen. Some of these, such as The Talba, or the Moor of Portugal dealt with foreign life, but she based her most popular novels on the principal families (the Trelawneys of Trelawne, the Pomeroys, and the Courtenays of Walreddon) of the counties of Devon and Cornwall. They were historical novels, and proved so popular that they were issued in a set of ten volumes by Longmans in 1845-6, and were reprinted by Chapman & Hall as recently as 1884. Her second husband died in 1857, and Mrs. Bray then removed to London, where she selected and edited some of her late husband's poetry and sermons, and then returned to original work. Her last years were embittered by a report that during a visit to Bayeux in 1816, she stole a piece of that city's famous tapestry. However, her character was cleared by correspondence and leading articles that appeared in the Times. After a long life in literary labours, she died in London on 21 Jan. 1883. Her autobiography, to 1843, was published by her nephew, Mr. John A. Kempe, in 1884--but it is sketchy, and less than accurate. It depicts an accomplished and kindly woman, proud of her own creations, and enthusiastic in praise of the literary characters she knew."        
        print(text + "\n")
        query = gen_query_googlere(test_entity, "place_of_birth", context=text)
        print(query)
    
    if not args.debug:
        if args.dataset == "googlere":
            rels_files = [rel+"_test.jsonl" for rel in args.rels]
            data = load_GoogleRE(args.data_path, rels_files)
            print(len(data))

            pcount = 0
            end = False
            while not end:
                try:
                    end = query_GPT_googlere(data, args.output_dir)
                except openai.error.RateLimitError:
                    print("openai.error.RateLimitError")
        
        if args.dataset == "boolqa":
            data = load_BoolQA("data/BoolQA/pos.json", "data/BoolQA/neg.json")
            print(len(data))
            
            pcount = 211
            end = False
            while not end:
                try:
                    end = query_GPT_boolqa(data, args.output_dir)
                except openai.error.RateLimitError:
                    print("openai.error.RateLimitError")
                #except openai.error.APIConnectionError:
                #    print("openai.error.RateLimitError")