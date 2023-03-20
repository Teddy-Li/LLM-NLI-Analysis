import json

def load_RTE(files):
    with open(files,"r") as f:
        return json.loads(f.read())

def load_SNLI(files):
    with open(files,"r") as f:
        return [json.loads(line) for line in f.readlines()]

def load_GoogleRE(files, rels):
    data = []
    for rel in rels:
        with open(files+rel, "r") as f:
            data += [json.loads(i) for i in f.readlines()]
    return data

def load_BoolQA(pfile, nfile):
    with open(pfile, "r") as f:
        pos = [json.loads(i) for i in f.readlines()]
    with open(nfile, "r") as f:
        neg = [json.loads(i) for i in f.readlines()]
    for i in pos:
        i["label"] = "Yes"
    for i in neg:
        i["label"] = "No"
    return pos+neg


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