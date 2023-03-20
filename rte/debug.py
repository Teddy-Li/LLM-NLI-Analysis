import spacy
from queryGPT_GoogleRE import rp_ent


if __name__ == '__main__':
    pipeline = spacy.load("en_core_web_trf")
    new_ent = "Steve Jobs XX"
    ent = "Khatchig Mouradian"
    text =  "Khatchig Mouradian is a journalist, writer and translator born in Lebanon. He was one of the junior editors of the Lebanese-Armenian daily newspaper Aztag from 2000 to 2007, when he moved to Boston and became the editor of the Armenian Weekly. Mouradian holds a B.S. in biology and has studied towards a graduate degree in clinical psychology. He is working towards a PhD in Genocide Studies at Clark University http://www.clarku.edu/departments/holocaust/phd/research.cfm."
    rp = rp_ent(ent, text, new_ent)
    print(rp)