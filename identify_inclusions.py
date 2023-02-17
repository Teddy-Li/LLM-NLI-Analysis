import json
import argparse
import time
import sys
import openai
from utils import wrap_prompt
from typing import List


def is_subsequence(subseq: List[str], seq: List[str]):
    def fuzzy_match(a, b):
        if a == b:
            return True
        elif len(a)-len(b) == 1 and a[:-1] == b and a[-1] in ['s', 'd', 'e']:
            return True
        elif len(a)-len(b) == 2 and a[:-2] == b and a[-2:] in ['ed', 'es']:
            return True
        elif len(a)-len(b) == 3 and a[:-3] == b and a[-3:] in ['ing', 'ied', 'ies']:
            return True
        elif len(a)-len(b) == 4 and a[:-4] == b and a[-4:] in ['ying']:
            return True
        elif len(a) == len(b) and a[:-1] == b[:-1] and a[-1] in ['s', 'd', 'e'] and b[-1] in ['s', 'd', 'e']:
            return True
        elif len(a) == len(b) and len(a) > 3 and a[:-2] == b[:-2] and a[-2:] in ['ed', 'es'] and b[-2:] in ['ed', 'es']:
            return True
        elif len(a) == len(b) and len(a) > 3 and a[:-3] == b[:-3] and a[-3:] in ['ing', 'ied', 'ies'] and b[-3:] in ['ing', 'ied', 'ies']:
            return True
        else:
            return False

    if len(subseq) > len(seq):
        return False
    elif len(subseq) == 0:
        return True
    elif all([fuzzy_match(a, b) or fuzzy_match(b, a) for a, b in zip(subseq, seq)]):
        return True
    else:
        subseq_i = 0
        for subseq_i in range(len(subseq)):
            if subseq[subseq_i] in ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'be', 'been']:
                continue
            else:
                break
        if subseq_i == len(subseq):
            return True

        for i in range(len(seq) - (len(subseq)-subseq_i) + 1):
            if fuzzy_match(seq[i], subseq[subseq_i]) or fuzzy_match(subseq[subseq_i], seq[i]):
                if is_subsequence(subseq[subseq_i+1:], seq[i + 1:]):
                    return True
        return False


def get_paraphrase_from_gpt(proposition: str, model_name: str) -> str:
    text = f"""Change the tense of this sentence into present simple, do not change anything else:

old: {proposition}
new:"""
    prompt_dict = wrap_prompt(text, model_name=model_name, max_tokens=48)
    response = None
    for i in range(3):
        try:
            response = openai.Completion.create(**prompt_dict)
            break
        except Exception as e:
            print(f"Error: {e}")
            if i == 2:
                pass
            else:
                time.sleep(3)
                print(f"Retrying...")
                continue
    if response is None:
        print(f"Error: response is None", file=sys.stderr)
        return text.strip(' ')
    else:
        response_text = response['choices'][0]['text'].strip(' ')
        try:
            response_text, suffix = response_text.split(' Y')
            if len(suffix) > 0:
                print(f"Warning: suffix is not empty: {response_text} Y{suffix}", file=sys.stderr)
        except Exception as e:
            print(f"Error in getting suffix: {e}", file=sys.stderr)
            return text.strip(' ')

        try:
            prefix, response_text = response_text.split('X ')
            if len(prefix) > 0:
                print(f"Warning: prefix is not empty: {prefix}X {response_text} Y{suffix}", file=sys.stderr)
        except Exception as e:
            print(f"Error in getting prefix: {e}", file=sys.stderr)
            return text.strip(' ')

        response_text = 'X ' + response_text + ' Y'
        return response_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='dir')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--ordered', action='store_true')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')

    args = parser.parse_args()

    newsent_mapping = {}

    ordered_txt_r = '_ordered' if args.ordered else ''
    ordered_txt_w = '_ordered' if args.ordered else '_entord'
    # lemmatized_fp = open(f'./{args.subset}_files/with_entities/{args.split}{ordered_txt_w}_lemmatized.txt', 'w', encoding='utf8')

    inclusion_flags = []
    with open(f'./{args.subset}_files/with_entities/{args.split}{ordered_txt_r}.txt', 'r', encoding='utf8') as fp:
        for lidx, line in enumerate(fp):
            if lidx % 100 == 0:
                print(f'Processing line {lidx} of {args.split}...')
            hyp, prm, lbl, lang = line.strip().split('\t')
            hyp_subj, hyp_pred, hyp_obj = hyp.split(',')
            prm_subj, prm_pred, prm_obj = prm.split(',')
            hyp_subj = hyp_subj.strip(' ')
            hyp_pred = hyp_pred.strip(' ')
            hyp_obj = hyp_obj.strip(' ')
            prm_subj = prm_subj.strip(' ')
            prm_pred = prm_pred.strip(' ')
            prm_obj = prm_obj.strip(' ')
            hyp_sent = f'X {hyp_pred} Y.'
            prm_sent = f'X {prm_pred} Y.'
            if is_subsequence(hyp_pred.split(' '), prm_pred.split(' ')):
                inclusion_flags.append(True)
                # lemmatized_fp.write(f"None\tNone\t{lbl}\t{lang}\n")
                continue
            elif is_subsequence(prm_pred.split(' '), hyp_pred.split(' ')):
                inclusion_flags.append(True)
                is_subsequence(prm_pred.split(' '), hyp_pred.split(' '))
                # lemmatized_fp.write(f"None\tNone\t{lbl}\t{lang}\n")
                continue
            else:
                # if hyp_sent in newsent_mapping:
                #     hyp_paraphrase = newsent_mapping[hyp_sent]
                # else:
                #     hyp_paraphrase = get_paraphrase_from_gpt(hyp_sent, args.model_name)
                #     newsent_mapping[hyp_sent] = hyp_paraphrase
                # if prm_sent in newsent_mapping:
                #     prm_paraphrase = newsent_mapping[prm_sent]
                # else:
                #     prm_paraphrase = get_paraphrase_from_gpt(prm_sent, args.model_name)
                #     newsent_mapping[prm_sent] = prm_paraphrase
                # hyp_paraphrase = hyp_paraphrase.rstrip('.')
                # prm_paraphrase = prm_paraphrase.rstrip('.')
                # hyp_lemmatized_pred = hyp_paraphrase.lstrip('X').rstrip('Y')
                # prm_lemmatized_pred = prm_paraphrase.lstrip('X').rstrip('Y')
                # hyp_lemmatized_sent = f"{hyp_subj},{hyp_lemmatized_pred},{hyp_obj}"
                # prm_lemmatized_sent = f"{prm_subj},{prm_lemmatized_pred},{prm_obj}"
                # lemmatized_fp.write(f"{hyp_lemmatized_sent}\t{prm_lemmatized_sent}\t{lbl}\t{lang}\n")
                # if is_subsequence(hyp_paraphrase, prm_paraphrase):
                #     inclusion_flags.append(True)
                #     continue
                # elif is_subsequence(prm_paraphrase, hyp_paraphrase):
                #     inclusion_flags.append(True)
                #     continue
                # else:
                #     inclusion_flags.append(False)
                #     continue
                inclusion_flags.append(False)
                # lemmatized_fp.write(f"None\tNone\t{lbl}\t{lang}\n")
                continue
    print(inclusion_flags)

    print(f"Number of True: {inclusion_flags.count(True)}")

    with open(f'./{args.subset}_files/with_entities/{args.split}_inclusion_flags{ordered_txt_w}.json', 'w', encoding='utf8') as ofp:
        json.dump(inclusion_flags, ofp, ensure_ascii=False, indent=4)
    print(f"Done! Wrote to {args.split}_inclusion_flags{ordered_txt_w}.json")







