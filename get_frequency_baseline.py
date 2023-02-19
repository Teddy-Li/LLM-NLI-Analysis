import json
import sys
import argparse
import spacy
import requests
import math
import time
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, auc


def remove_modals_from_predicate(pred: str, nlp):
    doc = nlp(pred)
    new_pred_list = []
    for idx, word in enumerate(doc):
        if word.text in ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did', 'can', 'could', 'may',
                    'might', 'must', 'shall', 'should', 'will', 'would', 'be', 'been']:
            continue
        else:
            new_pred_list.append(word.lemma_)
    new_pred_for_curl = '+'.join(new_pred_list)
    return new_pred_for_curl


def calc_metrics_from_freqs(freqs, subset: str, split: str, lemmatize: str, start_years=(1800, 1850, 1900, 1950, 2000), end_year=2019):
    for sy in start_years:
        scores = [x['score'][str(sy)] for x in freqs]
        scores_binary = [True if x > 0.5 else False for x in scores]
        labels = []
        for x in freqs:
            if x['lbl'] == 'True':
                labels.append(True)
            elif x['lbl'] == 'False':
                labels.append(False)
            else:
                print(f"ERROR: {x['lbl']} is not a valid label.")
                raise ValueError
        precision, recall, f1, _ = precision_recall_fscore_support(labels, scores_binary, average='binary')
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        auc_score = auc(recalls, precisions)
        ap = average_precision_score(labels, scores)
        print(f"For date range {sy} to 2019: Precision: {precision}; Recall: {recall}; F1: {f1}; AUC: {auc_score}; AP: {ap}.")
        plt.plot(recalls, precisions, label=f"ngram freqs, {sy}-{end_year}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision Recall Curves")
    plt.legend()
    plt.draw()
    plt.savefig(f'freq_baseline_{subset}_{split}_{lemmatize}.png')
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='dir')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--only_do_scr', action='store_true')
    parser.add_argument('--lemmatize', action='store_true')
    parser.add_argument('--entord', action='store_true')

    args = parser.parse_args()
    lemma_str = 'lemmatized' if args.lemmatize else 'raw'
    ord_read_str = '' if args.entord else '_ordered'
    ord_write_str = '_entord' if args.entord else '_ordered'

    in_fn = f'./{args.subset}_files/with_entities/{args.split}{ord_read_str}.txt'
    out_fn = f'./{args.subset}_files/with_entities/{args.split}{ord_write_str}_freqs_{lemma_str}.json'

    earliest_year = 1800
    latest_year = 2019
    max_retry = 3

    if args.only_do_scr:
        with open(out_fn, 'r', encoding='utf8') as fp:
            entries = json.load(fp)
        calc_metrics_from_freqs(entries, subset=args.subset, split=args.split, lemmatize=lemma_str, end_year=latest_year)
        return

    nlp = spacy.load('en_core_web_sm')

    entries = []
    num_hyps_skipped = 0
    num_prms_skipped = 0

    with open(in_fn, 'r', encoding='utf8') as fp:
        for lidx, line in enumerate(fp):
            if lidx % 5 == 0:
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

            if args.lemmatize:
                hyp_pred_for_curl = remove_modals_from_predicate(hyp_pred, nlp)
                prm_pred_for_curl = remove_modals_from_predicate(prm_pred, nlp)
            else:
                hyp_pred_for_curl = hyp_pred.replace(' ', '+')
                prm_pred_for_curl = prm_pred.replace(' ', '+')

            hyp_pred_url = f'https://books.google.com/ngrams/json?content={hyp_pred_for_curl}&year_start={earliest_year}&year_end={latest_year}&corpus=26&smoothing=3'
            prm_pred_url = f'https://books.google.com/ngrams/json?content={prm_pred_for_curl}&year_start={earliest_year}&year_end={latest_year}&corpus=26&smoothing=3'
            for i in range(max_retry):
                try:
                    hyp_pred_response = requests.get(hyp_pred_url)
                    hyp_pred_json = hyp_pred_response.json()
                    hyp_pred_freq = hyp_pred_json[0]['timeseries']
                    break
                except Exception as e:
                    print(f'Request Error: {e}')
                    print(f"pred: {hyp_pred_for_curl}")
                    print(f"Response: {hyp_pred_response}")
                    time.sleep(1)
                    if i < max_retry - 1:
                        print(f"Retrying...")
                        continue
                    else:
                        print("Skipping...")
                        hyp_pred_freq = [0.000000000000001] * (latest_year - earliest_year + 1)
                        num_hyps_skipped += 1
                        break
            time.sleep(1)
            for i in range(max_retry):
                try:
                    prm_pred_response = requests.get(prm_pred_url)
                    prm_pred_json = prm_pred_response.json()
                    prm_pred_freq = prm_pred_json[0]['timeseries']
                    break
                except Exception as e:
                    print(f'Request Error: {e}')
                    print(f"pred: {prm_pred_for_curl}")
                    print(f"Response: {prm_pred_response}")
                    time.sleep(1)
                    if i < max_retry - 1:
                        print(f"Retrying...")
                        continue
                    else:
                        print("Skipping...")
                        prm_pred_freq = [0.000000000000001] * (latest_year - earliest_year + 1)
                        num_prms_skipped += 1
                        break
            time.sleep(1)
            starting_idxes = [0, 50, 100, 150, 200]
            hyp_pred_avg_freqs = {sp+earliest_year: sum(hyp_pred_freq[sp:]) / (len(hyp_pred_freq)-sp) for sp in starting_idxes}
            prm_pred_avg_freqs = {sp+earliest_year: sum(prm_pred_freq[sp:]) / (len(prm_pred_freq)-sp) for sp in starting_idxes}

            portions = {sp+earliest_year: hyp_pred_avg_freqs[sp+earliest_year] / prm_pred_avg_freqs[sp+earliest_year] for sp in starting_idxes}
            scores = {str(sp+earliest_year): portions[sp+earliest_year] / (1 + portions[sp+earliest_year]) for sp in starting_idxes}

            entry = {
                'hyp': hyp,
                'prm': prm,
                'lbl': lbl,
                'lang': lang,
                'hyp_pred_freq': hyp_pred_freq,
                'prm_pred_freq': prm_pred_freq,
                'score': scores,
            }
            entries.append(entry)

    with open(out_fn, 'w', encoding='utf8') as fp:
        json.dump(entries, fp, indent=4)

    print(f"Lemmatization: {args.lemmatize}")
    print(f"Skipped {num_hyps_skipped} hypotheses and {num_prms_skipped} premises.")
    calc_metrics_from_freqs(entries, subset=args.subset, split=args.split, lemmatize=lemma_str, end_year=latest_year)
    return


if __name__ == '__main__':
    main()