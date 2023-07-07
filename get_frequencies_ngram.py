import json
import os.path
import sys
import argparse
import spacy
import requests
import math
import time
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, auc


starting_idxes = [0, 50, 100, 150, 200]


def remove_modals_from_predicate(pred: str, nlp):
    doc = nlp(pred)
    new_pred_list = []
    inf_flag = False
    for idx, word in enumerate(doc):
        if word.text in ['can', 'could', 'may',
                    'might', 'must', 'shall', 'should', 'will', 'would']:
            continue
        elif word.text in ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did']:
            if not inf_flag:
                new_pred_list.append(word.lemma_ + '_INF')
                inf_flag = True
            else:
                new_pred_list.append(word.lemma_)
                print(f"WARNING: {pred} has multiple potential inflections.")
        elif word.pos_ == 'VERB':
            if not inf_flag:
                new_pred_list.append(word.lemma_ + '_INF')
                inf_flag = True
            else:
                new_pred_list.append(word.text)
        else:
            new_pred_list.append(word.text)
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
            elif isinstance(x['lbl'], bool):
                labels.append(x['lbl'])
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


def is_type_identifier(tok: str):
    tok_lst = tok.split('.')
    if len(tok_lst) != 2:
        return False
    else:
        if not tok_lst[1].isdigit():
            return False
        else:
            if any([x.isdigit() for x in tok_lst[0]]):
                return False
            else:
                return True


def process_rte_sentences_to_bag_of_words(sent, nlp, lemmatize=False):
    sent = sent.split(' ')
    sent[0] = sent[0].lower()
    sent = [x for x in sent if (len(x) > 0 and not is_type_identifier(x))]
    sent = ' '.join(sent)
    doc = nlp(sent)
    out_lst = []
    inf_flag = False
    for tok in doc:
        if tok.is_stop:
            continue
        elif tok.is_punct:
            continue
        elif tok.pos_ == 'VERB':
            if lemmatize and not inf_flag:
                out_lst.append(tok.lemma_ + '_INF')
                inf_flag = True
            else:
                out_lst.append(tok.text)
        else:
            out_lst.append(tok.text)

    return out_lst


def get_ngram_frequency(query, earliest_year, latest_year, max_retry=3):
    query_url = f'https://books.google.com/ngrams/json?content={query}&year_start={earliest_year}&year_end={latest_year}&corpus=en-2019&smoothing=3&case_insensitive=true'
    assert max_retry > 0
    success = False
    for i in range(max_retry):
        try:
            query_response = requests.get(query_url)
            query_json = query_response.json()
            query_freqs = [sum(x['timeseries'][t] for x in query_json) for t in
                             range(len(query_json[0]['timeseries']))]
            success = True
            break
        except Exception as e:
            if i == 0 and '<Response [200]>' not in query_response:
                print(f'Request Error: {e}; pred: {query}; Response: {query_response}')
            time.sleep(1)
            if i < max_retry - 1:
                continue
            else:
                print("Skipping...")
                query_freqs = [0.000000000000001] * (latest_year - earliest_year + 1)
                break
    if not success:
        return None
    else:
        time.sleep(1)
        query_avg_freqs = {sp + earliest_year: sum(query_freqs[sp:]) / (len(query_freqs) - sp) + 0.00000001 for sp in
                              starting_idxes}  # add 0.00000001 to avoid division by 0
        return query_avg_freqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='levyholt')
    parser.add_argument('--subset', type=str, default='dir')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--only_do_scr', action='store_true')
    parser.add_argument('--lemmatize', action='store_true')
    parser.add_argument('--use_plhr', type=str, default='type')
    parser.add_argument('--entord', action='store_true')

    args = parser.parse_args()
    lemma_str = 'lemmatized' if args.lemmatize else 'raw'
    ord_read_str = '' if args.entord else '_ordered'
    ord_write_str = '_entord' if args.entord else '_ordered'

    nlp = spacy.load('en_core_web_sm')

    if args.dataset == 'levyholt':
        in_fn = f'./levyholt_files/{args.subset}_files/randprem_files/{args.split}_randprem.txt'
        out_fn = f'./levyholt_files/{args.subset}_files/randprem_files/{args.split}_randprem_freqs_{lemma_str}.json'
        # in_fn = f'./levyholt_files/{args.subset}_files/with_original/{args.split}{ord_read_str}.txt'
        # out_fn = f'./levyholt_files/{args.subset}_files/with_original/{args.split}{ord_write_str}_freqs_{lemma_str}.json'
        # in_fn = f'./levyholt_files/{args.subset}_files/with_type/{args.split}.txt'
        # out_fn = f'./levyholt_files/{args.subset}_files/with_type/{args.split}_freqs_{lemma_str}.json'

        input_entries = []
        with open(in_fn, 'r', encoding='utf8') as fp:
            for lidx, line in enumerate(fp):
                line_lst = line.strip().split('\t')
                if len(line_lst) == 3:
                    hyp, prm, lbl = line_lst
                elif len(line_lst) == 4:
                    hyp, prm, lbl, lang = line_lst
                else:
                    raise ValueError
                hyp_subj, hyp_pred, hyp_obj = hyp.split(',')
                prm_subj, prm_pred, prm_obj = prm.split(',')
                # hyp_subj = hyp_subj.strip(' ')
                hyp_pred = hyp_pred.strip(' ')
                # hyp_obj = hyp_obj.strip(' ')
                # prm_subj = prm_subj.strip(' ')
                prm_pred = prm_pred.strip(' ')
                # prm_obj = prm_obj.strip(' ')
                if args.lemmatize:
                    hyp_pred_for_curl = remove_modals_from_predicate(hyp_pred, nlp)
                    prm_pred_for_curl = remove_modals_from_predicate(prm_pred, nlp)
                    if lidx % 25 == 0:
                        print(f"hyp_pred_for_curl: {hyp_pred_for_curl}; prm_pred_for_curl: {prm_pred_for_curl}")
                        print(f"hyp_pred: {hyp_pred}; prm_pred: {prm_pred}")
                else:
                    hyp_pred_for_curl = hyp_pred.replace(' ', '+')
                    prm_pred_for_curl = prm_pred.replace(' ', '+')
                curr_entry = {
                    'hyp_queries': [hyp_pred_for_curl],
                    'prm_queries': [prm_pred_for_curl],
                    'lbl': lbl
                }
                input_entries.append(curr_entry)
        print(f"Read in {len(input_entries)} entries.")
    elif args.dataset == 'rte':
        in_fn = f'./rte_files/rte_raw_files/{args.split}_{args.use_plhr}.txt'
        out_fn = f'./rte_files/rte_ngram_frequencies/{args.split}_{args.use_plhr}_freqs_ngram_{lemma_str}.json'

        input_entries = []
        with open(in_fn, 'r', encoding='utf8') as fp:
            for lidx, line in enumerate(fp):
                hyp, prm, lbl = line.strip().split('\t')
                # for RTE, calculate frequencies as bag of words
                hyp_lst = process_rte_sentences_to_bag_of_words(hyp, nlp, args.lemmatize)
                prm_lst = process_rte_sentences_to_bag_of_words(prm, nlp, args.lemmatize)
                curr_entry = {
                    'hyp_queries': hyp_lst,
                    'prm_queries': prm_lst,
                    'lbl': lbl
                }
                input_entries.append(curr_entry)
    else:
        raise ValueError(f"ERROR: {args.dataset} is not a valid dataset.")

    earliest_year = 1800
    latest_year = 2019
    max_retry = 3

    if args.only_do_scr:
        with open(out_fn, 'r', encoding='utf8') as fp:
            entries = json.load(fp)
        calc_metrics_from_freqs(entries, subset=args.subset, split=args.split, lemmatize=lemma_str, end_year=latest_year)
        return

    entries = []
    num_hyps_skipped = 0
    num_prms_skipped = 0
    start_time = time.time()
    out_fn_tmp = out_fn + '_tmp'
    out_fp_tmp = open(out_fn_tmp, 'w', encoding='utf8')

    for eidx, entry in enumerate(input_entries):
        if eidx % 25 == 0:
            durr = time.time() - start_time
            print(f'Processing entry {eidx} of {len(input_entries)}; {durr // 3600} hours, {(durr % 3600) // 60} minutes, {durr % 60} seconds elapsed;')
        hyp_queries = entry['hyp_queries']
        prm_queries = entry['prm_queries']
        lbl = entry['lbl']
        hyp_freqs = []
        prm_freqs = []
        for query in hyp_queries:
            curr_freqs= get_ngram_frequency(query, earliest_year, latest_year, max_retry)
            if curr_freqs is not None:
                hyp_freqs.append(curr_freqs)
        for query in prm_queries:
            curr_freqs = get_ngram_frequency(query, earliest_year, latest_year, max_retry)
            if curr_freqs is not None:
                prm_freqs.append(curr_freqs)
        # If no success, take the first backoff, otherwise, take only the successes
        if len(hyp_freqs) == 0:
            dummy_freqs = [0.000000000000001] * (latest_year - earliest_year + 1)
            dummy_avg_freqs = {sp + earliest_year: sum(dummy_freqs[sp:]) / (len(dummy_freqs) - sp) + 0.00000001 for sp
                               in starting_idxes}
            hyp_freqs = [dummy_avg_freqs]
        else:
            pass
        if len(prm_freqs) == 0:
            dummy_freqs = [0.000000000000001] * (latest_year - earliest_year + 1)
            dummy_avg_freqs = {sp + earliest_year: sum(dummy_freqs[sp:]) / (len(dummy_freqs) - sp) + 0.00000001 for sp
                               in starting_idxes}
            prm_freqs = [dummy_avg_freqs]
        else:
            pass

        if args.dataset == 'levyholt':
            assert len(hyp_freqs) == 1
            assert len(prm_freqs) == 1
            hyp_freqs = hyp_freqs[0]
            prm_freqs = prm_freqs[0]
        elif args.dataset == 'rte':
            hyp_tokavg_freqs = {
                sp+earliest_year: sum([hyp_freqs[i][sp+earliest_year] for i in range(len(hyp_freqs))]) / len(hyp_freqs) for sp in starting_idxes
            }
            prm_tokavg_freqs = {
                sp+earliest_year: sum([prm_freqs[i][sp+earliest_year] for i in range(len(prm_freqs))]) / len(prm_freqs) for sp in starting_idxes
            }
            hyp_tokmax_freqs = {
                sp+earliest_year: max([hyp_freqs[i][sp+earliest_year] for i in range(len(hyp_freqs))]) for sp in starting_idxes
            }
            prm_tokmax_freqs = {
                sp+earliest_year: max([prm_freqs[i][sp+earliest_year] for i in range(len(prm_freqs))]) for sp in starting_idxes
            }
            hyp_tokmin_freqs = {
                sp+earliest_year: min([hyp_freqs[i][sp+earliest_year] for i in range(len(hyp_freqs))]) for sp in starting_idxes
            }
            prm_tokmin_freqs = {
                sp+earliest_year: min([prm_freqs[i][sp+earliest_year] for i in range(len(prm_freqs))]) for sp in starting_idxes
            }

        curr_entry = {
            'hyp_freqs': hyp_freqs,
            'prm_freqs': prm_freqs,
            'lbl': lbl
        }
        if args.dataset == 'rte':
            curr_entry['hyp_tokavg_freqs'] = hyp_tokavg_freqs
            curr_entry['prm_tokavg_freqs'] = prm_tokavg_freqs
            curr_entry['hyp_tokmax_freqs'] = hyp_tokmax_freqs
            curr_entry['prm_tokmax_freqs'] = prm_tokmax_freqs
            curr_entry['hyp_tokmin_freqs'] = hyp_tokmin_freqs
            curr_entry['prm_tokmin_freqs'] = prm_tokmin_freqs
            portions_tokavg = {
                sp+earliest_year: hyp_tokavg_freqs[sp+earliest_year] / prm_tokavg_freqs[sp+earliest_year] for sp in starting_idxes
            }
            portions_tokmax = {
                sp+earliest_year: hyp_tokmax_freqs[sp+earliest_year] / prm_tokmax_freqs[sp+earliest_year] for sp in starting_idxes
            }
            portions_tokmin = {
                sp+earliest_year: hyp_tokmin_freqs[sp+earliest_year] / prm_tokmin_freqs[sp+earliest_year] for sp in starting_idxes
            }
            scores_tokavg = {str(sp+earliest_year): portions_tokavg[sp+earliest_year] / (1 + portions_tokavg[sp+earliest_year]) for sp in starting_idxes}
            scores_tokmax = {str(sp+earliest_year): portions_tokmax[sp+earliest_year] / (1 + portions_tokmax[sp+earliest_year]) for sp in starting_idxes}
            scores_tokmin = {str(sp+earliest_year): portions_tokmin[sp+earliest_year] / (1 + portions_tokmin[sp+earliest_year]) for sp in starting_idxes}
            curr_entry['score_tokavg'] = scores_tokavg
            curr_entry['score_tokmax'] = scores_tokmax
            curr_entry['score_tokmin'] = scores_tokmin
        elif args.dataset == 'levyholt':
            portions = {
                sp + earliest_year: hyp_freqs[sp + earliest_year] / prm_freqs[sp + earliest_year] for
                sp in starting_idxes}
            scores = {str(sp + earliest_year): portions[sp + earliest_year] / (1 + portions[sp + earliest_year]) for sp
                      in starting_idxes}
            curr_entry['score'] = scores
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")

        entries.append(curr_entry)
        out_line = json.dumps(curr_entry, ensure_ascii=False)
        out_fp_tmp.write(out_line + '\n')

    out_fp_tmp.close()
    out_dir = os.path.dirname(out_fn)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_fn, 'w', encoding='utf8') as fp:
        json.dump(entries, fp, indent=4)

    print(f"Lemmatization: {args.lemmatize}")
    print(f"Skipped {num_hyps_skipped} hypotheses and {num_prms_skipped} premises.")
    calc_metrics_from_freqs(entries, subset=args.subset, split=args.split, lemmatize=lemma_str, end_year=latest_year)
    return


if __name__ == '__main__':
    main()
    # x = remove_modals_from_predicate('ran for governor of', spacy.load('en_core_web_sm'))
    # print(x)