from matplotlib import pyplot as plt
import json
import argparse
import os
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from utils import load_general_entries, load_typed_general_entries, get_auc_norm_from_prec_recs, print_metrics, \
    get_freq_halves
from randprem_experiments import calc_score_from_predscr


def get_ranking_from_scores(scores):
    sorted_scores = sorted(scores, reverse=True)
    ranking = []
    for score in scores:
        ranking.append(sorted_scores.index(score))
    return ranking


def phi_coefficient(preds, labels):
    assert len(preds) == len(labels)
    assert all([isinstance(x, bool) for x in preds])
    assert all([isinstance(x, bool) for x in labels])
    num_true_true = 0
    num_true_false = 0
    num_false_true = 0
    num_false_false = 0
    for i in range(len(preds)):
        if preds[i] and labels[i]:
            num_true_true += 1
        elif preds[i] and not labels[i]:
            num_true_false += 1
        elif not preds[i] and labels[i]:
            num_false_true += 1
        elif not preds[i] and not labels[i]:
            num_false_false += 1
    num_true = num_true_true + num_true_false
    num_false = num_false_true + num_false_false
    num_pred_true = num_true_true + num_false_true
    num_pred_false = num_true_false + num_false_false
    phi = (num_true_true * num_false_false - num_true_false * num_false_true) / \
          (num_true * num_false * num_pred_true * num_pred_false) ** 0.5
    return phi


def discretize_freqscores(freqscores, margin):
    freq_discretes = []
    for x in freqscores:
        if x > margin / (1 + margin):
            freq_discretes.append('A')
        elif x < 1 / (1 + margin):
            freq_discretes.append('C')
        else:
            freq_discretes.append('B')
    return freq_discretes


def evaluate_subsets(preds: list, labels: list, crit: list, entries: list,
                     fscore_beta: float, name_of_prior: str, entries_out_path: str):
    c_true_preds = []
    c_unk_preds = []
    c_false_preds = []
    c_true_golds = []
    c_unk_golds = []
    c_false_golds = []

    c_consistent_preds = []
    c_neutral_preds = []
    c_adversarial_preds = []
    c_consistent_golds = []
    c_neutral_golds = []
    c_adversarial_golds = []

    c_consistent_entries = []
    c_neutral_entries = []
    c_adversarial_entries = []

    assert len(preds) == len(labels) == len(crit) == len(entries)
    for i, (p, l, c, e) in enumerate(zip(preds, labels, crit, entries)):
        # if i % 1 == 0:
        #     print(f"consistents: {len(c_consistent_preds)}; neutrals: {len(c_neutral_preds)}; adversarials: {len(c_adversarial_preds)}")
        assert isinstance(l, bool)
        if c == 'A':
            c_true_preds.append(p)
            c_true_golds.append(l)
            if l is True:
                c_consistent_preds.append(p)
                c_consistent_golds.append(l)
                c_consistent_entries.append(e)
            elif l is False:
                c_adversarial_preds.append(p)
                c_adversarial_golds.append(l)
                c_adversarial_entries.append(e)
            else:
                raise ValueError('Unknown label: {}'.format(l))
        elif c == 'B':
            c_unk_preds.append(p)
            c_unk_golds.append(l)
            c_neutral_preds.append(p)
            c_neutral_golds.append(l)
            c_neutral_entries.append(e)
        elif c == 'C':
            c_false_preds.append(p)
            c_false_golds.append(l)
            if l is True:
                c_adversarial_preds.append(p)
                c_adversarial_golds.append(l)
                c_adversarial_entries.append(e)
            elif l is False:
                c_consistent_preds.append(p)
                c_consistent_golds.append(l)
                c_consistent_entries.append(e)
            else:
                raise ValueError('Unknown label: {}'.format(l))
        else:
            raise ValueError('Unknown prior value: {}'.format(c))

    c_true_subset_posis = len([x for x in c_true_preds if x > 0.5])
    c_unk_subset_posis = len([x for x in c_unk_preds if x > 0.5])
    c_false_subset_posis = len([x for x in c_false_preds if x > 0.5])
    print(f"{name_of_prior} True: {c_true_subset_posis} / {len(c_true_preds)}: {c_true_subset_posis / len(c_true_preds):.4f}")
    print(f"{name_of_prior} Unknown: {c_unk_subset_posis} / {len(c_unk_preds)}: {c_unk_subset_posis / len(c_unk_preds):.4f}")
    print(f"{name_of_prior} False: {c_false_subset_posis} / {len(c_false_preds)}: {c_false_subset_posis / len(c_false_preds):.4f}")

    print_metrics(c_true_golds, c_true_preds, f'{name_of_prior} True', beta=fscore_beta)
    print_metrics(c_unk_golds, c_unk_preds, f'{name_of_prior} Unknown', beta=fscore_beta)
    print_metrics(c_false_golds, c_false_preds, f'{name_of_prior} False', beta=fscore_beta)
    print_metrics(c_consistent_golds, c_consistent_preds, f'{name_of_prior} Consistent', beta=fscore_beta)
    print_metrics(c_neutral_golds, c_neutral_preds, f'{name_of_prior} Neutral', beta=fscore_beta)
    print_metrics(c_adversarial_golds, c_adversarial_preds, f'{name_of_prior} Adversarial', beta=fscore_beta)
    print_metrics(labels, preds, f'{name_of_prior} All', beta=fscore_beta)

    with open(entries_out_path % 'consistent', 'w') as fp:
        for e in c_consistent_entries:
            oline = json.dumps(e, ensure_ascii=False)
            fp.write(oline + '\n')
    with open(entries_out_path % 'neutral', 'w') as fp:
        for e in c_neutral_entries:
            oline = json.dumps(e, ensure_ascii=False)
            fp.write(oline + '\n')
    with open(entries_out_path % 'adversarial', 'w') as fp:
        for e in c_adversarial_entries:
            oline = json.dumps(e, ensure_ascii=False)
            fp.write(oline + '\n')
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='levyholt')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='llama', choices=['llama', 'gpt'])
    # parser.add_argument('--use_plhr', type=str, default='type', help='Only relevant for search engine frequencies.')
    parser.add_argument('--in_context', type=str, default='cot')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ordered', action='store_true')
    parser.add_argument('--lemmatized', action='store_true')
    parser.add_argument('--diff_num_chunks', type=int, default=5)
    parser.add_argument('--start_year', type=str, default="2000")
    parser.add_argument('--prompt_idx', type=int, default=None)
    parser.add_argument('--num_templates', type=int, default=4)
    parser.add_argument('--freq_margin', type=float, default=5)
    parser.add_argument('--fscore_beta', type=float, default=0.5)
    parser.add_argument('--results_root', type=str, default='./results')

    parser.add_argument('--always_type_freq', action='store_true')
    parser.add_argument('--rte_ngram_aggr', type=str, default='max', choices=['max', 'avg', 'min', 'geo', 'tfidf'])
    args = parser.parse_args()

    ordstr = 'Ordered' if args.ordered else 'Entord'
    lemmstr = 'lemmatized' if args.lemmatized else 'raw'
    if args.model_name is None:
        if args.model_type == 'llama':
            args.model_name = 'llama-65b-hf'
        elif args.model_type == 'gpt':
            args.model_name = 'text-davinci-003'
        else:
            raise ValueError('Unknown model type: {}'.format(args.model_type))
    else:
        pass
    print(f"Model type: {args.model_type}; Model name: {args.model_name}")

    # Load the frequency baselines
    if args.dataset == 'levyholt':
        # The N-Gram frequency baseline is available only for LevyHolt and does not differ between different use_plhr settings
        with open(f'./levyholt_files/dir_files/with_original/{args.split}_{ordstr.lower()}_freqs_{lemmstr}.json', 'r', encoding='utf8') as fp:
            ngram_freq_baseline = json.load(fp)
        ngram_freq_res = [x['score'][args.start_year] for x in ngram_freq_baseline]
        ngram_freq_trinaries = discretize_freqscores(ngram_freq_res, args.freq_margin)

        # if args.use_plhr == 'type' or args.always_type_freq:
        #     plhr_str = 'type'
        # elif args.use_plhr == 'original':
        #     plhr_str = 'original'
        # else:
        #     raise ValueError('Unknown use_plhr value: {}'.format(args.use_plhr))
        # with open(f'./levyholt_files/dir_files/with_{plhr_str}/{args.split}_ordered_freqs_search_engine.json_noargs_lemmatized', 'r') as fp:
        #     search_engine_freq_baseline = json.load(fp)
        #     search_engine_freq_res = [x['score'] for x in search_engine_freq_baseline]
        #     search_engine_freq_trinaries = discretize_freqscores(search_engine_freq_res, args.freq_margin)
    elif args.dataset == 'rte':
        with open(f'./rte_files/rte_ngram_frequencies/{args.split}_type_freqs_ngram_{lemmstr}.json', 'r', encoding='utf8') as fp:
            ngram_freq_baseline = json.load(fp)
        ngram_freq_res = [x[f'score_tok{args.rte_ngram_aggr}'][args.start_year] for x in ngram_freq_baseline]
        ngram_freq_trinaries = discretize_freqscores(ngram_freq_res, args.freq_margin)

        # plhr_str = 'type' if args.always_type_freq else args.use_plhr
        # with open(f'./rte_files/rte_search_engine_frequencies/{args.split}_{plhr_str}_freqs_search_engine.json', 'r', encoding='utf8') as fp:
        #     search_engine_freq_baseline = json.load(fp)
        #     search_engine_freq_res = [x['score'] for x in search_engine_freq_baseline]
        #     search_engine_freq_trinaries = discretize_freqscores(search_engine_freq_res, args.freq_margin)
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    # Load the model results
    if args.model_type == 'llama':
        if args.dataset == 'levyholt':
            with open(f'./results/levyholt_results/llama_results/llama_{args.model_name}_res_dir_text_{args.split}_{args.use_plhr}_icl={args.in_context}_{args.num_templates}.json', 'r', encoding='utf8') as fp:
                all_results = json.load(fp)
                if len(all_results) == 1:
                    assert args.prompt_idx is None
                    model_res = all_results[0]
                else:
                    assert args.prompt_idx is not None
                    model_res = all_results[args.prompt_idx]
        elif args.dataset == 'rte':
            print(f"RTE results for {args.model_name} comes only with COT.")
            with open(f'./rte_results_llama/rte_{args.split}_{args.use_plhr}_results_{args.model_name}.json', 'r', encoding='utf8') as fp:
                all_results = json.load(fp)
                model_res = all_results['scores']
                model_res = [x for x in model_res if len(x) > 0]
                assert len(model_res) == 1
                model_res = model_res[0]
        else:
            raise ValueError(f'Invalid dataset: {args.dataset}')
    elif args.model_type == 'gpt':
        if args.dataset == 'levyholt':
            model_res = []
            with open(f'./results/levyholt_results/gpt_results/gpt3_{args.model_name}_res_dir_text_{args.split}_{args.use_plhr}_icl={args.in_context}_trinary_{args.num_templates}.json', 'r', encoding='utf8') as fp:
                for line in fp:
                    data = json.loads(line)
                    curr_preds = data['preds']
                    assert isinstance(curr_preds, list)
                    if len(curr_preds) == 1:
                        assert args.prompt_idx is None
                        model_res.append(curr_preds[0])
                    else:
                        assert args.prompt_idx is not None
                        model_res.append(curr_preds[args.prompt_idx])
        elif args.dataset == 'rte':
            model_res = []
            assert args.split == 'test', f"DEV set results for GPT-3 on RTE are not available yet."
            with open(f'./results/gpt3_{args.model_name}_rte_{args.split}_{args.use_plhr}_cot_res.json', 'r', encoding='utf8') as fp:
                for line in fp:
                    item = json.loads(line)
                    curr_preds = item['preds']
                    assert isinstance(curr_preds, list) and len(curr_preds) == 1
                    model_res.append(curr_preds[0])
        else:
            raise ValueError(f'Invalid dataset: {args.dataset}')
    else:
        raise ValueError(f'Invalid model name: {args.model_name}')

    # Load the input entries
    input_entries = []
    golds = []
    if args.dataset == 'levyholt':
        if args.use_plhr == 'type':
            data_path = f'./levyholt_files/dir_files/with_type/{args.split}%s.txt'
            prem_hyp_pairs = load_typed_general_entries(data_path)
        elif args.use_plhr in ['original', 'random', 'lowfreq', 'highfreq', 'randprem']:
            data_path = f'./levyholt_files/dir_files/with_original/{args.split}_ordered.txt'
            prem_hyp_pairs = load_general_entries(data_path)
        else:
            raise NotImplementedError(f"Unknown placeholder type: {args.use_plhr}")
        for prm, hyp, gold, _ in prem_hyp_pairs:
            if gold == 'True':
                gold = True
            elif gold == 'False':
                gold = False
            else:
                raise ValueError(f"Unknown gold value: {gold}")
            input_entries.append({'premise': prm, 'hypothesis': hyp, 'gold': gold})
            golds.append(gold)
    elif args.dataset == 'rte':
        data_path = f'./rte_files/rte_raw_files/{args.split}_{args.use_plhr}.txt'
        with open(data_path, 'r', encoding='utf8') as fp:
            for line in fp:
                if len(line) < 2:
                    continue
                hyp, prm, gold = line.rstrip('\n').split('\t')
                if gold == 'True':
                    gold = True
                elif gold == 'False':
                    gold = False
                else:
                    raise ValueError(f"Unknown gold value: {gold}")
                input_entries.append({'premise': prm, 'hypothesis': hyp, 'gold': gold})
                golds.append(gold)
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    if args.dataset == 'levyholt':
        ngram_subsets_out_fn = f'./levyholt_files/dir_files/with_original/{args.split}_{args.use_plhr}_ngram_%s_entries.json'
        # search_engine_subsets_out_fn = f'./levyholt_files/dir_files/with_original/{args.split}_{args.use_plhr}_search_engine_%s_entries.json'
    elif args.dataset == 'rte':
        ngram_subsets_out_fn = f'./rte_files/rte_raw_files/{args.split}_{args.use_plhr}_ngram_%s_entries.json'
        # search_engine_subsets_out_fn = f'./rte_files/rte_raw_files/{args.split}_{args.use_plhr}_search_engine_%s_entries.json'
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    # Evaluate the model results on the frequency-based subsets
    if ngram_freq_trinaries is not None:
        print_metrics(golds=golds, scores=ngram_freq_res, legend_str='ngram_freq_performance', beta=args.fscore_beta)
        evaluate_subsets(model_res, golds, ngram_freq_trinaries, entries=input_entries, fscore_beta=args.fscore_beta,
                         name_of_prior='ngram_freq', entries_out_path=ngram_subsets_out_fn)
    else:
        pass

    # print_metrics(golds=golds, scores=search_engine_freq_res, legend_str='search_engine_freq_performance', beta=args.fscore_beta)
    # evaluate_subsets(model_res, golds, search_engine_freq_trinaries, entries=input_entries, fscore_beta=args.fscore_beta,
                        # name_of_prior='search_engine_freq', entries_out_path=search_engine_subsets_out_fn)

    lmodel_ranking = get_ranking_from_scores(model_res)

    if ngram_freq_res is not None:
        ngram_freq_ranking = get_ranking_from_scores(ngram_freq_res)
        spearman_rho, spearman_p = spearmanr(lmodel_ranking, ngram_freq_ranking)
        print(f"N-Gram Frequency - Spearman's rho: {spearman_rho}, p-value: {spearman_p}")
    else:
        pass
    # search_engine_freq_ranking = get_ranking_from_scores(search_engine_freq_res)
    # spearman_rho, spearman_p = spearmanr(lmodel_ranking, search_engine_freq_ranking)
    # print(f"Search Engine Frequency - Spearman's rho: {spearman_rho}, p-value: {spearman_p}")


if __name__ == '__main__':
    main()