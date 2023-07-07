import json
import argparse
from typing import List
import math
import sys

def print_option_stats(preds):
    bucket = {'A': 0, 'B': 0, 'C': 0}

    for pred in preds:
        assert pred in ['A', 'B', 'C']
        bucket[pred] += 1

    print(f"Option Stats: {bucket}; Total: {sum(bucket.values())}; ratio of positives: {bucket['A'] / sum(bucket.values()) * 100:.2f}%")


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


def calc_score_from_predscr(predscrs: List[float]):
    assert len(predscrs) == 3
    assert all([s < 0 for s in predscrs])
    if predscrs[0] >= predscrs[1] and predscrs[0] >= predscrs[2]:
        pred = 'A'
        raw_scr = predscrs[0]
    elif predscrs[1] >= predscrs[2]:
        pred = 'B'
        raw_scr = predscrs[1]
    else:
        pred = 'C'
        raw_scr = predscrs[2]

    exp_scr = math.exp(raw_scr)

    if pred == 'A':
        scr = 0.5 + exp_scr / 2
    else:
        assert pred in ['B', 'C']
        scr = 0.5 - exp_scr / 2
    assert 0 <= scr <= 1
    return pred, scr


def print_conditional_probs(rp_preds, ref_preds, ref_str: str):
    rp_hypTrue = {'A': 0, 'B': 0, 'C': 0, 'Total': 0}
    rp_hypUnknown = {'A': 0, 'B': 0, 'C': 0, 'Total': 0}
    rp_hypFalse = {'A': 0, 'B': 0, 'C': 0, 'Total': 0}
    rp_hypBC = {'A': 0, 'B': 0, 'C': 0, 'Total': 0}

    for rp_pred, hyp_pred in zip(rp_preds, ref_preds):
        assert rp_pred in ['A', 'B', 'C']
        assert hyp_pred in ['A', 'B', 'C']
        if hyp_pred == 'A':
            rp_hypTrue[rp_pred] += 1
            rp_hypTrue['Total'] += 1
        elif hyp_pred == 'B':
            rp_hypUnknown[rp_pred] += 1
            rp_hypUnknown['Total'] += 1
            rp_hypBC[rp_pred] += 1
            rp_hypBC['Total'] += 1
        else:
            rp_hypFalse[rp_pred] += 1
            rp_hypFalse['Total'] += 1
            rp_hypBC[rp_pred] += 1
            rp_hypBC['Total'] += 1

    print(f"Conditional Probabilities:")
    print(f"rp_{ref_str}-Positive: 'A': {rp_hypTrue['A']} ({rp_hypTrue['A'] / (rp_hypTrue['Total']+0.000000001) * 100:.2f}%),"
          f" 'B': {rp_hypTrue['B']} ({rp_hypTrue['B'] / (rp_hypTrue['Total']+0.000000001) * 100:.2f}%),"
          f" 'C': {rp_hypTrue['C']} ({rp_hypTrue['C'] / (rp_hypTrue['Total']+0.000000001) * 100:.2f}%), Total: {rp_hypTrue['Total']}")

    print(f"rp_{ref_str}-Neutral: 'A': {rp_hypUnknown['A']} ({rp_hypUnknown['A'] / (rp_hypUnknown['Total']+0.000000001) * 100:.2f}%),"
          f" 'B': {rp_hypUnknown['B']} ({rp_hypUnknown['B'] / (rp_hypUnknown['Total']+0.000000001) * 100:.2f}%),"
          f" 'C': {rp_hypUnknown['C']} ({rp_hypUnknown['C'] / (rp_hypUnknown['Total']+0.000000001) * 100:.2f}%), Total: {rp_hypUnknown['Total']}")

    print(f"rp_{ref_str}-Negative: 'A': {rp_hypFalse['A']} ({rp_hypFalse['A'] / (rp_hypFalse['Total']+0.000000001) * 100:.2f}%),"
          f" 'B': {rp_hypFalse['B']} ({rp_hypFalse['B'] / (rp_hypFalse['Total']+0.000000001) * 100:.2f}%),"
          f" 'C': {rp_hypFalse['C']} ({rp_hypFalse['C'] / (rp_hypFalse['Total']+0.000000001) * 100:.2f}%), Total: {rp_hypFalse['Total']}")

    print(f"rp_{ref_str}-BC: 'A': {rp_hypBC['A']} ({rp_hypBC['A'] / (rp_hypBC['Total']+0.000000001) * 100:.2f}%),"
            f" 'B': {rp_hypBC['B']} ({rp_hypBC['B'] / (rp_hypBC['Total']+0.000000001) * 100:.2f}%),"
            f" 'C': {rp_hypBC['C']} ({rp_hypBC['C'] / (rp_hypBC['Total']+0.000000001) * 100:.2f}%), Total: {rp_hypBC['Total']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['gpt', 'llama'], default='llama')
    parser.add_argument('--model_name', type=str, default='llama-65b-hf')
    parser.add_argument('--use_plhr', type=str, default='original')
    parser.add_argument('--task', type=str, default='lh')
    parser.add_argument('--freq_margin', type=float, default=5.0)
    parser.add_argument('--lemmatized', action='store_true')
    parser.add_argument('--rte_ngram_aggr', type=str, default='min', choices=['max', 'min', 'avg'])

    args = parser.parse_args()

    # Load Hyp-only results
    if args.model_type == 'gpt':
        if args.task == 'lh':
            honly_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_{args.use_plhr}_icl=lbl_trinary_1_hyponly.json'
        elif args.task == 'rte':
            honly_path = f'./results/gpt3_text-davinci-003_rte_hyponly_test_{args.use_plhr}_lbl_res.json'
        else:
            raise ValueError(f"Invalid task: {args.task}")
    elif args.model_type == 'llama':
        if args.task == 'lh':
            honly_path = f'./results/levyholt_results/llama_results/llama_{args.model_name}_res_dir_text_test_{args.use_plhr}_icl=lbl_hypOnly.json'
        elif args.task == 'rte':
            honly_path = f'./llama_rte_hyponly_results/rte_hyponly_test_{args.use_plhr}_results_{args.model_name}.json'
        else:
            raise ValueError(f"Invalid task: {args.task}")
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    if honly_path is not None:
        with open(honly_path, 'r', encoding='utf-8') as f:
            if args.model_type == 'gpt':
                hyponly_predictions = []
                for line in f:
                    item = json.loads(line)
                    if 'preds_tokenized' in item:
                        assert len(item['preds_tokenized']) == 1
                        hyponly_predictions.append(item['preds_tokenized'][0])
                    else:
                        assert len(item['preds']) == 1
                        if item['preds'][0] > 0.5:
                            hyponly_predictions.append('A')
                        else:
                            hyponly_predictions.append('B')
            elif args.model_type == 'llama':
                data = json.load(f)
                hyponly_predictions = data['predictions']
                assert len(hyponly_predictions) == 1
                hyponly_predictions = hyponly_predictions[0]
            else:
                raise ValueError(f"Invalid model type: {args.model_type}")
            print(f"Hyp-only Option Stats:")
            print_option_stats(hyponly_predictions)

    # Load Real Inference results
    if args.model_type == 'gpt':
        if args.task == 'lh':
            real_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_{args.use_plhr}_icl=cot_trinary_1.json'
            # real_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_original_icl=cot_trinary_instruct_1.json'
        elif args.task == 'rte':
            real_path = f'./results/gpt3_text-davinci-003_rte_test_{args.use_plhr}_cot_res.json'
        else:
            raise ValueError(f"Invalid task: {args.task}")
    elif args.model_type == 'llama':
        # real_path = f'./llama_results/llama_{args.model_name}_res_dir_text_test_{args.use_plhr}_icl=cot_1.json'
        if args.task == 'lh':
            real_path = f'./llama_results/llama_{args.model_name}_res_dir_text_test_{args.use_plhr}_icl=cot_1.json'
        elif args.task == 'rte':
            real_path = f'./rte_results_llama/rte_test_{args.use_plhr}_results_{args.model_name}.json'
        else:
            raise ValueError(f"Invalid task: {args.task}")
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    with open(real_path, 'r', encoding='utf-8') as f:
        if args.model_type == 'gpt':
            real_predictions = []
            for line in f:
                item = json.loads(line)
                if 'preds_tokenized' in item:
                    assert len(item['preds_tokenized']) == 1
                    real_predictions.append(item['preds_tokenized'][0])
                else:
                    assert len(item['preds']) == 1
                    if item['preds'][0] > 0.5:
                        real_predictions.append('A')
                    else:
                        real_predictions.append('B')
        elif args.model_type == 'llama':
            data = json.load(f)
            if isinstance(data, list):
                data = [x for x in data if len(x) > 0]
                assert len(data) == 1
                data = data[0]
                real_predictions = []
                for scr in data:
                    if scr > 0.5:
                        real_predictions.append('A')
                    else:
                        real_predictions.append('B')
            elif isinstance(data, dict):
                real_predictions = data['predictions']
                real_predictions = [x for x in real_predictions if len(x) > 0]
                assert len(real_predictions) == 1
                real_predictions = real_predictions[0]
            else:
                raise ValueError(f"Invalid data type: {type(data)}")
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")
        print(f"Real Dataset Option Stats:")
        print_option_stats(real_predictions)
        print(f"Real Dataset Conditional Probabilities on Hyp-Only:")
        print_conditional_probs(real_predictions, hyponly_predictions, ref_str='hyponly')

    if args.model_type in ['gpt']:
        pass
    elif args.model_type == 'llama':
        if args.task == 'lh':
            for replace_condition in ['low', 'draw', 'high', 'none']:
                print(f"Evaluating RandPrem-HypOnly correlations for {replace_condition} replace condition:")
                with open(f'./results/rp_freq_{args.model_name}_{args.use_plhr}/llama_{args.model_name}_res_dir_text_test_rp_{args.use_plhr}_{replace_condition}_icl=cot_1.json',
                          'r', encoding='utf-8') as f:
                    data = json.load(f)
                    rp_predictions = data['predictions']
                    assert len(rp_predictions) == 1
                    rp_predictions = rp_predictions[0]
                    print(f"RandPrem Option Stats:")
                    print_option_stats(rp_predictions)
                    print_conditional_probs(rp_predictions, hyponly_predictions, ref_str=f'{replace_condition}_hyponly')
        elif args.task == 'rte':
            print(f"Skipping RandPrem-HypOnly correlations for RTE", file=sys.stderr)
        else:
            raise ValueError(f"Invalid task: {args.task}")
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    if args.task == 'lh':
        ngram_freq_predictions = []
        search_engine_freq_predictions = []
        with open(f"./levyholt_files/dir_files/randprem_files/test_freqs_lemmatized.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                hyp_freqs = entry['hyp_pred_freq']
                prm_freqs = entry['prm_pred_freq']
                assert len(hyp_freqs) == len(prm_freqs) == 220
                hyp_avg_freq = sum(hyp_freqs[150:]) / len(hyp_freqs[150:])
                prm_avg_freq = sum(prm_freqs[150:]) / len(prm_freqs[150:])
                if hyp_avg_freq / prm_avg_freq > args.freq_margin:
                    ngram_freq_predictions.append('A')
                elif hyp_avg_freq / prm_avg_freq < 1 / args.freq_margin:
                    ngram_freq_predictions.append('C')
                else:
                    ngram_freq_predictions.append('B')
            print(f"A: {ngram_freq_predictions.count('A')}, B: {ngram_freq_predictions.count('B')}, C: {ngram_freq_predictions.count('C')}, Total: {len(ngram_freq_predictions)}")

        with open(f'./levyholt_files/dir_files/with_type/test_ordered_freqs_search_engine.json_noargs_lemmatized', 'r', encoding='utf8') as fp:
            data = json.load(fp)
            for entry in data:
                hyp_freq = entry['num_hyp_res']
                prm_freq = entry['num_prm_res']
                if hyp_freq / prm_freq > args.freq_margin:
                    search_engine_freq_predictions.append('A')
                elif hyp_freq / prm_freq < 1 / args.freq_margin:
                    search_engine_freq_predictions.append('C')
                else:
                    search_engine_freq_predictions.append('B')
    elif args.task == 'rte':
        with open(f'./rte_files/rte_ngram_frequencies/{args.split}_type_freqs_ngram_lemmatized.json', 'r', encoding='utf8') as fp:
            ngram_freq_baseline = json.load(fp)
        ngram_freq_res = [x[f'score_tok{args.rte_ngram_aggr}'][args.start_year] for x in ngram_freq_baseline]
        ngram_freq_predictions = discretize_freqscores(ngram_freq_res, args.freq_margin)

        search_engine_freq_predictions = []
        with open(f'./rte_files/rte_se_frequencies/test_type_freqs_search_engine.json', 'r', encoding='utf8') as fp:
            data = json.load(fp)
            for entry in data:
                hyp_freq = entry['num_hyp_res']
                prm_freq = entry['num_prm_res']
                if hyp_freq / (prm_freq+0.00001) > args.freq_margin:
                    search_engine_freq_predictions.append('A')
                elif hyp_freq / (prm_freq+0.00001) < args.freq_margin:
                    search_engine_freq_predictions.append('C')
                else:
                    search_engine_freq_predictions.append('B')
    else:
        raise ValueError(f"Invalid task: {args.task}")

    print(f"Loading random premise predictions:")
    if args.model_type == 'gpt':
        if args.task == 'lh':
            randprem_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_randprem-{args.use_plhr}_icl=cot_trinary_1.json'
            # randprem_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_randprem-{args.use_plhr}_icl=cot_trinary_instruct_1.json'
        elif args.task == 'rte':
            randprem_path = None
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid task: {args.task}")
    elif args.model_type == 'llama':
        if args.task == 'lh':
            randprem_path = f'./randprem-results/llama_{args.model_name}_res_dir_text_test_randprem_{args.use_plhr}_icl=cot_1.json'
        elif args.task == 'rte':
            randprem_path = None
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid task: {args.task}")
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    with open(randprem_path, 'r', encoding='utf8') as fp:
        if args.model_type == 'gpt':
            rp_predictions = []
            for line in fp:
                item = json.loads(line)
                if 'preds_tokenized' in item:
                    assert len(item['preds_tokenized']) == 1
                    rp_predictions.append(item['preds_tokenized'][0])
                else:
                    assert len(item['preds']) == 1
                    if item['preds'][0] > 0.5:
                        rp_predictions.append('A')
                    else:
                        rp_predictions.append('B')
        elif args.model_type == 'llama':
            data = json.load(fp)
            if isinstance(data, list):
                data = data[0]
                rp_predictions = []
                for scr in data:
                    if scr > 0.5:
                        rp_predictions.append('A')
                    else:
                        rp_predictions.append('B')
            elif isinstance(data, dict):
                rp_predictions = data['predictions']
                assert len(rp_predictions) == 1
                rp_predictions = rp_predictions[0]
            else:
                raise ValueError(f"Invalid data type: {type(data)}")
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")
        print(f"RandPrem Option Stats:")
        print_option_stats(rp_predictions)
        print(f"Correlation with hyp-only:")
        print_conditional_probs(rp_predictions, hyponly_predictions, ref_str='hyponly')
        if args.task == 'lh':
            print(f"Correlation with N-Gram frequency:")
            print_conditional_probs(rp_predictions, ngram_freq_predictions, ref_str='ngram-freq')
            print(f"Correlation with search_engine frequency:")
            print_conditional_probs(rp_predictions, search_engine_freq_predictions, ref_str='search_engine-freq')
            print(f"Correlation with frequency when hyp-only=Unknown:")
            rp_predictions_hypNeutral, ngram_freq_predictions_hypNeutral, search_engine_freq_predictions_hypNeutral = [], [], []
            for i in range(len(rp_predictions)):
                if hyponly_predictions[i] == 'B':
                    rp_predictions_hypNeutral.append(rp_predictions[i])
                    ngram_freq_predictions_hypNeutral.append(ngram_freq_predictions[i])
                    search_engine_freq_predictions_hypNeutral.append(search_engine_freq_predictions[i])
            print_conditional_probs(rp_predictions_hypNeutral, ngram_freq_predictions_hypNeutral, ref_str='ngram-freq-hypNeutral')
            print_conditional_probs(rp_predictions_hypNeutral, search_engine_freq_predictions_hypNeutral, ref_str='search_engine-freq-hypNeutral')
        elif args.task == 'rte':
            print(f"Correlation with search_engine frequency:")
            print_conditional_probs(rp_predictions, search_engine_freq_predictions, ref_str='search_engine-freq')
            print(f"Correlation with frequency when hyp-only=Unknown:")
            rp_predictions_hypNeutral, search_engine_freq_predictions_hypNeutral = [], []
            for i in range(len(rp_predictions)):
                if hyponly_predictions[i] == 'B':
                    rp_predictions_hypNeutral.append(rp_predictions[i])
                    search_engine_freq_predictions_hypNeutral.append(search_engine_freq_predictions[i])
            print_conditional_probs(rp_predictions_hypNeutral, search_engine_freq_predictions_hypNeutral, ref_str='search_engine-freq-hypNeutral')
        else:
            raise ValueError(f"Invalid task: {args.task}")

