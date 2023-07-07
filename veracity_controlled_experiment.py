import json
import argparse
import os
from matplotlib import pyplot as plt
from utils import print_metrics, phi_coefficient, get_freq_halves
from randprem_experiments import calc_score_from_predscr
from sklearn.metrics import precision_recall_fscore_support


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['gpt', 'llama'], default='llama')
    parser.add_argument('--model_name', type=str, default='llama-65b-hf')
    parser.add_argument('--task', type=str, choices=['lh', 'rte'], default='lh')
    parser.add_argument('--use_plhr', type=str, default='original')
    parser.add_argument('--data_root', type=str, default='./levyholt_files/dir_files/')
    parser.add_argument('--beta', type=float, default=0.5)

    args = parser.parse_args()

    # load hyp-only results
    # TODO: Attention! Whatever the use_plhr configs, we always use the original arguments for hyp-only results (only that measures the attestation!)
    if args.task == 'lh':
        if args.model_type == 'gpt':
            honly_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_original_icl=lbl_trinary_1_hyponly.json'
        elif args.model_type == 'llama':
            honly_path = f'./results/levyholt_results/llama_results/llama_{args.model_name}_res_dir_text_test_original_icl=lbl_hypOnly.json'
        else:
            raise ValueError(f'Unexpected model type: {args.model_type}')
    elif args.task == 'rte':
        if args.model_type == 'gpt':
            honly_path = f'./results/gpt3_text-davinci-003_rte_hyponly_test_original_lbl_res.json'
        elif args.model_type == 'llama':
            honly_path = f'./llama_rte_hyponly_results/rte_hyponly_test_original_results_llama-65b-hf.json'
        else:
            raise ValueError(f'Unexpected model type: {args.model_type}')
    else:
        raise ValueError(f'Unexpected task: {args.task}')

    with open(honly_path, 'r', encoding='utf-8') as f:
        if args.model_type == 'gpt':
            honly_binaries = []
            honly_scores = []
            honly_gold_labels = []
            for line in f:
                item = json.loads(line)
                assert len(item['preds']) == 1
                if item['preds'][0] > 0.5:
                    honly_binaries.append(True)
                else:
                    honly_binaries.append(False)
                honly_scores.append(item['preds'][0])
                honly_gold_labels.append(item['gold'])
        elif args.model_type == 'llama':
            data = json.load(f)
            assert isinstance(data, dict)
            honly_predictions = data['predictions']
            honly_scores = data['scores']
            assert len(honly_predictions) == 1 and len(honly_scores) == 1
            honly_predictions = honly_predictions[0]
            honly_scores = honly_scores[0]
            honly_binaries = []
            for pred in honly_predictions:
                assert pred in ['A', 'B', 'C']
                if pred == 'A':
                    honly_binaries.append(True)
                else:
                    honly_binaries.append(False)
        else:
            raise ValueError(f'Unexpected model type: {args.model_type}')

    with open(f'./results/levyholt_results/polled_honly/llama_gpt_strict_honly_original_binaries.json', 'r', encoding='utf-8') as f:
        strict_honly_binaries = json.load(f)
    with open(f'./results/levyholt_results/polled_honly/llama_gpt_vote_honly_original_binaries.json', 'r', encoding='utf-8') as f:
        vote_honly_binaries = json.load(f)

    # load inference results
    if args.model_type == 'gpt':
        if args.task == 'lh':
            real_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_{args.use_plhr}_icl=cot_trinary_1.json'
            # real_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_{args.use_plhr}_icl=cot_trinary_instruct_1.json'
            # real_path = f'./results/levyholt_results/gpt_results/gpt3_gpt-4-0314_res_dir_text_test_original_icl=none_trinary_instruct_1.json'
        elif args.task == 'rte':
            real_path = f'./results/gpt3_text-davinci-003_rte_test_{args.use_plhr}_cot_res.json'
        else:
            raise ValueError(f'Unexpected task: {args.task}')
    elif args.model_type == 'llama':
        if args.task == 'lh':
            real_path = f'./results/levyholt_results/llama_results/llama_{args.model_name}_res_dir_text_test_{args.use_plhr}_icl=cot_1.json'
        elif args.task == 'rte':
            real_path = f'./rte_results_llama/rte_test_{args.use_plhr}_results_{args.model_name}.json'
        else:
            raise ValueError(f'Unexpected task: {args.task}')
    else:
        raise ValueError(f'Unexpected model type: {args.model_type}')
    with open(real_path, 'r', encoding='utf-8') as f:
        if args.model_type == 'gpt':
            inf_gold_labels = []
            inf_scores = []
            for line in f:
                item = json.loads(line)
                assert len(item['preds']) == 1
                inf_scores.append(item['preds'][0])
                inf_gold_labels.append(item['gold'])
        elif args.model_type == 'llama':
            data = json.load(f)
            if isinstance(data, dict):
                inf_scores = [x for x in data['scores'] if len(x) > 0]
                assert len(inf_scores) == 1
                inf_scores = inf_scores[0]
            elif isinstance(data, list):
                assert len(data) == 1
                inf_scores = data[0]
            else:
                raise ValueError(f'Unexpected data type: {type(data)}')
        else:
            raise ValueError(f'Unexpected model type: {args.model_type}')

    if args.model_type == 'gpt' and args.use_plhr == 'original' and args.task == 'lh':
        with open('./levyholt_files/dir_files/with_original/test_ord2entord.json', 'r', encoding='utf-8') as f:
            ord2entord_dict = json.load(f)
        assert len(ord2entord_dict) == len(inf_scores)
        new_inf_scores = []
        new_inf_gold_labels = []
        for i in range(len(inf_scores)):
            new_inf_scores.append(inf_scores[ord2entord_dict[str(i)]])
            new_inf_gold_labels.append(inf_gold_labels[ord2entord_dict[str(i)]])
        inf_scores = new_inf_scores
        inf_gold_labels = new_inf_gold_labels
        del new_inf_scores, new_inf_gold_labels

    # if args.task == 'lh':
    #     with open(
    #             f'./levyholt_files/dir_files/with_original/test_entord_freqs_lemmatized.json',
    #             'r', encoding='utf8') as fp:
    #         freq_baseline = json.load(fp)
    #     freq_halves_preds = get_freq_halves(freq_baseline)
    #     phi_hyponly_freqs = phi_coefficient(honly_binaries, freq_halves_preds)
    #     print(f"Phi coefficient between hyponly and frequent/infrequent: {phi_hyponly_freqs}")
    #
    # phi_honly_real = phi_coefficient(honly_binaries, [x > 0.5 for x in inf_scores])
    # print(f"Phi coefficient between hyponly and {args.model_type} inf scores: {phi_honly_real}")


    # load gold labels
    if args.task == 'lh':
        in_path = os.path.join(args.data_root, 'with_original', 'test_ordered.txt')
        gold_labels = []
        with open(in_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) < 2:
                    continue
                lst = line.split('\t')
                assert len(lst) == 4
                if lst[2].strip() == 'True':
                    gold_labels.append(True)
                elif lst[2].strip() == 'False':
                    gold_labels.append(False)
                else:
                    raise ValueError(f'Unexpected label: {lst[2]}')
    elif args.task == 'rte':
        gold_labels = []
        with open('./rte_files/rte_test/test_original.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data[0]:
                lbl = item['label']
                if lbl == 'True':
                    gold_labels.append(True)
                elif lbl == 'False':
                    gold_labels.append(False)
                else:
                    raise ValueError(f'Unexpected label: {lbl}')
    else:
        raise ValueError(f'Unexpected task: {args.task}')

    # phi_hyponly_golds = phi_coefficient(honly_binaries, gold_labels)
    # print(f"Phi coefficient between hyponly and gold labels: {phi_hyponly_golds}")
    # print(precision_recall_fscore_support(gold_labels, honly_binaries, average='binary'))
    if args.model_type == 'gpt':
        assert all([x == y for (x, y) in zip(gold_labels, honly_gold_labels)])
        assert all([x == y == z for (x, y, z) in zip(gold_labels, honly_gold_labels, inf_gold_labels)])
    assert len(gold_labels) == len(honly_binaries) == len(inf_scores)

    print_metrics(gold_labels, honly_scores, 'hyponly', beta=0.5)
    print_metrics(gold_labels, inf_scores, 'inference', beta=0.5)


    v_consistent_golds = []
    v_consistent_infs = []
    v_adversarial_golds = []
    v_adversarial_infs = []
    v_strict_consistent_golds = []
    v_strict_consistent_infs = []
    v_strict_adversarial_golds = []
    v_strict_adversarial_infs = []
    v_vote_consistent_golds = []
    v_vote_consistent_infs = []
    v_vote_adversarial_golds = []
    v_vote_adversarial_infs = []
    v_true_golds = []
    v_true_infs = []
    v_false_golds = []
    v_false_infs = []

    for i in range(len(gold_labels)):
        if gold_labels[i] == honly_binaries[i]:
            v_consistent_golds.append(gold_labels[i])
            v_consistent_infs.append(inf_scores[i])
        else:
            v_adversarial_golds.append(gold_labels[i])
            v_adversarial_infs.append(inf_scores[i])
        if honly_binaries[i] is True:
            v_true_golds.append(gold_labels[i])
            v_true_infs.append(inf_scores[i])
        else:
            v_false_golds.append(gold_labels[i])
            v_false_infs.append(inf_scores[i])
        if args.task == 'lh':
            assert strict_honly_binaries[i] is None or isinstance(strict_honly_binaries[i], bool)
            if strict_honly_binaries[i] is not None and strict_honly_binaries[i] == gold_labels[i]:
                v_strict_consistent_golds.append(gold_labels[i])
                v_strict_consistent_infs.append(inf_scores[i])
            elif strict_honly_binaries[i] is not None and strict_honly_binaries[i] != gold_labels[i]:
                v_strict_adversarial_golds.append(gold_labels[i])
                v_strict_adversarial_infs.append(inf_scores[i])
            else:
                assert strict_honly_binaries[i] is None
            if gold_labels[i] == vote_honly_binaries[i]:
                v_vote_consistent_golds.append(gold_labels[i])
                v_vote_consistent_infs.append(inf_scores[i])
            else:
                v_vote_adversarial_golds.append(gold_labels[i])
                v_vote_adversarial_infs.append(inf_scores[i])
        elif args.task == 'rte':
            pass
        else:
            raise ValueError(f'Unexpected task: {args.task}')

    print(f'Veracity consistent: {len(v_consistent_golds)}; Veracity adversarial: {len(v_adversarial_golds)}')
    print(f'Veracity consistent random precision: {sum(v_consistent_golds) / len(v_consistent_golds)}')
    print(f'Veracity adversarial random precision: {sum(v_adversarial_golds) / len(v_adversarial_golds)}')

    print_metrics(v_consistent_golds, v_consistent_infs, 'Veracity consistent', beta=args.beta)
    print_metrics(v_adversarial_golds, v_adversarial_infs, 'Veracity adversarial', beta=args.beta)
    plt.legend()
    plt.show()
    # plt.savefig(f'./results/levyholt_results/llama_results/llama_{args.model_name}_res_dir_text_test_{args.use_plhr}_icl=cot_1_attestation_subsets.png')

    if args.task == 'lh':
        print(f'Veracity strict consistent: {len(v_strict_consistent_golds)}; Veracity strict adversarial: {len(v_strict_adversarial_golds)}')
        print(f'Veracity strict consistent random precision: {sum(v_strict_consistent_golds) / len(v_strict_consistent_golds)}')
        print(f'Veracity strict adversarial random precision: {sum(v_strict_adversarial_golds) / len(v_strict_adversarial_golds)}')
        print_metrics(v_strict_consistent_golds, v_strict_consistent_infs, 'Veracity strict consistent', beta=args.beta)
        print_metrics(v_strict_adversarial_golds, v_strict_adversarial_infs, 'Veracity strict adversarial', beta=args.beta)
        plt.legend()
        plt.show()

        print(f'Veracity vote consistent: {len(v_vote_consistent_golds)}; Veracity vote adversarial: {len(v_vote_adversarial_golds)}')
        print(f'Veracity vote consistent random precision: {sum(v_vote_consistent_golds) / len(v_vote_consistent_golds)}')
        print(f'Veracity vote adversarial random precision: {sum(v_vote_adversarial_golds) / len(v_vote_adversarial_golds)}')
        print_metrics(v_vote_consistent_golds, v_vote_consistent_infs, 'Veracity vote consistent', beta=args.beta)
        print_metrics(v_vote_adversarial_golds, v_vote_adversarial_infs, 'Veracity vote adversarial', beta=args.beta)
        plt.legend()
        plt.show()
    elif args.task == 'rte':
        pass
    else:
        raise ValueError(f'Unknown task: {args.task}')

    print(f'Veracity true: {len(v_true_golds)}; Veracity false: {len(v_false_golds)}')
    print(f'Veracity true random precision: {sum(v_true_golds) / len(v_true_golds)}')
    print(f'Veracity false random precision: {sum(v_false_golds) / len(v_false_golds)}')

    print_metrics(v_true_golds, v_true_infs, 'Veracity true', beta=args.beta)
    print_metrics(v_false_golds, v_false_infs, 'Veracity false', beta=args.beta)
    plt.legend()
    plt.show()
    plt.savefig(f'./results/levyholt_results/llama_results/llama_{args.model_name}_res_dir_text_test_{args.use_plhr}_icl=cot_1_attestation_subsets.png')