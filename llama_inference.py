import json
import time
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, GenerationConfig
import torch
import argparse
import os
import sys
import random
import math
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, auc
import matplotlib.pyplot as plt

from utils import load_typed_general_entries, load_general_entries, get_gpt_template, find_best_f_beta_from_curve, \
    print_metrics


sent_template_to_test = [
    {'s': "{prm}, which means that {hyp}.", 'do_neg': False},
    {'s': "If {prm}, then {hyp}.", 'do_neg': False},
    {'s': "{prm}, so {hyp}.", 'do_neg': False},
    {'s': "{prm} entails {hyp}.", 'do_neg': False},
]

knowledge_templates_to_test = {
    'h': {'s': "{hyp}.", 'do_neg': False},
    'p': {'s': "{prm}.", 'do_neg': False},
}

option_indices = {'A': 319, 'B': 350, 'C': 315, 'a': 263, 'b': 289, 'c': 274, 'Entailment': 4284, 'entailment': 875,
                  'Neutral': 2448, 'neutral': 21104, 'Contradiction': 1281, 'contradiction': 23949,
                  'true': 1565, 'True': 5852, 'unknown': 9815, 'Unknown': 853, 'false': 2089, 'False': 7700}  # for multi-token words like ``entailment'', we take only the first token for scores.


def option_matcher(output: str, char: str) -> bool:

    if output == char:
        return True
    elif output == char.lower():
        return True
    elif output.startswith(char + ')'):
        return True
    elif output.startswith(char + ' '):
        return True
    elif output.startswith(char + '.'):
        return True
    elif output.startswith(char + '-'):
        return True
    else:
        return False


def process_paths(data_root, results_root, subset, split, use_plhr, in_context, instruct_str, num_templates,
                  single_statement, machine, model_name):
    if use_plhr == 'type':
        data_path = os.path.join(data_root, f"{subset}_files", f"with_type", f"{split}%s.txt")
    elif use_plhr == 'original':
        if machine == 'local':
            data_path = os.path.join(data_root, f"{subset}_files", f"with_original", f"{split}_ordered.txt")
        elif machine == 'cloud':
            data_path = os.path.join(data_root, f"{subset}_files", f"with_original",
                                     f"{split}_ordered.txt")
        else:
            raise ValueError(f"Unknown machine name!")
    elif use_plhr == 'random':
        data_path = os.path.join(data_root, f"{subset}_files", f"swapped_entities", f"{split}_uniform.txt")
    elif use_plhr == 'lowfreq':
        data_path = os.path.join(data_root, f"{subset}_files", f"swapped_entities", f"{split}_bottom0.05.txt")
    elif use_plhr == 'highfreq':
        data_path = os.path.join(data_root, f"{subset}_files", f"swapped_entities", f"{split}_top0.05.txt")
    elif use_plhr == 'randprem_orig':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files", f"{split}_randprem.txt")
    elif use_plhr == 'randprem_type':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files", f"{split}_randprem%s.txt")
    elif use_plhr == 'randhyp_orig':
        data_path = os.path.join(data_root, f"{subset}_files", f"swapped_hypotheses", f"_{split}_randhyp.txt")
    elif use_plhr == 'randhyp_type':
        raise NotImplementedError
    elif use_plhr == 'rp_original_low':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_original", f"{split}_ordered_randprem_low.txt")
    elif use_plhr == 'rp_original_high':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_original", f"{split}_ordered_randprem_high.txt")
    elif use_plhr == 'rp_original_draw':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_original", f"{split}_ordered_randprem_draw.txt")
    elif use_plhr == 'rp_original_none':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_original", f"{split}_ordered_randprem_none.txt")
    elif use_plhr == 'rp_type_low':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_type", f"{split}_randprem_low%s.txt")
    elif use_plhr == 'rp_type_high':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_type", f"{split}_randprem_high%s.txt")
    elif use_plhr == 'rp_type_draw':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_type", f"{split}_randprem_draw%s.txt")
    elif use_plhr == 'rp_type_none':
        data_path = os.path.join(data_root, f"{subset}_files", f"randprem_files_freqband_type", f"{split}_randprem_none%s.txt")
    elif use_plhr == 'ant':
        assert subset == 'ant'
        data_path = os.path.join(data_root, f"{subset}_files", f"{split}.txt")
    else:
        raise NotImplementedError(f"Unknown placeholder type: {use_plhr}")
    results_path = os.path.join(results_root,
                                f'llama_{model_name}_res_{subset}_text_{split}_{use_plhr}_icl={in_context}{instruct_str}_{num_templates}.json')
    if single_statement == 'h':
        results_path = results_path.replace('.json', '_hypOnly.json')
    else:
        assert single_statement is None
        pass

    return data_path, results_path


def judgement(net_scores, net_charlist, tokenizer, is_single_statement: bool):
    curr_pred = None
    curr_scr = None
    output_irregular_flag = False
    output_tokens = tokenizer.decode(net_charlist).lower()
    for i in range(7):
        if net_scores[i] < 0:
            print(f"Warning: net score is negative: {net_scores[i]}", file=sys.stderr)
            net_scr = 0
        else:
            net_scr = net_scores[i]
        sigmoid_scr = net_scr / (1 + net_scr)

        if net_charlist[i] in [option_indices['A'], option_indices['a'], option_indices['Entailment'],
                              option_indices['entailment']]:
            curr_pred = 'A'
            curr_scr = 0.5 + 0.5 * sigmoid_scr
            break
        elif net_charlist[i] in [option_indices['true'], option_indices['True']] and is_single_statement is True:
            curr_pred = 'A'
            curr_scr = 0.5 + 0.5 * sigmoid_scr
            break
        elif net_charlist[i] in [option_indices['B'], option_indices['b'], option_indices['Neutral'],
                                option_indices['neutral']]:
            curr_pred = 'B'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
            break
        elif net_charlist[i] in [option_indices['unknown'], option_indices['Unknown']] and is_single_statement is True:
            curr_pred = 'B'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
            break
        elif net_charlist[i] in [option_indices['C'], option_indices['c'], option_indices['Contradiction'],
                                option_indices['contradiction']]:
            curr_pred = 'C'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
            break
        elif net_charlist[i] in [option_indices['false'], option_indices['False']] and is_single_statement is True:
            curr_pred = 'C'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
            break
        else:
            pass

    if curr_pred is not None or curr_scr is not None:
        assert curr_pred is not None and curr_scr is not None
    else:
        if net_scores[0] < 0:
            print(f"Warning: net score is negative: {net_scores[0]}", file=sys.stderr)
            net_scr = 0
        else:
            net_scr = net_scores[0]
        sigmoid_scr = net_scr / (1 + net_scr)
        if output_tokens.startswith('entail') and len(net_scores) > 0 and not is_single_statement:
            curr_pred = 'A'
            curr_scr = 0.5 + 0.5 * sigmoid_scr
        elif output_tokens.startswith('true') and len(net_scores) > 0 and is_single_statement:
            curr_pred = 'A'
            curr_scr = 0.5 + 0.5 * sigmoid_scr
        elif output_tokens.startswith('neutral') and len(net_scores) > 0 and not is_single_statement:
            curr_pred = 'B'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
        elif output_tokens.startswith('unknown') and len(net_scores) > 0 and is_single_statement:
            curr_pred = 'B'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
        elif output_tokens.startswith('contradiction') and len(net_scores) > 0 and not is_single_statement:
            curr_pred = 'C'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
        elif output_tokens.startswith('false') and len(net_scores) > 0 and is_single_statement:
            curr_pred = 'C'
            curr_scr = 0.5 - 0.5 * sigmoid_scr
        else:
            print(f"Irregular output: {output_tokens}; {net_charlist}")
            output_irregular_flag = True
            curr_pred = 'B'
            curr_scr = 0.0

    if not -0.00001 <= curr_scr <= 1.00001:
        print(f"Error!!!!!!!!!!!!!!!!!!! CURR_SCR OUT OF RANGE: {curr_scr}", file=sys.stderr)

    return curr_pred, curr_scr, output_tokens, output_irregular_flag


def run_over_levy_holt(data_path, results_path, model, tokenizer, gen_configs, use_plhr: str,
                       in_context: str, has_instruction: bool, beta: float, tplt_id,
                       single_statement: str, batch_size: int):

    print_template_flag = True
    output_irregular_cnt = 0
    print(f"Instantiating Templates...")
    embedded_prompts = collect_templates(data_path, embedded_path=None, use_plhr=use_plhr, in_context=in_context,
                                         has_instruction=has_instruction, tplt_id=tplt_id, single_statement=single_statement)
    start_time = time.time()
    predictions = [[] for x in range(len(embedded_prompts))]
    scores = [[] for x in range(len(embedded_prompts))]
    out_tokens = [[] for x in range(len(embedded_prompts))]
    golds = [[] for x in range(len(embedded_prompts))]

    for tidx, curr_tplt_instrs in enumerate(embedded_prompts):
        print(f"Generating predictions for template {tidx}...")
        for bstart_eidx in range(0, len(curr_tplt_instrs), batch_size):
            if bstart_eidx // batch_size % 5 == 0:
                durr = time.time() - start_time
                print(f"Batch {bstart_eidx // batch_size} / {len(curr_tplt_instrs) // batch_size + 1}; duration {durr // 3600}h {durr % 3600 // 60}m {durr % 60}s;")
            bstop_eidx = min(bstart_eidx + batch_size, len(curr_tplt_instrs))
            curr_batch = curr_tplt_instrs[bstart_eidx:bstop_eidx]
            curr_instrs = [x['in'] for x in curr_batch]
            curr_labels = [x['label'] for x in curr_batch]

            curr_inputs = tokenizer(curr_instrs, return_tensors="pt", padding=True)
            # print(curr_inputs['input_ids'])
            if torch.cuda.is_available():
                curr_inputs = curr_inputs.to(0)
            else:
                curr_inputs = curr_inputs.to('cpu')
            try:
                curr_outputs = model.generate(curr_inputs['input_ids'], generation_config=gen_configs)
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                print(f"batch index: {bstart_eidx // batch_size}")
                print(f"curr_inputs['input_ids']: {curr_inputs['input_ids']}")
                print(f"curr instrs: {curr_instrs}")
                for inbatch_eidx in range(len(curr_inputs['input_ids'])):
                    predictions[tidx].append('B')
                    scores[tidx].append(0.0)
                    out_tokens[tidx].append('')
                    golds[tidx].append(curr_labels[inbatch_eidx])
                    output_irregular_cnt += 1
                continue

            total_outlists = curr_outputs.sequences.tolist()
            net_scores = curr_outputs.scores  # seq_len * (batch_size, vocab_size)
            # print(f"net_scores:")
            # print(net_scores)
            # print(f"len(net_scores): {len(net_scores)}; size: {net_scores[0].shape}")
            # assert len(total_outlists) == len(net_scores)

            for inbatch_eidx in range(len(total_outlists)):
                this_net_outlist = []
                this_outlist = total_outlists[inbatch_eidx]
                for i in range(len(this_outlist)):
                    if i < len(curr_inputs['input_ids'][inbatch_eidx]):
                        assert this_outlist[i] == curr_inputs['input_ids'][inbatch_eidx][i].item()
                    else:
                        this_net_outlist.append(this_outlist[i])
                assert len(this_net_outlist) <= len(net_scores), f"{len(this_net_outlist)} vs {len(net_scores)}"
                this_net_scores = [net_scores[i][inbatch_eidx][this_net_outlist[i]].item() for i in range(len(this_net_outlist))]
                # print(f"this_net_scores: {this_net_scores}")
                curr_pred, curr_scr, curr_outtokens, output_irregular_flag = judgement(this_net_scores, this_net_outlist,
                                                                                       tokenizer, single_statement is not None)
                if output_irregular_flag:
                    output_irregular_cnt += 1
                predictions[tidx].append(curr_pred)
                scores[tidx].append(curr_scr)
                out_tokens[tidx] = curr_outtokens
                golds[tidx].append(curr_labels[inbatch_eidx])

    print(f"output_irregular_count: {output_irregular_cnt} / {len(embedded_prompts) * len(embedded_prompts[0])}")

    with open(results_path, 'w', encoding='utf8') as ofp:
        json.dump({'predictions': predictions, 'scores': scores, 'tokens': out_tokens}, ofp, indent=2, ensure_ascii=False)

    for tplt_idx in range(len(embedded_prompts)):
        assert all(x in ['A', 'B', 'C'] for x in predictions[tplt_idx])
        bin_predictions = [True if pred == 'A' else False for pred in predictions[tplt_idx]]
        prec, rec, f_score, _ = precision_recall_fscore_support(golds[tplt_idx], bin_predictions, beta=beta,
                                                                average='binary')
        precisions, recalls, thresholds = precision_recall_curve(golds[tplt_idx], scores[tplt_idx])
        auc_score = auc(recalls, precisions)
        average_prec = average_precision_score(golds[tplt_idx], scores[tplt_idx])
        best_f, best_p, best_r = find_best_f_beta_from_curve(precisions, recalls, beta=beta)
        best_f1, best_pf1, best_rf1 = find_best_f_beta_from_curve(precisions, recalls, beta=1.0)

        print(f"template index: {tplt_idx}; template: {embedded_prompts[tplt_idx][0]};")
        print(f"Ratio of positive predictions: {sum(1 for x in predictions[tplt_idx] if x == 'A') / len(predictions[tplt_idx])};")
        print(f"Binary: f_{beta} score: {f_score}, precision {prec}, recall {rec};")
        print(f"Best F-{beta} score along the curve: {best_f}; precision: {best_p}; recall: {best_r};")
        print(f"Best F1-score along the curve: {best_f1}; precision: {best_pf1}; recall: {best_rf1};")
        print(f"prc AUC: {auc_score}; average precision: {average_prec}")
        if single_statement is not None:
            print(f"Single statement: {single_statement};")
            posi_count = sum(1 for x in predictions[tplt_idx] if x == 'A')
            unknown_count = sum(1 for x in predictions[tplt_idx] if x == 'B')
            nega_count = sum(1 for x in predictions[tplt_idx] if x == 'C')
            print(f"Positive rate: {posi_count} / {len(predictions[tplt_idx])} = {posi_count / len(predictions[tplt_idx])};")
            print(f"Unknown rate: {unknown_count} / {len(predictions[tplt_idx])} = {unknown_count / len(predictions[tplt_idx])};")
            print(f"Negative rate: {nega_count} / {len(predictions[tplt_idx])} = {nega_count / len(predictions[tplt_idx])};")
        plt.plot(recalls, precisions, label=f"Template {tplt_idx}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision Recall Curves")
    plt.legend()
    plt.draw()
    plt.show()
    assert results_path.endswith('.json')
    plt.savefig(f"{results_path}".replace('.json', f'.png'))


def collect_templates(data_path, embedded_path, use_plhr, in_context, has_instruction, tplt_id, single_statement):
    print_template_flag = True
    # start_time = time.time()

    if tplt_id is None:
        if single_statement == 'h':
            effective_sent_template_to_test = [knowledge_templates_to_test['h']]
        elif single_statement == 'p':
            effective_sent_template_to_test = [knowledge_templates_to_test['p']]
        else:
            assert single_statement is None
            effective_sent_template_to_test = sent_template_to_test
    else:
        assert single_statement is None
        effective_sent_template_to_test = [sent_template_to_test[tplt_id]]

    if use_plhr in ['type', 'rp_type_low', 'rp_type_high', 'rp_type_draw', 'rp_type_none', 'randprem_type',
                    'randhyp_type']:
        prem_hyp_pairs = load_typed_general_entries(data_path)
    elif use_plhr in ['original', 'random', 'lowfreq', 'highfreq', 'randprem_orig', 'randhyp_orig', 'rp_original_low',
                      'rp_original_high', 'rp_original_draw', 'rp_original_none']:
        prem_hyp_pairs = load_general_entries(data_path)
    else:
        raise NotImplementedError(f"Unknown placeholder type: {use_plhr}")

    instantiated_templates = [[] for x in range(len(effective_sent_template_to_test))]

    for ent_idx, (prem, hyp, lbl, aligned_flag) in enumerate(prem_hyp_pairs):
        if lbl == 'True':
            lbl = True
        elif lbl == 'False':
            lbl = False
        else:
            raise NotImplementedError(f"Unknown label: {lbl}")
        if single_statement == 'h':
            prem = None
        elif single_statement == 'p':
            hyp = None
        else:
            assert single_statement is None

        for tplt_idx in range(len(effective_sent_template_to_test)):
            curr_t = get_gpt_template(prem, hyp, aligned=aligned_flag, use_plhr=use_plhr,
                                      in_context=in_context,
                                      tplt_fmt=effective_sent_template_to_test[tplt_idx]['s'],
                                      do_neg=effective_sent_template_to_test[tplt_idx]['do_neg'],
                                      use_binary_options=False, single_statement=single_statement,
                                      rev_hyp_args=False, has_instruction=has_instruction)
            if print_template_flag:
                print(f"{curr_t}")
                print_template_flag = False
            else:
                pass
            instantiated_templates[tplt_idx].append({'in': curr_t, 'label': lbl})

    # print(f"Time elapsed: {time.time() - start_time}")
    if embedded_path is not None:
        with open(embedded_path, 'w', encoding='utf8') as ofp:
            json.dump(instantiated_templates, ofp, indent=4, ensure_ascii=False)
        print(f"Saved to {embedded_path}")
    else:
        print(f"Instantiated prompts not saved.")

    return instantiated_templates


def benchmark_results(data_path, results_path, beta: float, tplt_id, use_plhr: str, is_single_statement: bool):
    with open(results_path, 'r', encoding='utf8') as ifp:
        dct = json.load(ifp)
        if isinstance(dct, list):
            predictions = None
            scores = dct
        else:
            assert isinstance(dct, dict)
            predictions = dct['predictions']
            scores = dct['scores']

    if tplt_id is None:
        if is_single_statement:
            effective_sent_template_to_test = knowledge_templates_to_test
        else:
            effective_sent_template_to_test = sent_template_to_test
    else:
        assert is_single_statement is False
        effective_sent_template_to_test = [sent_template_to_test[tplt_id]]

    if use_plhr in ['type', 'rp_type_low', 'rp_type_high', 'rp_type_draw', 'rp_type_none', 'randprem_type',
                    'randhyp_type']:
        prem_hyp_pairs = load_typed_general_entries(data_path)
    elif use_plhr in ['original', 'random', 'lowfreq', 'highfreq', 'randprem_orig', 'randhyp_orig', 'rp_original_low',
                      'rp_original_high', 'rp_original_draw', 'rp_original_none', 'ant']:
        prem_hyp_pairs = load_general_entries(data_path)
    else:
        raise NotImplementedError(f"Unknown placeholder type: {use_plhr}")

    golds = []
    for ent_idx, (prem, hyp, lbl, aligned_flag) in enumerate(prem_hyp_pairs):
        if lbl == 'True':
            lbl = True
        elif lbl == 'False':
            lbl = False
        else:
            raise AssertionError(f"Unknown label: {lbl}")
        golds.append(lbl)

    for tplt_idx in range(len(effective_sent_template_to_test)):
        assert len(golds) == len(scores[tplt_idx])
        prec, rec, f_score, _ = precision_recall_fscore_support(golds, [x > 0.5 for x in scores[tplt_idx]], beta=beta, average='binary')
        _, _, binary_f1, _ = precision_recall_fscore_support(golds, [x > 0.5 for x in scores[tplt_idx]], beta=1.0, average='binary')
        precisions, recalls, thresholds = precision_recall_curve(golds, scores[tplt_idx])
        auc_score = auc(recalls, precisions)
        average_prec = average_precision_score(golds, scores[tplt_idx])
        best_f, best_p, best_r = find_best_f_beta_from_curve(precisions, recalls, beta=beta)
        best_f1, best_pf1, best_rf1 = find_best_f_beta_from_curve(precisions, recalls, beta=1.0)

        print(f"template index: {tplt_idx}; template: {effective_sent_template_to_test[tplt_idx]};")
        print(f"Binary: f_1 score: {binary_f1}, f_{beta} score: {f_score}, precision {prec}, recall {rec};")
        print(f"Best F-{beta} score along the curve: {best_f}; precision: {best_p}; recall: {best_r};")
        print(f"Best F1-score along the curve: {best_f1}; precision: {best_pf1}; recall: {best_rf1};")
        print(f"prc AUC: {auc_score}; average precision: {average_prec}")
        if is_single_statement and predictions is not None:
            posi_count = sum(1 for x in predictions[tplt_idx] if x == 'A')
            unknown_count = sum(1 for x in predictions[tplt_idx] if x == 'B')
            nega_count = sum(1 for x in predictions[tplt_idx] if x == 'C')
            print(f"Positive rate: {posi_count} / {len(predictions[tplt_idx])} = {posi_count / len(predictions[tplt_idx])};")
            print(f"Unknown rate: {unknown_count} / {len(predictions[tplt_idx])} = {unknown_count / len(predictions[tplt_idx])};")
            print(f"Negative rate: {nega_count} / {len(predictions[tplt_idx])} = {nega_count / len(predictions[tplt_idx])};")


def benchmark_any(data_path, results_path, beta: float, tplt_id, is_single_statement: bool):
    with open(results_path, 'r', encoding='utf8') as ifp:
        dct = json.load(ifp)
        if isinstance(dct, list):
            predictions = None
            scores = dct
        else:
            assert isinstance(dct, dict)
            predictions = dct['predictions']
            scores = dct['scores']

    if tplt_id is None:
        if is_single_statement:
            effective_sent_template_to_test = knowledge_templates_to_test
        else:
            effective_sent_template_to_test = sent_template_to_test
    else:
        assert is_single_statement is False
        effective_sent_template_to_test = [sent_template_to_test[tplt_id]]

    golds = []
    with open(data_path, 'r', encoding='utf8') as ifp:
        data_items = json.load(ifp)
        assert len(data_items) < 10
        for data_item in data_items[0]:
            if data_item['label'] == 'True':
                lbl = True
            elif data_item['label'] == 'False':
                lbl = False
            elif isinstance(data_item['label'], bool):
                lbl = data_item['label']
            else:
                raise AssertionError(f"Unknown label: {lbl}")
            golds.append(lbl)

    for tplt_idx in range(len(effective_sent_template_to_test)):
        if len(scores[tplt_idx]) == 0:
            continue
        assert len(golds) == len(scores[tplt_idx])
        prec, rec, f_score, _ = precision_recall_fscore_support(golds, [x > 0.5 for x in scores[tplt_idx]], beta=beta, average='binary')
        _, _, binary_f1, _ = precision_recall_fscore_support(golds, [x > 0.5 for x in scores[tplt_idx]], beta=1.0, average='binary')
        precisions, recalls, thresholds = precision_recall_curve(golds, scores[tplt_idx])
        auc_score = auc(recalls, precisions)
        average_prec = average_precision_score(golds, scores[tplt_idx])
        best_f, best_p, best_r = find_best_f_beta_from_curve(precisions, recalls, beta=beta)
        best_f1, best_pf1, best_rf1 = find_best_f_beta_from_curve(precisions, recalls, beta=1.0)

        print(f"template index: {tplt_idx}; template: {effective_sent_template_to_test[tplt_idx]};")
        print(f"Binary: f_1 score: {binary_f1}, f_{beta} score: {f_score}, precision {prec}, recall {rec};")
        print(f"Best F-{beta} score along the curve: {best_f}; precision: {best_p}; recall: {best_r};")
        print(f"Best F1-score along the curve: {best_f1}; precision: {best_pf1}; recall: {best_rf1};")
        print(f"prc AUC: {auc_score}; average precision: {average_prec}")
        if is_single_statement and predictions is not None:
            posi_count = sum(1 for x in predictions[tplt_idx] if x == 'A')
            unknown_count = sum(1 for x in predictions[tplt_idx] if x == 'B')
            nega_count = sum(1 for x in predictions[tplt_idx] if x == 'C')
            print(f"Positive rate: {posi_count} / {len(predictions[tplt_idx])} = {posi_count / len(predictions[tplt_idx])};")
            print(f"Unknown rate: {unknown_count} / {len(predictions[tplt_idx])} = {unknown_count / len(predictions[tplt_idx])};")
            print(f"Negative rate: {nega_count} / {len(predictions[tplt_idx])} = {nega_count / len(predictions[tplt_idx])};")


def run_any(data_path, results_path, model, tokenizer, gen_configs, model_name, beta, tplt_id, is_single_statement,
            batch_size: int, speedtest_id: int, verbose: bool):
    print(f"Loading data from {data_path}...")
    print(f"Results will be written to {results_path}...")
    with open(data_path, 'r', encoding='utf8') as ifp:
        input_entries = json.load(ifp)

    if isinstance(input_entries[0], list):
        pass
    elif isinstance(input_entries[0], dict) and 'in' in input_entries[0] and 'label' in input_entries[0] and len(input_entries[0]) == 2:
        input_entries = [input_entries]
    else:
        raise ValueError

    predictions = [[] for x in input_entries]
    scores = [[] for x in input_entries]
    out_tokens = [[] for x in input_entries]
    golds = [[] for x in input_entries]
    output_irregular_cnt = 0
    start_time = time.time()

    if speedtest_id is not None:
        start_batch_id = 10*speedtest_id
        end_batch_id = start_batch_id + 10
    else:
        start_batch_id = None
        end_batch_id = None

    for tidx, curr_tplt_instrs in enumerate(input_entries):
        temp_fp = open(results_path % model_name + f'_{tidx}_tmp', 'w', encoding='utf8')
        if tplt_id is not None and tidx != tplt_id:
            print(f'Skipping template {tidx} (keeping only template {tplt_id})...')
            continue
        print(f"Generating predictions for template {tidx}...")
        for bstart_eidx in range(0, len(curr_tplt_instrs), batch_size):
            # if bstart_eidx // batch_size < 100:
            #     print(f"bstart eidx: {bstart_eidx}")
            #     continue
            if start_batch_id is not None and end_batch_id is not None:
                if bstart_eidx // batch_size < start_batch_id or bstart_eidx // batch_size >= end_batch_id:
                    continue
            if bstart_eidx // batch_size % 5 == 0 or verbose:
                durr = time.time() - start_time
                print(f"Batch {bstart_eidx // batch_size} / {len(curr_tplt_instrs) // batch_size + 1}; duration: {durr // 3600}h {durr % 3600 // 60}m {durr % 60}s;")
                # print(golds[tidx])
            bstop_eidx = min(bstart_eidx + batch_size, len(curr_tplt_instrs))
            curr_batch = curr_tplt_instrs[bstart_eidx:bstop_eidx]
            # print(f"curr batch: {len(curr_batch)}")
            curr_instrs = [x['in'] for x in curr_batch]
            curr_labels = [x['label'] for x in curr_batch]
            for cli in range(len(curr_labels)):
                if isinstance(curr_labels[cli], str):
                    if curr_labels[cli] == 'True':
                        curr_labels[cli] = True
                    elif curr_labels[cli] == 'False':
                        curr_labels[cli] = False
                    else:
                        raise AssertionError(f"Unknown label: {curr_labels[cli]}")
                else:
                    assert isinstance(curr_labels[cli], bool)
            # print(tokenizer.pad_token)
            # print(tokenizer.pad_token_id)
            curr_inputs = tokenizer(curr_instrs, return_tensors="pt", padding=True)
            # print(curr_inputs['input_ids'])
            if torch.cuda.is_available():
                curr_inputs = curr_inputs.to(0)
            else:
                curr_inputs = curr_inputs.to('cpu')
            try:
                curr_outputs = model.generate(curr_inputs['input_ids'], generation_config=gen_configs)
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                print(f"curr_inputs['input_ids']: {curr_inputs['input_ids']}")
                print(f"curr instrs: {curr_instrs}")
                for inbatch_eidx in range(len(curr_inputs['input_ids'])):
                    predictions[tidx].append('B')
                    scores[tidx].append(0.0)
                    out_tokens[tidx].append('')
                    golds[tidx].append(curr_labels[inbatch_eidx])
                    output_irregular_cnt += 1
                continue
            # print(curr_outputs)

            total_outlists = curr_outputs.sequences.tolist()
            net_scores = curr_outputs.scores  # seq_len * (batch_size, vocab_size)
            # print(f"net_scores:")
            # print(net_scores)
            # print(f"len(net_scores): {len(net_scores)}; size: {net_scores[0].shape}")
            # assert len(total_outlists) == len(net_scores)

            for inbatch_eidx in range(len(total_outlists)):
                this_net_outlist = []
                this_outlist = total_outlists[inbatch_eidx]
                for i in range(len(this_outlist)):
                    if i < len(curr_inputs['input_ids'][inbatch_eidx]):
                        assert this_outlist[i] == curr_inputs['input_ids'][inbatch_eidx][i].item()
                    else:
                        this_net_outlist.append(this_outlist[i])
                assert len(this_net_outlist) <= len(net_scores), f"{len(this_net_outlist)} vs {len(net_scores)}"
                this_net_scores = [net_scores[i][inbatch_eidx][this_net_outlist[i]].item() for i in
                                   range(len(this_net_outlist))]
                # print(f"this_net_scores: {this_net_scores}")
                curr_pred, curr_scr, curr_outtokens, output_irregular_flag = judgement(this_net_scores, this_net_outlist,
                                                                                       tokenizer, is_single_statement)
                if output_irregular_flag:
                    output_irregular_cnt += 1
                predictions[tidx].append(curr_pred)
                scores[tidx].append(curr_scr)
                out_tokens[tidx].append(curr_outtokens)
                golds[tidx].append(curr_labels[inbatch_eidx])

                o_item = {'pred': curr_pred, 'score': curr_scr, 'out_tokens': curr_outtokens, 'gold': curr_labels[inbatch_eidx]}
                temp_fp.write(json.dumps(o_item) + '\n')
        temp_fp.close()

    total_durr = time.time() - start_time
    print(f"Total duration: {total_durr // 3600}h {total_durr % 3600 // 60}m {total_durr % 60}s;")
    print(f"output_irregular_cnt: {output_irregular_cnt}")
    for tidx in range(len(predictions)):
        if tplt_id is not None and tidx != tplt_id:
            print(f'Skipping template {tidx} (keeping only template {tplt_id})...')
            continue
        assert len(golds[tidx]) == len(scores[tidx])
        assert all(x in ['A', 'B', 'C'] for x in predictions[tidx])
        bin_predictions = [True if pred == 'A' else False for pred in predictions[tidx]]
        # print(golds)
        prec, rec, f_score, _ = precision_recall_fscore_support(golds[tidx], bin_predictions, beta=beta, average='binary')
        precisions, recalls, thresholds = precision_recall_curve(golds[tidx], scores[tidx])
        auc_score = auc(recalls, precisions)
        average_prec = average_precision_score(golds[tidx], scores[tidx])
        best_f, best_p, best_r = find_best_f_beta_from_curve(precisions, recalls, beta=beta)

        print(f"template index: {tidx};")
        print(f"Binary: f_{beta} score: {f_score}, precision {prec}, recall {rec};")
        print(f"Best F-score along the curve: {best_f}; precision: {best_p}; recall: {best_r};")
        print(f"prc AUC: {auc_score}; average precision: {average_prec}")

    speedtest_suffix = f'_speedtest_{speedtest_id}' if speedtest_id is not None else ''
    with open(results_path % model_name + speedtest_suffix, 'w', encoding='utf8') as ofp:
        json.dump({'predictions': predictions, 'scores': scores, 'tokens': out_tokens}, ofp, indent=2, ensure_ascii=False)
    print(f"Results saved to {results_path}.")


def main(args):
    model_path = os.path.join(args.model_root, args.model_name)

    if args.task not in ['benchmark', 'template', 'benchmark_run_any']:
        accelerator = Accelerator()
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # if tokenizer.pad_token is None or tokenizer.pad_token_id < 0:
        #     assert tokenizer.eos_token is not None and tokenizer.eos_token_id >= 0
        #     tokenizer.pad_token = tokenizer.eos_token
        configs = LlamaConfig.from_pretrained(model_path)
        gen_configs = GenerationConfig(max_new_tokens=args.max_new_tokens, output_scores=True,
                                       return_dict_in_generate=True)
        if args.no_accelerate:
            model = LlamaForCausalLM.from_pretrained(model_path)
        else:
            with init_empty_weights():
                model = LlamaForCausalLM._from_config(configs)
            num_gpus = torch.cuda.device_count()
            if args.machine == 'local':
                if num_gpus == 2:
                    max_memory = {0: "4GIB", 1: "9GIB", "cpu": "260GIB"}
                elif num_gpus == 4:
                    max_memory = {0: "4GIB", 1: "9GIB", 2: "9GIB", 3: "9GIB", "cpu": "75GIB"}
                elif num_gpus == 8:
                    max_memory = {0: "4GIB", 1: "8GIB", 2: "8GIB", 3: "8GIB", 4: "8GIB", 5: "8GIB", 6: "8GIB", 7: "8GIB",
                                  "cpu": "180GIB"}
                else:
                    raise AssertionError(f"Unknown number of GPUs: {num_gpus}")
            elif args.machine == 'cloud':
                if num_gpus == 1:
                    max_memory = {0: "26GIB", "cpu": "40GIB"}
                elif num_gpus == 2:
                    max_memory = {0: "24GIB", 1: "28GIB", "cpu": "90GIB"}
                elif num_gpus == 4:
                    max_memory = {0: "22GIB", 1: "28GIB", 2: "28GIB", 3: "28GIB", "cpu": "200GIB"}
                elif num_gpus == 8:
                    max_memory = {0: "20GIB", 1: "28GIB", 2: "28GIB", 3: "28GIB", 4: "28GIB", 5: "28GIB", 6: "28GIB",
                                  7: "28GIB", "cpu": "450GIB"}
                else:
                    raise AssertionError(f"Unknown number of GPUs: {num_gpus}")
            else:
                raise ValueError(f"Unknown machine: {args.machine}")
            model = load_checkpoint_and_dispatch(model, model_path,
                                                 device_map='auto',
                                                 # device_map doesn't matter, since we don't have enough GPU memory altogether
                                                 no_split_module_classes=['LlamaDecoderLayer'],
                                                 max_memory=max_memory,
                                                 offload_folder=None)
            print(model.hf_device_map)
        tokenizer.padding_side = 'left'  # pad on the left because we want to generate output after the last token in input (don't want </s> between input and output!)
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
    else:
        model = None
        tokenizer = None
        gen_configs = None

    if args.tplt_id is not None:
        num_templates = 1
    elif args.single_statement is not None:
        assert args.single_statement in ['h', 'p']
        num_templates = len(knowledge_templates_to_test)
    else:
        num_templates = len(sent_template_to_test)

    instruct_str = '_instruct' if args.instruction else ''

    if args.task == 'lh':
        if args.use_plhr == 'ALL':
            for pidx, plhr in enumerate(['original', 'type', 'lowfreq', 'highfreq']):  # , 'random']:
                print(f"Running with placeholder: {plhr}")
                if args.tplt_id is not None:
                    print(f"Evaluating only template #{args.tplt_id[pidx]}.")
                data_path, results_path = process_paths(args.data_root, args.results_root, args.subset, args.split,
                                                        plhr, args.in_context, instruct_str, num_templates, args.single_statement,
                                                        args.machine, args.model_name)
                run_over_levy_holt(data_path, results_path, model=model, tokenizer=tokenizer, gen_configs=gen_configs,
                                   use_plhr=plhr, in_context=args.in_context, has_instruction=args.instruction,
                                   beta=args.beta, tplt_id=args.tplt_id[pidx] if args.tplt_id is not None else None,
                                   single_statement=args.single_statement, batch_size=args.batch_size)
        elif args.use_plhr == 'ALL_RP_ORIG':
            for pidx, plhr in enumerate(['rp_original_low', 'rp_original_high', 'rp_original_draw', 'rp_original_none']):
                print(f"Running with placeholder: {plhr}")
                if args.tplt_id is not None:
                    print(f"Evaluating only template #{args.tplt_id[pidx]}.")
                data_path, results_path = process_paths(args.data_root, args.results_root, args.subset, args.split,
                                                        plhr, args.in_context, instruct_str, num_templates, args.single_statement,
                                                        args.machine, args.model_name)
                run_over_levy_holt(data_path, results_path, model=model, tokenizer=tokenizer, gen_configs=gen_configs,
                                   use_plhr=plhr, in_context=args.in_context, has_instruction=args.instruction,
                                   beta=args.beta, tplt_id=args.tplt_id[pidx] if args.tplt_id is not None else None,
                                   single_statement=args.single_statement, batch_size=args.batch_size)
        elif args.use_plhr == 'ALL_RP_TYPE':
            for pidx, plhr in enumerate(['rp_type_low', 'rp_type_high', 'rp_type_draw', 'rp_type_none']):
                print(f"Running with placeholder: {plhr}")
                if args.tplt_id is not None:
                    print(f"Evaluating only template #{args.tplt_id[pidx]}.")
                data_path, results_path = process_paths(args.data_root, args.results_root, args.subset, args.split,
                                                        plhr, args.in_context, instruct_str, num_templates, args.single_statement,
                                                        args.machine, args.model_name)
                run_over_levy_holt(data_path, results_path, model=model, tokenizer=tokenizer, gen_configs=gen_configs,
                                   use_plhr=plhr, in_context=args.in_context, has_instruction=args.instruction,
                                   beta=args.beta, tplt_id=args.tplt_id[pidx] if args.tplt_id is not None else None,
                                   single_statement=args.single_statement, batch_size=args.batch_size)
        else:
            data_path, results_path = process_paths(args.data_root, args.results_root, args.subset, args.split,
                                                    args.use_plhr, args.in_context, instruct_str, num_templates,
                                                    args.single_statement, args.machine, args.model_name)
            assert args.tplt_id is None or len(args.tplt_id) == 1
            run_over_levy_holt(data_path, results_path, model=model, tokenizer=tokenizer, gen_configs=gen_configs,
                               use_plhr=args.use_plhr, in_context=args.in_context, has_instruction=args.instruction,
                               beta=args.beta, tplt_id=args.tplt_id[0] if args.tplt_id is not None else None,
                               single_statement=args.single_statement, batch_size=args.batch_size)
    elif args.task == 'template':
        data_path, _ = process_paths(args.data_root, args.results_root, args.subset, args.split, args.use_plhr,
                                     args.in_context, instruct_str, num_templates, args.single_statement,
                                     args.machine, args.model_name)
        assert args.tplt_id is None or len(args.tplt_id) == 1
        if args.single_statement == 'h':
            ss_str = '_honly'
        else:
            assert args.single_statement is None
            ss_str = ''
        embedded_path = data_path.replace('.txt', f'_{args.use_plhr}{ss_str}_tplt{args.tplt_id[0] if args.tplt_id is not None else "N"}_embedded.json')
        assert embedded_path != data_path
        collect_templates(data_path=data_path, embedded_path=embedded_path, use_plhr=args.use_plhr, in_context=args.in_context,
                          has_instruction=args.instruction, tplt_id=args.tplt_id[0] if args.tplt_id is not None else None,
                          single_statement=args.single_statement)
    elif args.task == 'run_any':
        # If running pre-processed hyp/prm-only data, you still need to set the single_statement flag to corresponding values,
        # so that the option matching could work.
        for pidx, (d_path, r_path) in enumerate(zip(args.data_path, args.results_path)):
            if args.tplt_id is not None:
                print(f"Evaluating only template #{args.tplt_id[pidx]}.")
            run_any(d_path, r_path, model, tokenizer, gen_configs, model_name=args.model_name, beta=args.beta,
                    tplt_id=args.tplt_id[pidx] if args.tplt_id is not None else None,
                    is_single_statement=(args.single_statement is not None), batch_size=args.batch_size,
                    speedtest_id=args.speedtest_id, verbose=args.verbose)
    elif args.task == 'benchmark':
        data_path, results_path = process_paths(args.data_root, args.results_root, args.subset, args.split, args.use_plhr,
                                                args.in_context, instruct_str, num_templates, args.single_statement,
                                                args.machine, args.model_name)
        assert args.tplt_id is None or len(args.tplt_id) == 1
        benchmark_results(data_path, results_path, beta=args.beta, tplt_id=args.tplt_id[0] if args.tplt_id is not None else None,
                          use_plhr=args.use_plhr, is_single_statement=(args.single_statement is not None))
    elif args.task == 'benchmark_run_any':
        for pidx, (d_path, r_path) in enumerate(zip(args.data_path, args.results_path)):
            if args.tplt_id is not None:
                print(f"Evaluating only template #{args.tplt_id[pidx]}.")
            benchmark_any(d_path, r_path, beta=args.beta, tplt_id=args.tplt_id[pidx] if args.tplt_id is not None else None,
                          is_single_statement=(args.single_statement is not None))
    elif args.task == 'toy':
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task name: {args.task}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='llama-65b-hf')
    parser.add_argument('--subset', type=str, default='dir')
    parser.add_argument('--data_root', type=str,
                        default='./levyholt_files/',
                        )
    parser.add_argument('--results_root', type=str,
                        default='./results/levyholt_results/llama_results')
    parser.add_argument('--use_plhr', type=str, default='original')
    parser.add_argument('--in_context', type=str, default='none')
    parser.add_argument('--instruction', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--task', type=str, default='lh')
    parser.add_argument('--no_accelerate', action='store_true')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--tplt_id', type=str, default=None)
    parser.add_argument('--machine', type=str, default='local')
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--single_statement', type=str, choices=['h'], default=None)
    parser.add_argument('--speedtest_id', type=int, default=None, help="Used only when task=run_any.")

    parser.add_argument('--data_path', type=str, default=None, help=f"Explicit path on top of data_root, used only when task=run_any.")
    parser.add_argument('--results_path', type=str, default=None,
                        help=f"Explicit path to results file, masks over the default combination, used only when task=run_any.")
    parser.add_argument('--use_cached_paths', action='store_true', help=f"Use cached paths, used only when task=run_any.")
    parser.add_argument('--verbose', action='store_true', help=f"Print out more information, used only when task=run_any.")

    # parser.add_argument()
    args = parser.parse_args()

    if args.tplt_id is not None:
        args.tplt_id = [int(x) for x in args.tplt_id.split(' ')]

    if args.data_path is not None:
        assert args.task in ['run_any', 'benchmark_run_any']
        args.data_path = [os.path.join(args.data_root, args.data_path)]
    if args.results_path is not None:
        assert args.task in ['run_any', 'benchmark_run_any']
        args.results_path = [os.path.join(args.results_root, args.results_path)]

    if args.use_cached_paths:
        assert args.data_path is None
        assert args.results_path is None
        assert args.task in ['run_any', 'benchmark_run_any']
        args.data_path = [
            f'rte_files/rte_test/{args.split}_original.json',
            f'rte_files/rte_test/{args.split}_type.json',
            f'rte_files/rte_test/{args.split}_lowfreq.json',
            f'rte_files/rte_test/{args.split}_highfreq.json',
        ]
        args.results_path = [
            f'rte_results_llama/rte_{args.split}_original_results_{args.model_name}.json',
            f'rte_results_llama/rte_{args.split}_type_results_{args.model_name}.json',
            f'rte_results_llama/rte_{args.split}_lowfreq_results_{args.model_name}.json',
            f'rte_results_llama/rte_{args.split}_highfreq_results_{args.model_name}.json',
        ]
        for i in range(len(args.data_path)):
            args.data_path[i] = os.path.join(args.data_root, args.data_path[i])
            args.results_path[i] = os.path.join(args.results_root, args.results_path[i])

    print(f"Args: {args}")

    main(args)
