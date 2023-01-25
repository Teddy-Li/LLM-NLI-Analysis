import json
from utils import load_pseudoent_entries, negate
import argparse
import openai
import os
import random
import sys
import time
import math
from typing import List
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, auc
import matplotlib.pyplot as plt


OPRION_STR = "A) Entailment\nB) Neutral\nC) Contradiction\nAnswer:"
# OPRION_STR = "A) Certain\nB) Likely\nC) Unlikely\nD) Impossible\nAnswer:"  # This 4-way version does not make a difference for the <Google, Youtube> example.


def format_proppairs_with_template(tplt_fmt: str, prem_pred: str, hypo_pred: str, psubj: str, pobj: str, aligned: bool):
    if aligned:
        prem = ' '.join([psubj, prem_pred, pobj])
        hypo = ' '.join([psubj, hypo_pred, pobj])
    else:
        prem = ' '.join([psubj, prem_pred, pobj])
        hypo = ' '.join([pobj, hypo_pred, psubj])
    result_str = tplt_fmt.format(prm=prem, hyp=hypo)
    return result_str


def acquire_pseudotype_incontext_examples(tplt_fmt: str, do_neg: bool):
    if do_neg:
        exmpl_p1_pred = negate('bought')
        exmpl_h1_pred = negate('owns')
        exmpl_p2_pred = negate('drove to')
        exmpl_h2_pred = negate('went to')
    else:
        exmpl_p1_pred = 'bought'
        exmpl_h1_pred = 'owns'
        exmpl_p2_pred = 'drove to'
        exmpl_h2_pred = 'went to'

    exmpl1_psubj = 'Google'
    exmpl1_pobj = 'Youtube'
    exmpl1_aligned = True
    exmpl2_psubj = 'John'
    exmpl2_pobj = 'the mall'
    exmpl2_aligned = True

    exmpl1_argtypesent = f'{exmpl1_psubj} is a company and {exmpl1_pobj} is a website.'
    exmpl2_argtypesent = f'{exmpl2_psubj} is a person and {exmpl2_pobj} is a location.'

    context_cot = f"""{exmpl1_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_p1_pred, exmpl_h1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}
A) Entailment 
B) Neutral
C) Contradiction
Answer: A) Entailment. Owning is a consequence of buying.

{exmpl1_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_h1_pred, exmpl_p1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}
A) Entailment
B) Neutral
C) Contradiction
Answer: B) Neutral. Owning does not imply buying, the ownership may come from other means.

{exmpl2_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_h2_pred, exmpl_p2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}
A) Entailment
B) Neutral
C) Contradiction
Answer: B) Neutral. John may have gone to the mall by other means.

{exmpl2_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_p2_pred, exmpl_h2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}
A) Entailment 
B) Neutral
C) Contradiction
Answer: A) Entailment. Driving is a means of going to the mall.\n\n"""

    context_lblonly = f"""{exmpl1_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_p1_pred, exmpl_h1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}
A) Entailment 
B) Neutral
C) Contradiction
Answer: A) Entailment.

{exmpl1_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_h1_pred, exmpl_p1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}
A) Entailment
B) Neutral
C) Contradiction
Answer: B) Neutral.

{exmpl2_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_h2_pred, exmpl_p2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}
A) Entailment
B) Neutral
C) Contradiction
Answer: B) Neutral.

{exmpl2_argtypesent} {format_proppairs_with_template(tplt_fmt, exmpl_p2_pred, exmpl_h2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}
A) Entailment 
B) Neutral
C) Contradiction
Answer: A) Entailment.\n\n"""
    return context_cot, context_lblonly


def get_pseudoent_gpt_template(dct: dict, in_context: str, tplt_fmt: str, do_neg: bool) -> str:
    """
    Get the template for the premise and hypothesis pair. The template is borrowed from GPT-3.
    :param tplt_idx:
    :param in_context:
    :param p:
    :param h:
    :return:
    """
    context_cot, context_lblonly = acquire_pseudotype_incontext_examples(tplt_fmt, do_neg=do_neg)

    p = dct['prem'].lower()
    h = dct['hyp'].lower()
    p_subj, p_pred, p_obj = p.split(',')
    h_subj, h_pred, h_obj = h.split(',')

    p_subj = p_subj.strip()
    p_obj = p_obj.strip()
    h_subj = h_subj.strip()
    h_obj = h_obj.strip()

    if do_neg is True:
        h_pred = negate(h_pred.strip())
        p_pred = negate(p_pred.strip())
    else:
        assert do_neg is False
        h_pred = h_pred.strip()
        p_pred = p_pred.strip()

    p_sent = f'{p_subj} {p_pred} {p_obj}'
    h_sent = f'{h_subj} {h_pred} {h_obj}'
    argtype_sent = f'{p_subj} is a {dct["prem_tsubj"]} and {p_obj} is a {dct["prem_tobj"]}.'

    # template = f"{p_sent}, which means that {h_sent}. \n A) Entailment \n B) Neutral \n C) Contradiction \n answer:"
    template = f"""{argtype_sent} {tplt_fmt.format(prm=p_sent, hyp=h_sent)}
{OPRION_STR}"""
    if in_context == 'cot':
        template = context_cot + template
    elif in_context == 'lbl':
        template = context_lblonly + template
    elif in_context == 'none':
        pass
    else:
        raise ValueError(f"Unknown in_context value: {in_context}")
    return template


def wrap_prompt(prompt, model_name: str = "text-davinci-003", max_tokens: int = 32, temperature: float = 0.0,
                top_p: float = 1.0):
    ret_dict = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "stream": False,
        "logprobs": 5,
        "stop": "\n"
    }
    return ret_dict


def get_gpt3_output(prompt: str, model_name: str = "text-davinci-003", max_tokens: int = 32, temperature: float = 0.0,
                    top_p: float = 1.0, debug: bool = False):
    def judger(output: str, char: str) -> bool:
        if output == char:
            return True
        elif output == char.lower():
            return True
        elif output.startswith(char+')'):
            return True
        elif output.startswith(char + ' '):
            return True
        elif output.startswith(char + '.'):
            return True
        elif output.startswith(char + '-'):
            return True
        else:
            return False

    prompt_dict = wrap_prompt(prompt, model_name, max_tokens, temperature, top_p)
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
                time.sleep(args.sleep_after_query)
                print(f"Retrying...")
                continue

    if response is not None:
        answer = response['choices'][0]['text'].strip(' ')
        if response['choices'][0]['logprobs']['tokens'][0].strip() == ':':
            logprobs_first_token = response['choices'][0]['logprobs']['tokens'][1]
        else:
            logprobs_first_token = response['choices'][0]['logprobs']['tokens'][0]
        if logprobs_first_token.strip().lower() not in ['a', 'b', 'c']:
            print(f"Error in logprobs_first_token: {logprobs_first_token}", file=sys.stderr)
            pass
        logprob = response['choices'][0]['logprobs']['token_logprobs'][0]
    else:
        answer = None
        logprobs_first_token = None
        logprob = None

    if debug:
        print(answer)
    if answer is None:
        return False, 0.0
    elif judger(answer, 'A'):
        # print("!")
        assert 0 < math.exp(logprob) < 1
        effective_scr = 0.5 + 0.5*math.exp(logprob)
        return True, effective_scr
    elif judger(answer, 'B') or judger(answer, 'C') or judger(answer, 'D'):
        assert 0 < math.exp(logprob) < 1
        effective_scr = 0.5 - 0.5*math.exp(logprob)
        return False, effective_scr
    else:
        print(f"Unexpected answer: {answer}", file=sys.stderr)
        return False, 0.0


def vote(answers: List[bool]):
    return sum(answers) > len(answers) / 2


def retrieve_results_main(args):
    sent_template_activate_flags = [True, True, True, True, False, False, False]
    # sent_template_activate_flags = [True, True, True, True, True, True, True]
    sent_template_to_test = [
        {'s': "{prm}, which means that {hyp}.", 'do_neg': False},
        {'s': "If {prm}, then {hyp}.", 'do_neg': False},
        {'s': "{hyp}, because {prm}.", 'do_neg': False},
        {'s': "{prm}, so {hyp}.", 'do_neg': False},
        {'s': "It is not the case that {hyp}, let alone {prm}.", 'do_neg': False},
        {'s': "{prm}, because {hyp}.", 'do_neg': True},
        {'s': "{hyp}, which means that {prm}.", 'do_neg': True},
    ]
    sent_template_to_test = [x for x, y in zip(sent_template_to_test, sent_template_activate_flags) if y]
    assert args.num_templates == len(sent_template_to_test)

    openai.organization = 'org-odsud9J1u1ZhPl33GtI35fOR'
    x = os.getenv('OPENAI_API_KEY')
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Load data
    prem_hyp_dicts = load_pseudoent_entries(args.infn_for_eval)

    preds = [[] for x in range(args.num_templates+3)]  # the +2 are for voting and maximum
    golds = [[] for x in range(args.num_templates+3)]  # the +2 are for voting and maximum

    ofp = open(args.res_fn, 'w', encoding='utf-8')

    # For each premise-hypothesis pair, get the templates and score them with the model;
    # let the 5 templates vote on which one is better.
    for ent_idx, dct in enumerate(prem_hyp_dicts):
        if ent_idx % 5 == 0:
            print(f'Processing entry {ent_idx} of {len(prem_hyp_dicts)};')
            # time.sleep(5)

        if dct['label'] == 'True':
            lbl = True
        else:
            assert dct['label'] == 'False'
            lbl = False

        entry_preds = []
        entry_preds_binarized = []
        for tplt_idx in range(args.num_templates):
            curr_t = get_pseudoent_gpt_template(dct, in_context=args.in_context,
                                         tplt_fmt=sent_template_to_test[tplt_idx]['s'],
                                         do_neg=sent_template_to_test[tplt_idx]['do_neg'])
            if args.debug:
                print(f"Current prompt:")
                print(curr_t)
            curr_res, curr_scr = get_gpt3_output(curr_t, args.model_name, max_tokens=args.max_tokens,
                                          temperature=args.temperature, debug=args.debug)
            assert isinstance(curr_res, bool) and isinstance(curr_scr, float)
            preds[tplt_idx].append(curr_scr)  # here the scr > 0.5 means binary-True, and < 0.5 means binary-False
            entry_preds_binarized.append(curr_res)
            entry_preds.append(curr_scr)
            if args.sleep_after_query > 0:
                time.sleep(args.sleep_after_query)
        preds[args.num_templates].append(vote(entry_preds_binarized))
        preds[args.num_templates+1].append(any(entry_preds_binarized))
        preds[args.num_templates+2].append(all(entry_preds_binarized))
        for i in range(args.num_templates+3):
            golds[i].append(lbl)

        out_item = {
            'premise': dct['prem'],
            'hypothesis': dct['hyp'],
            'preds': entry_preds,
            'gold': lbl,
        }
        ofp.write(json.dumps(out_item, ensure_ascii=False) + '\n')

    for tplt_idx in range(args.num_templates+3):
        # Calculate the binary scores
        curr_tplt_binarized_preds = [x > 0.5 for x in preds[tplt_idx]]
        prec, rec, f1, _ = precision_recall_fscore_support(golds[tplt_idx], curr_tplt_binarized_preds, beta=1.0,
                                                           average='binary')
        if tplt_idx == args.num_templates:
            print(f"Voting:")
        elif tplt_idx == args.num_templates+1:
            print(f"Any:")
        elif tplt_idx == args.num_templates+2:
            print(f"Consensus:")
        else:
            print(f"Template {tplt_idx}: {sent_template_to_test[tplt_idx]}")
        print(f"Precision: {prec}, Recall: {rec}, F1: {f1};")

        # Calculate the precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(golds[tplt_idx], preds[tplt_idx])
        auc_score = auc(recalls, precisions)
        average_prec = average_precision_score(golds[tplt_idx], preds[tplt_idx])
        print(f"AUC: {auc_score}, Average precision: {average_prec}")
        if tplt_idx < args.num_templates:
            plt.plot(recalls, precisions, label=f"Template {tplt_idx}")
        ofp.close()
        print(f"Finished! Results written to {args.res_fn}.")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision Recall Curves")
    plt.legend()
    plt.draw()
    plt.show()
    assert args.res_fn.endswith('.json')
    plt.savefig(f"{args.res_fn}".replace('.json', '.png'))

    if args.subset == 'full':
        print(f"Also doing evaluation on the directional subset:")
        try:
            get_scr_from_full_result(args, dirscr=True)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Skipping directional subset evaluation.")


def get_scr_from_full_result(args, dirscr: bool):
    banned_template_ids = [4,5,6]
    if dirscr:
        diridx_fpath = f'./dir_files/with_entities/{args.split}_idxes.json'
        with open(diridx_fpath, 'r', encoding='utf-8') as diridx_fp:
            diridxes = json.load(diridx_fp)
    else:
        diridxes = None
    full_results = []
    with open(args.res_fn, 'r', encoding='utf-8') as res_fp:
        for line in res_fp:
            full_results.append(json.loads(line))

    preds = [[] for x in range(args.num_templates + 3)]  # the +2 are for voting and maximum
    golds = [[] for x in range(args.num_templates + 3)]  # the +2 are for voting and maximum
    for ridx, res_entry in enumerate(full_results):
        if diridxes is not None and ridx not in diridxes:
            continue
        eligible_preds = []
        for tplt_idx in range(args.num_templates):
            if tplt_idx in banned_template_ids:
                continue
            preds[tplt_idx].append(res_entry['preds'][tplt_idx])
            eligible_preds.append(res_entry['preds'][tplt_idx])
        eligible_preds_binarized = [x > 0.5 for x in eligible_preds]
        preds[args.num_templates].append(vote(eligible_preds_binarized))
        preds[args.num_templates+1].append(any(eligible_preds_binarized))
        preds[args.num_templates+2].append(all(eligible_preds_binarized))
        for i in range(args.num_templates+3):
            golds[i].append(res_entry['gold'])

    for tplt_idx in range(args.num_templates+3):
        if tplt_idx in banned_template_ids:
            continue
        # Calculate the binary scores
        curr_tplt_binarized_preds = [x > 0.5 for x in preds[tplt_idx]]
        prec, rec, f1, _ = precision_recall_fscore_support(golds[tplt_idx], curr_tplt_binarized_preds, beta=1.0,
                                                           average='binary')
        if tplt_idx == args.num_templates:
            print(f"Voting:")
        elif tplt_idx == args.num_templates + 1:
            print(f"Any:")
        elif tplt_idx == args.num_templates + 2:
            print(f"Consensus:")
        else:
            print(f"Template {tplt_idx}:")
        print(f"Precision: {prec}, Recall: {rec}, F1: {f1};")

        # Calculate the precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(golds[tplt_idx], preds[tplt_idx])
        auc_score = auc(recalls, precisions)
        average_prec = average_precision_score(golds[tplt_idx], preds[tplt_idx])
        print(f"AUC: {auc_score}, Average precision: {average_prec}")
        if tplt_idx < args.num_templates:
            plt.plot(recalls, precisions, label=f"Template {tplt_idx}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision Recall Curves")
    plt.legend()
    plt.draw()
    plt.show()
    assert args.res_fn.endswith('.json')
    plt.savefig(f"{args.res_fn}".replace('.json', '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_fn', type=str,
                        default='./%s_files/with_pseudoents/%s.txt')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--max_tokens', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--res_fn', type=str, default='./results/gpt3_%s_res_%s_text_%s_pseudoent_icl=%s.json')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--in_context', type=str, default='none')
    parser.add_argument('--num_templates', type=int, default=7)
    parser.add_argument('--subset', type=str, default='full', choices=['dir', 'full'])
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--only_do_scr', action='store_true')

    parser.add_argument('--sleep_after_query', type=float, default=0)

    args = parser.parse_args()
    print(args)
    args.res_fn = args.res_fn % (args.model_name, args.subset, args.split, args.in_context)
    args.infn_for_eval = args.in_fn % (args.subset, args.split)
    print(f"Evaluating {args.infn_for_eval} with model {args.model_name}, and saving results to {args.res_fn}")

    if args.only_do_scr:
        print(f"Getting scores for the full dataset:")
        get_scr_from_full_result(args, dirscr=False)
        print(f"Getting scores for the directional subset:")
        get_scr_from_full_result(args, dirscr=True)
    else:
        retrieve_results_main(args)
