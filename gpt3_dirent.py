import json
from utils import load_entries, load_typed_entries
import argparse
import openai
import os
import random
import sys
import time
from typing import List
from sklearn.metrics import precision_recall_fscore_support


OPRION_STR = "A) Entailment\nB) Neutral\nC) Contradiction\nAnswer:"
# OPRION_STR = "A) Certain\nB) Likely\nC) Unlikely\nD) Impossible\nAnswer:"  # This 4-way version does not make a difference for the <Google, Youtube> example.


def get_gpt_template(p: str, h: str, use_plhr: str, in_context: bool, tplt_fmt: str) -> str:
    """
    Get the template for the premise and hypothesis pair. The template is borrowed from GPT-3.
    :param tplt_idx:
    :param in_context:
    :param use_plhr:
    :param p:
    :param h:
    :return:
    """
    context_prem = 'Google bought Youtube.'
    context_hypo = 'Google owns Youtube.'
    counterfactual_hypo = 'Google sold Youtube.'

    context = f"""
    {tplt_fmt.format(prm=context_prem, hyp=context_hypo)}
    A) Entailment 
    B) Neutral
    C) Contradiction
    Answer: A) Entailment
    
    {tplt_fmt.format(prm=context_hypo, hyp=context_prem)}
    A) Entailment
    B) Neutral
    C) Contradiction
    Answer: B) Neutral
    
    {tplt_fmt.format(prm=context_prem, hyp=counterfactual_hypo)}
    A) Entailment
    B) Neutral
    C) Contradiction
    Answer: C) Contradiction\n\n"""

    p = p.lower()
    h = h.lower()
    p_subj, p_pred, p_obj = p.split(',')
    h_subj, h_pred, h_obj = h.split(',')
    if use_plhr in ['xy']:
        p_subj = 'X'
        p_obj = 'Y'
        h_subj = 'X'
        h_obj = 'Y'
    elif use_plhr in ['none', 'type']:
        p_subj = p_subj.strip()
        p_obj = p_obj.strip()
        h_subj = h_subj.strip()
        h_obj = h_obj.strip()
    else:
        raise ValueError(f"Unknown use_plhr value: {use_plhr}")

    p_sent = f'{p_subj} {p_pred.strip()} {p_obj}'
    h_sent = f'{h_subj} {h_pred.strip()} {h_obj}'

    # template = f"{p_sent}, which means that {h_sent}. \n A) Entailment \n B) Neutral \n C) Contradiction \n answer:"
    template = f"{tplt_fmt.format(prm=p_sent, hyp=h_sent)} \n {OPRION_STR}"
    # if in_context:
    #     template = context + template
    return template


def get_gpt3_comparison_templates(p: str, h: str, use_plhr: bool):
    p = p.lower()
    h = h.lower()
    p_subj, p_pred, p_obj = p.split(',')
    h_subj, h_pred, h_obj = h.split(',')
    if use_plhr in ['xy']:
        p_subj = 'X'
        p_obj = 'Y'
        h_subj = 'X'
        h_obj = 'Y'
    elif use_plhr in ['none', 'type']:
        p_subj = p_subj.strip()
        p_obj = p_obj.strip()
        h_subj = h_subj.strip()
        h_obj = h_obj.strip()
    else:
        raise ValueError(f"Unknown use_plhr value: {use_plhr}")

    p_sent = f'{p_subj} {p_pred.strip()} {p_obj}'
    h_sent = f'{h_subj} {h_pred.strip()} {h_obj}'

    fwd_sent = f"{p_sent}, which means that {h_sent}."
    bwd_sent = f"{h_sent}, which means that {p_sent}."

    if random.random() < 0.5:
        sent1 = fwd_sent
        sent2 = bwd_sent
        label = 'A'
    else:
        sent1 = bwd_sent
        sent2 = fwd_sent
        label = 'B'

    template = f"Which of the following is more likely to be true? \n A) {sent1} \n B) {sent2} \n answer:"

    return template, label


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
        "logprobs": None,
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
                print(f"Retrying...")
                continue

    answer = response['choices'][0]['text'].strip(' ') if response is not None else None
    if debug:
        print(answer)
    if answer is None:
        return False
    elif judger(answer, 'A'):
        return True
    elif judger(answer, 'B') or judger(answer, 'C') or judger(answer, 'D'):
        return False
    else:
        print(f"Unexpected answer: {answer}", file=sys.stderr)
        return False


def retrieve_results_main(args):
    def vote(answers: List[bool]):
        return sum(answers) > len(answers) / 2

    sent_template_to_test = [
        "{prm}, which means that {hyp}.",
        "If {prm}, then {hyp}.",
        "{hyp}, because {prm}.",
        "{prm}, so {hyp}.",
        "It is not the case that {hyp}, let alone {prm}.",
    ]

    openai.organization = 'org-odsud9J1u1ZhPl33GtI35fOR'
    x = os.getenv('OPENAI_API_KEY')
    openai.api_key = os.getenv('OPENAI_API_KEY')

    if args.use_plhr in ['none', 'xy']:
        prem_hyp_pairs = load_entries(args.dirfn_for_eval)  # these are the premise-hypothesis pairs that are True Entailments
    elif args.use_plhr == 'type':
        prem_hyp_pairs = load_typed_entries(args.dirfn_for_eval)
    else:
        raise AssertionError(f"Unknown use_plhr value: {args.use_plhr}")

    prem_hyp_pairs = sorted(list(prem_hyp_pairs), key=lambda s: s[0])

    preds = [[] for x in range(args.num_templates+2)]  # the +2 are for voting and maximum
    golds = [[] for x in range(args.num_templates+2)]  # the +2 are for voting and maximum

    ready_entries = []
    try:
        ref_fp = open(args.res_fn+'_ref', 'r', encoding='utf-8')
        for line in ref_fp:
            if len(line) < 2:
                continue
            item = json.loads(line)
            ready_entries.append(item)
        ref_fp.close()
    except FileNotFoundError:
        pass

    ofp = open(args.res_fn, 'w', encoding='utf-8')
    ready_count = 0

    # For each premise-hypothesis pair, get the templates and score them with the model;
    # let the 5 templates vote on which one is better.
    for ent_idx, (prem, hyp) in enumerate(prem_hyp_pairs):
        if ent_idx % 20 == 0:
            print(f'Processing entry {ent_idx} of {len(prem_hyp_pairs)};')
            # time.sleep(5)

        ready_found = False
        for ready_ent in ready_entries:
            if prem == ready_ent['premise'] and hyp == ready_ent['hypothesis']:
                ready_found = True
                ready_count += 1
                print(f"Ready entry found for {prem} and {hyp}: cnt: {ready_count};")
                for i in range(args.num_templates):
                    preds[i].append(ready_ent['forward_res'][i])
                    golds[i].append(True)
                preds[args.num_templates].append(vote(ready_ent['forward_res']))
                golds[args.num_templates].append(True)
                preds[args.num_templates+1].append(any(ready_ent['forward_res']))
                golds[args.num_templates+1].append(True)

                for i in range(args.num_templates):
                    preds[i].append(ready_ent['reversed_res'][i])
                    golds[i].append(False)
                preds[args.num_templates].append(vote(ready_ent['reversed_res']))
                golds[args.num_templates].append(False)
                preds[args.num_templates+1].append(any(ready_ent['reversed_res']))
                golds[args.num_templates+1].append(False)
                ofp.write(json.dumps(ready_ent, ensure_ascii=False) + '\n')
                break
        if ready_found:
            continue

        forward_preds = []
        backward_preds = []
        for tplt_idx in range(args.num_templates):
            forward_t = get_gpt_template(prem, hyp, use_plhr=args.use_plhr, in_context=args.in_context,
                                         tplt_fmt=sent_template_to_test[tplt_idx])
            reversed_t = get_gpt_template(hyp, prem, use_plhr=args.use_plhr, in_context=args.in_context,
                                          tplt_fmt=sent_template_to_test[tplt_idx])
            if args.debug:
                print(f"Forward prompt:")
                print(forward_t)
                print(f"Reversed prompt:")
                print(reversed_t)
            forward_res = get_gpt3_output(forward_t, args.model_name, max_tokens=args.max_tokens,
                                          temperature=args.temperature, debug=args.debug)
            reversed_res = get_gpt3_output(reversed_t, args.model_name, max_tokens=args.max_tokens,
                                           temperature=args.temperature, debug=args.debug)
            assert isinstance(forward_res, bool) and isinstance(reversed_res, bool)
            preds[tplt_idx].append(forward_res)
            forward_preds.append(forward_res)
            golds[tplt_idx].append(True)
            preds[tplt_idx].append(reversed_res)
            backward_preds.append(reversed_res)
            golds[tplt_idx].append(False)
        preds[args.num_templates].append(vote(forward_preds))
        golds[args.num_templates].append(True)
        preds[args.num_templates+1].append(any(forward_preds))
        golds[args.num_templates+1].append(True)
        preds[args.num_templates].append(vote(backward_preds))
        golds[args.num_templates].append(False)
        preds[args.num_templates+1].append(any(backward_preds))
        golds[args.num_templates+1].append(False)
        out_item = {
            'premise': prem,
            'hypothesis': hyp,
            'forward_res': forward_preds,
            'reversed_res': backward_preds,
        }
        ofp.write(json.dumps(out_item, ensure_ascii=False) + '\n')

    for tplt_idx in range(args.num_templates+2):
        prec, rec, f1, _ = precision_recall_fscore_support(golds[tplt_idx], preds[tplt_idx], beta=1.0, average='binary')
        if tplt_idx == args.num_templates:
            print(f"Voting:")
        elif tplt_idx == args.num_templates+1:
            print(f"Maximum:")
        else:
            print(f"Template {tplt_idx}: {sent_template_to_test[tplt_idx]}")
        print(f"Using placeholders for the subjects and objects? {args.use_plhr}")
        print(f"Precision: {prec}, Recall: {rec}, F1: {f1};")
        ofp.close()
        print(f"Finished! Results written to {args.res_fn}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_fn', type=str,
                        default='./dir_files/with_entities/%s.txt')
    parser.add_argument('--typed_dir_fn', type=str,
                        default='./dir_files/with_type/%s_dir%s.txt')  # from '../../entgraph_eval/gfiles/ent/test_dir%s.txt'
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--res_fn', type=str, default='./results/gpt3_%s_res_dir_text_%s_%s.json')
    parser.add_argument('--use_plhr', type=str, default='none')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--in_context', action='store_true')
    parser.add_argument('--num_templates', type=int, default=5)
    parser.add_argument('--split', type=str, default='dev')

    args = parser.parse_args()
    assert args.use_plhr in ['none', 'xy', 'type']
    model_version = args.model_name.split('-')[1]
    assert model_version in ['davinci', 'curie', 'babbage', 'ada']
    args.res_fn = args.res_fn % (model_version, args.split, args.use_plhr)
    args.dirfn_for_eval = args.dir_fn % args.split if args.use_plhr in ['none', 'xy'] else args.typed_dir_fn % (args.split, '%s')
    print(f"Evaluating {args.dirfn_for_eval} with model {model_version}, and saving results to {args.res_fn}")

    retrieve_results_main(args)
