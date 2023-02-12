import json
from utils import load_general_entries, load_typed_general_entries, negate
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


INFERENCE_OPTION_STR_TRINARY = "\nA) Entailment\nB) Neutral\nC) Contradiction\nAnswer:"
KNOWLEDGE_OPTION_STR_TRINARY = "\nA) True\nB) Unknown\nC) False\nAnswer:"
INFERENCE_OPTION_STR_BINARY = " Is this True or False?\nA) True\nB) False\nAnswer:"
KNOWLEDGE_OPTION_STR_BINARY = " Is this True or False?\nA) True\nB) False\nAnswer:"

# OPRION_STR = "A) Certain\nB) Likely\nC) Unlikely\nD) Impossible\nAnswer:"  # This 4-way version does not make a difference for the <Google, Youtube> example.


def format_proppairs_with_template(tplt_fmt: str, prem_pred: str, hypo_pred: str, psubj: str, pobj: str, aligned: bool):
    if aligned:
        prem = ' '.join([psubj, prem_pred, pobj]) if prem_pred is not None else None
        hypo = ' '.join([psubj, hypo_pred, pobj])
    else:
        prem = ' '.join([psubj, prem_pred, pobj]) if prem_pred is not None else None
        hypo = ' '.join([pobj, hypo_pred, psubj])
    if prem is not None:
        result_str = tplt_fmt.format(prm=prem, hyp=hypo)
    else:
        result_str = tplt_fmt.format(hyp=hypo)
    return result_str


def acquire_in_context_examples(tplt_fmt: str, use_plhr: str, do_neg: bool, use_binary_options: bool, is_hyp_only: bool):

    if use_binary_options is True:
        positive_option = "A) True"
        negative_option = "B) False"
        if is_hyp_only:
            option_str = KNOWLEDGE_OPTION_STR_BINARY
        else:
            option_str = INFERENCE_OPTION_STR_BINARY
    else:
        positive_option = "A) Entailment"
        negative_option = "B) Neutral"  # TODO: check this when swapping the in-context examples, negatives may become "contradiction" instead of "neutral"
        if is_hyp_only:
            option_str = KNOWLEDGE_OPTION_STR_TRINARY
        else:
            option_str = INFERENCE_OPTION_STR_TRINARY

    if not is_hyp_only:
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

        if use_plhr == 'original':
            exmpl1_psubj = 'Google'
            exmpl1_pobj = 'Youtube'
            exmpl1_aligned = True
            exmpl2_psubj = 'John'
            exmpl2_pobj = 'the mall'
            exmpl2_aligned = True
        elif use_plhr == 'type':
            exmpl1_psubj = 'organization x'
            exmpl1_pobj = 'organization y'
            exmpl1_aligned = True
            exmpl2_psubj = 'person x'
            exmpl2_pobj = 'location y'
            exmpl2_aligned = True
        elif use_plhr == 'shuffled':
            exmpl1_psubj = 'Sony'
            exmpl1_pobj = 'Honda'
            exmpl1_aligned = True
            exmpl2_psubj = 'Angela Merkel'
            exmpl2_pobj = 'Ikea'
            exmpl2_aligned = True
        elif use_plhr == 'xy':
            exmpl1_psubj = 'x'
            exmpl1_pobj = 'y'
            exmpl1_aligned = True
            exmpl2_psubj = 'x'
            exmpl2_pobj = 'y'
            exmpl2_aligned = True
        else:
            raise ValueError("Unknown placeholder type: {}".format(use_plhr))
        context_cot = f"""{format_proppairs_with_template(tplt_fmt, exmpl_p1_pred, exmpl_h1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {positive_option}. Owning is a consequence of buying.

{format_proppairs_with_template(tplt_fmt, exmpl_h1_pred, exmpl_p1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {negative_option}. Owning does not imply buying, the ownership may come from other means.

{format_proppairs_with_template(tplt_fmt, exmpl_h2_pred, exmpl_p2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {negative_option}. {exmpl2_psubj} may have gone to the mall by other means.

{format_proppairs_with_template(tplt_fmt, exmpl_p2_pred, exmpl_h2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {positive_option}. Driving is a means of going to the mall.\n\n"""

        context_lblonly = f"""{format_proppairs_with_template(tplt_fmt, exmpl_p1_pred, exmpl_h1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {positive_option}.

{format_proppairs_with_template(tplt_fmt, exmpl_h1_pred, exmpl_p1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {negative_option}.

{format_proppairs_with_template(tplt_fmt, exmpl_h2_pred, exmpl_p2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {negative_option}.

{format_proppairs_with_template(tplt_fmt, exmpl_p2_pred, exmpl_h2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {positive_option}.\n\n"""
    else:
        assert use_plhr == 'original'
        context_cot = None
        context_lblonly = f"""Google bought Youtube.{option_str} {positive_option}.

The sun rises from the west and sets in the east.{option_str} {negative_option}."""
    return context_cot, context_lblonly


def get_gpt_template(p: str, h: str, aligned: bool, use_plhr: str, in_context: str, tplt_fmt: str, do_neg: bool,
                     use_binary_options: bool, rev_hyp_args = False) -> str:
    """
    Get the template for the premise and hypothesis pair. The template is borrowed from GPT-3.
    :param tplt_idx:
    :param in_context:
    :param use_plhr:
    :param p:
    :param h:
    :return:
    """
    assert h is not None
    is_hyp_only = p is None
    context_cot, context_lblonly = acquire_in_context_examples(tplt_fmt, use_plhr, do_neg, use_binary_options, is_hyp_only)

    def clean_sentence(sent: str, role: str):
        subj, pred, obj = sent.lower().split(',')
        if use_plhr in ['xy']:
            subj = 'X'
            obj = 'Y'
        elif use_plhr in ['original', 'type', 'shuffled']:
            subj = subj.strip()
            obj = obj.strip()
        else:
            raise ValueError(f"Unknown use_plhr value: {use_plhr}")

        if do_neg is True:
            pred = negate(pred.strip())
        else:
            assert do_neg is False
            pred = pred.strip()

        if role == 'hyp' and rev_hyp_args:
            return f'{obj} {pred} {subj}'

        return f'{subj} {pred} {obj}'

    p_sent = clean_sentence(p, 'prem') if p else None
    h_sent = clean_sentence(h, 'hyp') if h else None

    sent_args = {}
    if p_sent:
        sent_args['prm'] = p_sent
    if h_sent:
        sent_args['hyp'] = h_sent

    # template = f"{p_sent}, which means that {h_sent}. \n A) Entailment \n B) Neutral \n C) Contradiction \n answer:"
    if use_binary_options and is_hyp_only:
        option_str = KNOWLEDGE_OPTION_STR_BINARY
    elif use_binary_options and not is_hyp_only:
        option_str = INFERENCE_OPTION_STR_BINARY
    elif not use_binary_options and is_hyp_only:
        option_str = KNOWLEDGE_OPTION_STR_TRINARY
    elif not use_binary_options and not is_hyp_only:
        option_str = INFERENCE_OPTION_STR_TRINARY
    else:
        raise ValueError(f"Unknown combination of use_binary_options and is_hyp_only: {use_binary_options}, {is_hyp_only}")
    template = f"""{tplt_fmt.format_map(sent_args)}{option_str}"""
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
                    top_p: float = 1.0, use_binary_options: bool = False, debug: bool = False):
    if args.dry_run:
        scr = random.random()
        label = scr > 0.5
        return label, scr, DUMMY_RESPONSE

    def option_matcher(output: str, char: str) -> bool:
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
        return False, 0.0, response
    elif option_matcher(answer, 'A'):
        # print("!")
        assert 0 < math.exp(logprob) < 1
        effective_scr = 0.5 + 0.5*math.exp(logprob)
        return True, effective_scr, response
    elif use_binary_options and option_matcher(answer, 'B'):
        assert 0 < math.exp(logprob) < 1
        effective_scr = 0.5 - 0.5 * math.exp(logprob)
        return False, effective_scr, response
    elif (not use_binary_options) and (option_matcher(answer, 'B') or option_matcher(answer, 'C')):
        assert 0 < math.exp(logprob) < 1
        effective_scr = 0.5 - 0.5*math.exp(logprob)
        return False, effective_scr, response
    else:
        print(f"Unexpected answer for binary_options={use_binary_options}: {answer}", file=sys.stderr)
        return False, 0.0, response


def vote(answers: List[bool]):
    return sum(answers) > len(answers) / 2


def retrieve_results_main(args):
    if args.hypothesis_only:
        sent_template_activate_flags = [True]
        sent_template_to_test = [
                {'s': '{hyp}.', 'do_neg': False}
        ]
    else:
        sent_template_activate_flags = [True, True, True, True, True, False, False, False]
        # sent_template_activate_flags = [True, True, True, True, True, True, True, True]
        sent_template_to_test = [
            {'s': "{prm}, which means that {hyp}.", 'do_neg': False},
            {'s': "If {prm}, then {hyp}.", 'do_neg': False},
            {'s': "{hyp}, because {prm}.", 'do_neg': False},
            {'s': "{prm}, so {hyp}.", 'do_neg': False},
            {'s': "{prm} entails {hyp}.", 'do_neg': False},
            {'s': "It is not the case that {hyp}, let alone {prm}.", 'do_neg': False},
            {'s': "{prm}, because {hyp}.", 'do_neg': True},
            {'s': "{hyp}, which means that {prm}.", 'do_neg': True},
        ]
    sent_template_to_test = [x for x, y in zip(sent_template_to_test, sent_template_activate_flags) if y]
    assert args.num_templates == len(sent_template_to_test)

    openai.organization = os.getenv('OPENAI_ORG_ID')
    openai.api_key = os.getenv('OPENAI_API_KEY')

    if args.use_plhr in ['original', 'xy', 'shuffled']:
        prem_hyp_pairs = load_general_entries(args.infn_for_eval)  # these are the premise-hypothesis pairs that are True Entailments
    elif args.use_plhr == 'type':
        prem_hyp_pairs = load_typed_general_entries(args.infn_for_eval)
    else:
        raise AssertionError(f"Unknown use_plhr value: {args.use_plhr}")

    preds = [[] for x in range(args.num_templates+3)]  # the +2 are for voting and maximum
    golds = [[] for x in range(args.num_templates+3)]  # the +2 are for voting and maximum
    responses = [[] for x in range(args.num_templates)]

    ready_entries = []
    try:
        ref_fp = open(args.res_fn+'_ref', 'r', encoding='utf-8')
        for line in ref_fp:
            if len(line) < 2:
                continue
            item = json.loads(line)
            ready_entries.append(item)
        ref_fp.close()
        print(f"Loaded {len(ready_entries)} entries from {args.res_fn+'_ref'}")
    except FileNotFoundError:
        print(f"File {args.res_fn+'_ref'} not found, will start from scratch.")

    ofp = open(args.res_fn, 'w', encoding='utf-8')
    ready_count = 0

    # For each premise-hypothesis pair, get the templates and score them with the model;
    # let the 5 templates vote on which one is better.
    for ent_idx, (prem, hyp, lbl, aligned_flag) in enumerate(prem_hyp_pairs):
        if ent_idx % 1 == 0:
            print(f'Processing entry {ent_idx} of {len(prem_hyp_pairs)};')
            time.sleep(3)

        if lbl == 'True':
            lbl = True
        elif lbl == 'False':
            lbl = False
        else:
            raise AssertionError(f"Unknown label: {lbl}")

        ready_found = False
        for ready_ent in ready_entries:
            if prem == ready_ent['premise'] and hyp == ready_ent['hypothesis']:
                assert lbl == ready_ent['gold']
                ready_found = True
                ready_count += 1
                print(f"Ready entry found for {prem} and {hyp}: cnt: {ready_count};")
                for i in range(args.num_templates):
                    preds[i].append(ready_ent['preds'][i])
                binarized_preds = [x > 0.5 for x in ready_ent['preds']]
                preds[args.num_templates].append(vote(binarized_preds))
                preds[args.num_templates+1].append(any(binarized_preds))
                preds[args.num_templates+2].append(all(binarized_preds))
                for i in range(args.num_templates+3):
                    golds[i].append(ready_ent['gold'])
                ofp.write(json.dumps(ready_ent, ensure_ascii=False) + '\n')
                break
        if ready_found:
            continue

        entry_preds = []
        entry_preds_binarized = []
        for tplt_idx in range(args.num_templates):
            if args.hypothesis_only:
                prem = None
            curr_t = get_gpt_template(prem, hyp, aligned=aligned_flag, use_plhr=args.use_plhr, in_context=args.in_context,
                                      tplt_fmt=sent_template_to_test[tplt_idx]['s'],
                                      do_neg=sent_template_to_test[tplt_idx]['do_neg'], use_binary_options=args.use_binary_options,
                                      rev_hyp_args=args.rev_hyp_args)
            if args.debug:
                print(f"Current prompt:")
                print(curr_t)
            curr_res, curr_scr, response = get_gpt3_output(curr_t, args.model_name, max_tokens=args.max_tokens,
                                                           temperature=args.temperature,
                                                           use_binary_options=args.use_binary_options, debug=args.debug)
            responses[tplt_idx].append(response)
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
            'premise': prem,
            'hypothesis': hyp,
            'preds': entry_preds,
            'gold': lbl,
        }
        ofp.write(json.dumps(out_item, ensure_ascii=False) + '\n')

    saved_responses_fn = args.res_fn.replace('.json', '__response.json')
    with open(saved_responses_fn, 'w', encoding='utf-8') as saved_responses_fp:
        json.dump(responses, saved_responses_fp, indent=4)

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
        print(f"Using placeholders for the subjects and objects? {args.use_plhr}")
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
    # plt.show()
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
    banned_template_ids = [5,6,7]  # These "banned templates" are effective only when calculating benchmark scores from raw results.
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
        print(f"Using placeholders for the subjects and objects? {args.use_plhr}")
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


DUMMY_RESPONSE = json.loads('''
{"id": "cmpl-6eSSiAM7jQ9K1kKxt9nLRtCv4Xqa9", 
"object": "text_completion", 
"created": 1675100548, 
"model": "text-ada-001", 
"choices": [{"text": " C", "index": 0, "logprobs": {"tokens": [" C", "<|endoftext|>", "The", " first", " time", " I", " ever", " saw"], 
"token_logprobs": [-0.47073913, -0.14019404, -1.7835075, -3.4638846, -0.5209532, -0.07405463, -0.05527818, -0.33280665], 
"top_logprobs": [
  {" Answer": -2.3953776, " A": -2.962859, " C": -0.47073913, " B": -1.5746171, " D": -5.110625}, 
  {".": -7.390925, "\\n": -4.163912, "<|endoftext|>": -0.14019404, "\\n\\n": -6.4590025, ")": -2.180008}, 
  {"This": -2.5528586, "\\n": -3.2013762, "I": -1.8916179, "The": -1.7835075, "\\n\\n": -1.919075}, 
  {" company": -4.010942, " following": -3.8313718, " previous": -4.2930555, " new": -4.0588293, " first": -3.4638846}, 
  {" day": -4.54038, " few": -4.996513, " thing": -3.2763078, " step": -2.15514, " time": -0.5209532}, 
  {" i": -7.478826, " I": -0.07405463, " that": -3.3122365, " you": -7.930172, " we": -3.4579055}, 
  {" read": -6.2403545, " met": -4.724202, " ever": -0.05527818, " saw": -3.6453538, " heard": -5.244365}, 
  {" encountered": -4.004792, " saw": -0.33280665, " heard": -3.2756853, " tried": -3.0985432, " used": -2.5871973}
  ], 
"text_offset": [129, 131, 131, 131, 131, 131, 131, 131]}, 
"finish_reason": "stop"}], "usage": {"prompt_tokens": 36, "completion_tokens": 1, "total_tokens": 37}}'''
                            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_fn', type=str,
                        default='./%s_files/with_entities/%s.txt')
    parser.add_argument('--typed_in_fn', type=str,
                        default='./%s_files/with_type/%s%s.txt')  # from '../../entgraph_eval/gfiles/ent/test_dir%s.txt'
    parser.add_argument('--shuffled_in_fn', type=str,
                        default='./%s_files/with_shuffled_entities/%s.txt')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--max_tokens', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--res_fn', type=str, default='./results/gpt3_%s_res_%s_text_%s_%s_icl=%s%s_%d.json')
    parser.add_argument('--use_plhr', type=str, default='original')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--in_context', type=str, default='none')
    parser.add_argument('--num_templates', type=int, default=7)
    parser.add_argument('--hypothesis-only', action='store_true')
    parser.add_argument('--subset', type=str, default='full', choices=['dir', 'full'])
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--only_do_scr', action='store_true')
    parser.add_argument('--dry-run', action='store_true')  # will not call the actual API; instead use random fake data
    parser.add_argument('--rev-hyp-args', action='store_true')
    parser.add_argument('--use-binary-options', action='store_true')

    parser.add_argument('--res_suffix', type=str, default='')
    parser.add_argument('--sleep_after_query', type=float, default=0)

    args = parser.parse_args()
    print(args)
    assert args.use_plhr in ['original', 'shuffled', 'xy', 'type']
    assert not (args.hypothesis_only and (args.in_context not in ['none', 'lbl'])), 'Not Implemented: ICL with Explanations with Hypothesis-only baseline'
    assert not (args.hypothesis_only and (args.use_plhr != 'original')), 'Not Implemented: argument replacements with Hypothesis-only baseline'

    binary_str = '_binary' if args.use_binary_options else '_trinary'
    args.res_fn = args.res_fn % (args.model_name, args.subset, args.split, args.use_plhr, args.in_context, binary_str, args.num_templates)
    if args.rev_hyp_args:
        args.res_fn = args.res_fn.replace('.json', '_rev-hyp-args.json')

    if args.use_plhr in ['original', 'xy']:
        args.infn_for_eval = args.in_fn % (args.subset, args.split)
    elif args.use_plhr == 'shuffled':
        args.infn_for_eval = args.shuffled_in_fn % (args.subset, args.split)
    elif args.use_plhr == 'type':
        args.infn_for_eval = args.typed_in_fn % (args.subset, args.split, '%s')
    else:
        raise NotImplementedError
    print(f"Evaluating {args.infn_for_eval} with model {args.model_name}, and saving results to {args.res_fn}")

    if args.only_do_scr:
        print(f"Getting scores for the full dataset:")
        get_scr_from_full_result(args, dirscr=False)
        print(f"Getting scores for the directional subset:")
        get_scr_from_full_result(args, dirscr=True)
    else:
        retrieve_results_main(args)
