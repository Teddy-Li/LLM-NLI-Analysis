import re
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, auc, \
    roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import heapq


INFERENCE_OPTION_STR_TRINARY = "\nA) Entailment\nB) Neutral\nC) Contradiction\nAnswer:"
KNOWLEDGE_OPTION_STR_TRINARY = "\nA) True\nB) Unknown\nC) False\nAnswer:"
INFERENCE_OPTION_STR_BINARY = " Is this True or False?\nA) True\nB) False\nAnswer:"
KNOWLEDGE_OPTION_STR_BINARY = " Is this True or False?\nA) True\nB) False\nAnswer:"


def load_general_entries(in_fn):
    # these original orders must be kept for the indices to match
    prem_hyps = []
    with open(in_fn, 'r', encoding='utf8') as fp:
        for line in fp:
            if len(line) < 2:
                continue
            lst = line.rstrip().split('\t')
            if len(lst) == 4:
                hyp, prem, label, lang = lst
            elif len(lst) == 3:
                hyp, prem, label = lst
            else:
                raise ValueError(f"Unexpected line: {line}")
            h_subj, h_pred, h_obj = hyp.split(',')
            p_subj, p_pred, p_obj = prem.split(',')
            if h_subj == p_obj or h_obj == p_subj:
                aligned_flag = False
            else:
                aligned_flag = True
            prem_hyps.append((prem, hyp, label, aligned_flag))
    return prem_hyps


def load_typed_general_entries(in_fn) -> List[Tuple[str, str, str, bool]]:
    # these original orders must be kept for the indices to match
    in_nl_fn = in_fn % ''
    in_rels_fn = in_fn % '_rels'

    prem_hyps = []
    with open(in_nl_fn, 'r', encoding='utf8') as nfp, open(in_rels_fn, 'r', encoding='utf8') as rfp:
        for n_line, r_line in zip(nfp, rfp):
            if len(n_line) < 2 or len(r_line) < 2:
                assert len(n_line) < 2 and len(r_line) < 2
                continue
            hyp_n, prem_n, label_n = n_line.rstrip().split('\t')
            hyp_r, prem_r, label_r = r_line.rstrip().split('\t')
            assert label_n == label_r
            try:
                hyp_pred, hyp_subj_typed, hyp_obj_typed = hyp_r.split(' ')
                hyp_subj, hyp_tsubj = hyp_subj_typed.split('::')
                hyp_obj, hyp_tobj = hyp_obj_typed.split('::')
            except ValueError:
                hyp_tsubj = None
                hyp_tobj = None
                hyp_subj = None
                hyp_obj = None
            try:
                prem_pred, prem_subj_typed, prem_obj_typed = prem_r.split(' ')
                prem_subj, prem_tsubj = prem_subj_typed.split('::')
                prem_obj, prem_tobj = prem_obj_typed.split('::')
            except ValueError:
                prem_tsubj = None
                prem_tobj = None
                prem_subj = None
                prem_obj = None

            # Figure out the aligned flag: when we know the args are not aligned, we set the flag to False;
            # Otherwise, if we know it to be aligned, or we don't know, we both set the flag to True

            orig_prem_subj, orig_prem_pred, orig_prem_obj = [part.strip().lower() for part in prem_n.split(',')]
            orig_hyp_subj, orig_hyp_pred, orig_hyp_obj = [part.strip().lower() for part in hyp_n.split(',')]

            if orig_prem_subj == orig_hyp_obj or orig_prem_obj == orig_hyp_subj:
                nl_aligned_flag = False
            else:
                nl_aligned_flag = True

            # When a passive is parsed into an active construction, the arguments are switched. Unswitch them!
            prem_ccg_arg_positions = [int(c) for c in prem_pred if c.isdigit()]
            if orig_prem_pred.endswith(' by') and '.by' not in prem_pred and prem_ccg_arg_positions and prem_ccg_arg_positions[0] == 1:
                prem_subj, prem_obj = prem_obj, prem_subj
                prem_tsubj, prem_tobj = prem_tobj, prem_tsubj

            hyp_ccg_arg_positions = [int(c) for c in hyp_pred if c.isdigit()]
            if orig_hyp_pred.endswith(' by') and '.by' not in hyp_pred and hyp_ccg_arg_positions and hyp_ccg_arg_positions[0] == 1:
                hyp_subj, hyp_obj = hyp_obj, hyp_subj
                hyp_tsubj, hyp_tobj = hyp_tobj, hyp_tsubj

            if prem_subj is not None and hyp_obj is not None and prem_subj == hyp_obj:
                parser_aligned_flag = False
            elif prem_obj is not None and hyp_subj is not None and prem_obj == hyp_subj:
                parser_aligned_flag = False
            else:
                parser_aligned_flag = True

            aligned_flag = False if (nl_aligned_flag is False) or (parser_aligned_flag is False) else True

            # Copy types in the case of a bad parse
            if hyp_tsubj is None or hyp_tobj is None:
                # assert aligned_flag is True, f'Not aligned: {prem_tsubj}, {prem_tobj} =/=> {hyp_tsubj}, {hyp_tobj}'
                if aligned_flag:
                    hyp_tsubj, hyp_tobj = prem_tsubj, prem_tobj
                else:
                    hyp_tsubj, hyp_tobj = prem_tobj, prem_tsubj
            if prem_tsubj is None or prem_tobj is None:
                # assert aligned_flag is True, f'Not aligned: {prem_tsubj}, {prem_tobj} =/=> {hyp_tsubj}, {hyp_tobj}'
                if aligned_flag:
                    prem_tsubj, prem_tobj = hyp_tsubj, hyp_tobj
                else:
                    prem_tsubj, prem_tobj = hyp_tobj, hyp_tsubj
            assert hyp_tsubj is not None and hyp_tobj is not None and prem_tsubj is not None and prem_tobj is not None

            if {prem_tsubj, prem_tobj} != {hyp_tsubj, hyp_tobj}:
                # Assume that one of the types will match across and the other will be specific/thing
                # Identify which pairing is the same, and reassign the other pairing.
                if prem_tsubj == hyp_tsubj:
                    prem_tobj, hyp_tobj = list({prem_tobj, hyp_tobj}.difference({'thing'}))[-1:] * 2
                elif prem_tsubj == hyp_tobj:
                    prem_tobj, hyp_tsubj = list({prem_tobj, hyp_tsubj}.difference({'thing'}))[-1:] * 2
                elif prem_tobj == hyp_tsubj:
                    prem_tsubj, hyp_tobj = list({prem_tsubj, hyp_tobj}.difference({'thing'}))[-1:] * 2
                elif prem_tobj == hyp_tobj:
                    prem_tsubj, hyp_tsubj = list({prem_tsubj, hyp_tsubj}.difference({'thing'}))[-1:] * 2
                elif prem_tsubj == 'thing' and prem_tobj == 'thing':
                    prem_tsubj = hyp_tsubj
                    prem_tobj = hyp_tobj
                elif hyp_tsubj == 'thing' and hyp_tobj == 'thing':
                    hyp_tsubj = prem_tsubj
                    hyp_tobj = prem_tobj

            assert {prem_tsubj, prem_tobj} == {hyp_tsubj, hyp_tobj}, \
                f'There is a type mismatch: {prem_tsubj}, {prem_tobj} =/=> {hyp_tsubj}, {hyp_tobj}'

            prem_s, prem_p, prem_o = prem_n.split(',')
            hyp_s, hyp_p, hyp_o = hyp_n.split(',')
            prem_tsubj = prem_tsubj.replace('_', ' ')
            prem_tobj = prem_tobj.replace('_', ' ')
            hyp_tsubj = hyp_tsubj.replace('_', ' ')
            hyp_tobj = hyp_tobj.replace('_', ' ')

            if aligned_flag is True:
                prem_tsubj += ' X'
                prem_tobj += ' Y'
                hyp_tsubj += ' X'
                hyp_tobj += ' Y'
            else:
                prem_tsubj += ' X'
                prem_tobj += ' Y'
                hyp_tsubj += ' Y'
                hyp_tobj += ' X'

            # Last check in case we mislabeled X and Y variables for any reason
            # (possibly for irregular passives which weren't caught, etc)
            if {prem_tsubj, prem_tobj} != {hyp_tsubj, hyp_tobj}:
                if aligned_flag:
                    hyp_tsubj, hyp_tobj = prem_tsubj, prem_tobj
                else:
                    hyp_tsubj, hyp_tobj = prem_tobj, prem_tsubj

            prem = ', '.join([prem_tsubj, prem_p.strip(' '), prem_tobj])
            hyp = ', '.join([hyp_tsubj, hyp_p.strip(' '), hyp_tobj])

            prem_hyps.append((prem, hyp, label_n, aligned_flag))
    return prem_hyps


def negate(verb_phrase: str) -> str:
    tokens = re.split(r'\s+', verb_phrase)
    if tokens[0] in ['is', 'are', 'were', 'was']:
        new_tokens = tokens[:1] + ['not'] + tokens[1:]
    else:
        if tokens[0].endswith('s'):
            new_tokens = ['does', 'not', tokens[0][:-1]] + tokens[1:]
        else:
            new_tokens = ['do', 'not', tokens[0][:-1]] + tokens[1:]
    return ' '.join(new_tokens)


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


def acquire_in_context_examples(tplt_fmt: str, use_plhr: str, do_neg: bool, use_binary_options: bool,
                                is_single_statement: bool):
    if use_binary_options is True:
        A_option = "A) True"
        B_option = "B) False"
        C_option = None
        if is_single_statement:
            option_str = KNOWLEDGE_OPTION_STR_BINARY
        else:
            option_str = INFERENCE_OPTION_STR_BINARY
    else:
        A_option = "A) True" if is_single_statement else "A) Entailment"
        B_option = "B) Unknown" if is_single_statement else "B) Neutral"
        C_option = "C) False" if is_single_statement else "C) Contradiction"
        if is_single_statement:
            option_str = KNOWLEDGE_OPTION_STR_TRINARY
        else:
            option_str = INFERENCE_OPTION_STR_TRINARY

    if not is_single_statement:
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

        if use_plhr in ['original', 'randprem_orig', 'randhyp_orig', 'rp_original_low', 'rp_original_high',
                        'rp_original_draw', 'rp_original_none']:
            exmpl1_psubj = 'Google'
            exmpl1_pobj = 'Youtube'
            exmpl1_aligned = True
            exmpl2_psubj = 'John'
            exmpl2_pobj = 'the mall'
            exmpl2_aligned = True
        elif use_plhr in ['type', 'rp_type_low', 'rp_type_high', 'rp_type_draw', 'rp_type_none', 'randprem_type',
                          'randhyp_type']:
            exmpl1_psubj = 'organization x'
            exmpl1_pobj = 'organization y'
            exmpl1_aligned = True
            exmpl2_psubj = 'person x'
            exmpl2_pobj = 'location y'
            exmpl2_aligned = True
        elif use_plhr in ['shuffled', 'lowfreq', 'highfreq']:
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
        context_cot = f"""{format_proppairs_with_template(tplt_fmt, exmpl_p1_pred, exmpl_h1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {A_option}. Owning is a consequence of buying.

{format_proppairs_with_template(tplt_fmt, exmpl_h1_pred, exmpl_p1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {B_option}. Owning does not imply buying, the ownership may come from other means.

{format_proppairs_with_template(tplt_fmt, exmpl_h2_pred, exmpl_p2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {B_option}. {exmpl2_psubj} may have gone to the mall by other means.

{format_proppairs_with_template(tplt_fmt, exmpl_p2_pred, exmpl_h2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {A_option}. Driving is a means of going to the mall.\n\n"""

        context_lblonly = f"""{format_proppairs_with_template(tplt_fmt, exmpl_p1_pred, exmpl_h1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {A_option}.

{format_proppairs_with_template(tplt_fmt, exmpl_h1_pred, exmpl_p1_pred, exmpl1_psubj, exmpl1_pobj, exmpl1_aligned)}{option_str} {B_option}.

{format_proppairs_with_template(tplt_fmt, exmpl_h2_pred, exmpl_p2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {B_option}.

{format_proppairs_with_template(tplt_fmt, exmpl_p2_pred, exmpl_h2_pred, exmpl2_psubj, exmpl2_pobj, exmpl2_aligned)}{option_str} {A_option}.\n\n"""
    else:
        # assert use_plhr == 'original'
        context_cot = None
        context_lblonly = f"""Google bought Youtube.{option_str} {A_option}.

Yoshua Bengio likes oak trees.{option_str} {B_option}.

The sun rises from the west.{option_str} {C_option}.\n\n"""
    return context_cot, context_lblonly


def get_gpt_template(p: str, h: str, aligned: bool, use_plhr: str, in_context: str, tplt_fmt: str, do_neg: bool,
                     use_binary_options: bool, single_statement: str, rev_hyp_args = False,
                     has_instruction: bool = False) -> str:
    """
    Get the template for the premise and hypothesis pair. The template is borrowed from GPT-3.
    :param tplt_idx:
    :param in_context:
    :param use_plhr:
    :param p:
    :param h:
    :return:
    """
    if single_statement == 'h':
        assert h is not None and p is None
    elif single_statement == 'p':
        assert h is None and p is not None
    elif single_statement is None:
        assert h is not None and p is not None
    else:
        raise ValueError("Unknown single_statement value: {}".format(single_statement))
    if not has_instruction:
        instruction = ""
    elif single_statement in ['h', 'p']:
        instruction = "Please determine the correctness of the following statement.\n\n"
    else:
        instruction = "Please check the entailments between the following statements.\n\n"  # hypothetical; Ignore the veracity of these statements.

    context_cot, context_lblonly = acquire_in_context_examples(tplt_fmt, use_plhr, do_neg, use_binary_options,
                                                               is_single_statement=(single_statement is not None))

    def clean_sentence(sent: str, role: str):
        subj, pred, obj = sent.lower().split(',')
        if use_plhr in ['xy']:
            subj = 'X'
            obj = 'Y'
        elif use_plhr in ['original', 'type', 'shuffled', 'random', 'lowfreq', 'highfreq', 'randprem_orig', 'randprem_type',
                          'randhyp_orig', 'randhyp_type', 'rp_original_low', 'rp_original_high', 'rp_original_draw',
                          'rp_original_none', 'rp_type_low', 'rp_type_high', 'rp_type_draw', 'rp_type_none', 'ant']:
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
    if use_binary_options and single_statement is not None:
        option_str = KNOWLEDGE_OPTION_STR_BINARY
    elif use_binary_options and single_statement is None:
        option_str = INFERENCE_OPTION_STR_BINARY
    elif not use_binary_options and single_statement is not None:
        option_str = KNOWLEDGE_OPTION_STR_TRINARY
    elif not use_binary_options and single_statement is None:
        option_str = INFERENCE_OPTION_STR_TRINARY
    else:
        raise ValueError(f"Unknown combination of use_binary_options and single_statement: {use_binary_options}, {single_statement}")
    template = f"""{tplt_fmt.format_map(sent_args)}{option_str}"""
    if in_context == 'cot':
        template = instruction + context_cot + template
    elif in_context == 'lbl':
        template = instruction + context_lblonly + template
    elif in_context == 'none':
        template = instruction + template
    else:
        raise ValueError(f"Unknown in_context value: {in_context}")
    return template


def f_beta_from_prec_rec(prec, rec, beta=1):
    return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)


def find_best_f_beta_from_curve(precisions, recalls, beta=1, epsilon=0.000001):
    best_f_beta = 0
    best_prec = 0
    best_rec = 0
    assert len(precisions) == len(recalls)
    for prec, rec in zip(precisions, recalls):
        f_beta = f_beta_from_prec_rec(max(prec, 0.000001), max(rec, 0.000001), beta)
        if f_beta > best_f_beta + epsilon:
            best_f_beta = f_beta
            best_prec = prec
            best_rec = rec
    return best_f_beta, best_prec, best_rec


def get_auc_norm_from_prec_recs(precs, recs, baseline_prec):
    prec_norm = [max(x-baseline_prec, 0) for x in precs]
    auc_norm = auc(recs, prec_norm)
    auc_norm /= (1-baseline_prec)
    return auc_norm


def print_metrics(golds, scores, legend_str: str, beta=1):
    print(f"Metrics for {legend_str}:")
    print(f"Positive predictions: {sum([(x > 0.5) for x in scores])} / {len(scores)}")
    pred_gold_diff_cnt = sum([(x > 0.5) != y for (x, y) in zip(scores, golds)])
    pred_gold_same_cnt = len(scores) - pred_gold_diff_cnt
    print(f"pred-gold diff: {pred_gold_diff_cnt} / {len(scores)}")
    print(f"pred-gold same: {pred_gold_same_cnt} / {len(scores)}")
    precisions, recalls, thresholds = precision_recall_curve(golds, scores)
    pr_auc_score = auc(recalls, precisions)
    average_prec = average_precision_score(golds, scores)
    bsln_precision = sum(golds) / len(golds)
    auc_norm = get_auc_norm_from_prec_recs(precisions, recalls, bsln_precision)
    best_f, best_p, best_r = find_best_f_beta_from_curve(precisions, recalls, beta=beta)
    best_f1, best_pf1, best_rf1 = find_best_f_beta_from_curve(precisions, recalls, beta=1)
    plt.plot(recalls, precisions, label=f'PR curve - {legend_str}')

    fpr, tpr, thresholds = roc_curve(golds, scores)
    roc_auc = roc_auc_score(golds, scores)
    plt.plot(fpr, tpr, label=f'ROC curve - {legend_str}')
    bin_prec, bin_rec, bin_fscore, _ = precision_recall_fscore_support(golds, [x > 0.5 for x in scores], beta=beta,
                                                                       average='binary')
    print(f"Binary F{beta}-score: {bin_fscore:.4f}; Binary precision: {bin_prec:.4f}; Binary recall: {bin_rec:.4f};")
    print(f"Best F-1: {best_f1:.4f}; Best precision: {best_pf1:.4f}; Best recall: {best_rf1:.4f};")
    print(f"Best F-{beta}: {best_f:.4f}; Best precision: {best_p:.4f}; Best recall: {best_r:.4f};")
    print(f"PR AUC: {pr_auc_score:.4f}; Average precision: {average_prec:.4f}; AUC norm: {auc_norm:.4f}; baseline precision: {bsln_precision:.4f};")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"number of entries: {len(golds)}")
    print(f"")


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


def get_freq_halves(freqs_in_dict):
    hyp_freqs = []
    for item in freqs_in_dict:
        curr_hfreqs = item['hyp_pred_freq']
        assert len(curr_hfreqs) == 220
        curr_hfreqs = curr_hfreqs[150:]  # this is from 1950-2019
        curr_avg_hfreq = sum(curr_hfreqs) / len(curr_hfreqs)
        hyp_freqs.append(curr_avg_hfreq)

    hyp_freq_threshold = heapq.nlargest(len(hyp_freqs)//2, hyp_freqs)[-1]
    hyp_halves_preds = []
    for hf in hyp_freqs:
        if hf >= hyp_freq_threshold:
            hyp_halves_preds.append(True)
        else:
            hyp_halves_preds.append(False)
    return hyp_halves_preds


def wrap_prompt_completion(prompt, model_name: str = "text-davinci-003", max_tokens: int = 32, temperature: float = 0.0,
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


def wrap_prompt_chat(prompt, model_name: str = "text-davinci-003", max_tokens: int = 32, temperature: float = 0.0,
                top_p: float = 1.0):
    messages = [
        {'role': 'system', 'content': 'You are a helpful and concise assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    ret_dict = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return ret_dict