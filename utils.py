import re
import json


def load_dir_entries(dir_fn):
    prem_hyp_pairs = set()
    with open(dir_fn, 'r', encoding='utf8') as dfp:
        for line in dfp:
            if len(line) < 2:
                continue
            hyp, prem, label, lang = line.rstrip().split('\t')
            if label == 'True':
                prem_hyp_pairs.add((prem, hyp))
            else:
                prem_hyp_pairs.add((hyp, prem))
    return prem_hyp_pairs


def load_general_entries(in_fn):
    # these original orders must be kept for the indices to match
    prem_hyps = []
    with open(in_fn, 'r', encoding='utf8') as fp:
        for line in fp:
            if len(line) < 2:
                continue
            hyp, prem, label, lang = line.rstrip().split('\t')
            h_subj, h_pred, h_obj = hyp.split(',')
            p_subj, p_pred, p_obj = prem.split(',')
            if h_subj == p_obj or h_obj == p_subj:
                aligned_flag = False
            else:
                aligned_flag = True
            prem_hyps.append((prem, hyp, label, aligned_flag))
    return prem_hyps


def load_typed_dir_entries(dir_fn):
    dir_nl_fn = dir_fn % ''
    dir_rels_fn = dir_fn % '_rels_v2'

    prem_hyp_pairs = set()
    with open(dir_nl_fn, 'r', encoding='utf8') as nfp, open(dir_rels_fn, 'r', encoding='utf8') as rfp:
        for n_line, r_line in zip(nfp, rfp):
            if len(n_line) < 2 or len(r_line) < 2:
                assert len(n_line) < 2 and len(r_line) < 2
                continue
            hyp_n, prem_n, label_n = n_line.rstrip().split('\t')
            hyp_r, prem_r, label_r = r_line.rstrip().split('\t')
            assert label_n == label_r
            try:
                _, hyp_subj_typed, hyp_obj_typed = hyp_r.split(' ')
                _, hyp_tsubj = hyp_subj_typed.split('::')
                _, hyp_tobj = hyp_obj_typed.split('::')
            except ValueError:
                hyp_tsubj = None
                hyp_tobj = None
            try:
                _, prem_subj_typed, prem_obj_typed = prem_r.split(' ')
                _, prem_tsubj = prem_subj_typed.split('::')
                _, prem_tobj = prem_obj_typed.split('::')
            except ValueError:
                prem_tsubj = None
                prem_tobj = None
            if hyp_tsubj is None or hyp_tobj is None:
                hyp_tsubj = prem_tsubj
                hyp_tobj = prem_tobj
            if prem_tsubj is None or prem_tobj is None:
                prem_tsubj = hyp_tsubj
                prem_tobj = hyp_tobj
            assert hyp_tsubj is not None and hyp_tobj is not None and prem_tsubj is not None and prem_tobj is not None
            prem_s, prem_p, prem_o = prem_n.split(',')
            hyp_s, hyp_p, hyp_o = hyp_n.split(',')
            prem_tsubj = prem_tsubj.replace('_', ' ')
            prem_tobj = prem_tobj.replace('_', ' ')
            hyp_tsubj = hyp_tsubj.replace('_', ' ')
            hyp_tobj = hyp_tobj.replace('_', ' ')
            prem = ', '.join([prem_tsubj, prem_p.strip(' '), prem_tobj])
            hyp = ', '.join([hyp_tsubj, hyp_p.strip(' '), hyp_tobj])

            if label_n == 'True':
                prem_hyp_pairs.add((prem, hyp))
            else:
                assert label_n == 'False'
                prem_hyp_pairs.add((hyp, prem))
    return prem_hyp_pairs


def load_typed_general_entries(in_fn):
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
                _, hyp_subj_typed, hyp_obj_typed = hyp_r.split(' ')
                hyp_subj, hyp_tsubj = hyp_subj_typed.split('::')
                hyp_obj, hyp_tobj = hyp_obj_typed.split('::')
            except ValueError:
                hyp_tsubj = None
                hyp_tobj = None
                hyp_subj = None
                hyp_obj = None
            try:
                _, prem_subj_typed, prem_obj_typed = prem_r.split(' ')
                prem_subj, prem_tsubj = prem_subj_typed.split('::')
                prem_obj, prem_tobj = prem_obj_typed.split('::')
            except ValueError:
                prem_tsubj = None
                prem_tobj = None
                prem_subj = None
                prem_obj = None

            # Figure out the aligned flag: when we know the args are not aligned, we set the flag to False;
            # Otherwise, if we know it to be aligned, or we don't know, we both set the flag to True
            if prem_subj is not None and hyp_obj is not None and prem_subj == hyp_obj:
                aligned_flag = False
            elif prem_obj is not None and hyp_subj is not None and prem_obj == hyp_subj:
                aligned_flag = False
            else:
                aligned_flag = True

            if hyp_tsubj is None or hyp_tobj is None:
                assert aligned_flag is True
                hyp_tsubj = prem_tsubj
                hyp_tobj = prem_tobj
            if prem_tsubj is None or prem_tobj is None:
                assert aligned_flag is True
                prem_tsubj = hyp_tsubj
                prem_tobj = hyp_tobj
            assert hyp_tsubj is not None and hyp_tobj is not None and prem_tsubj is not None and prem_tobj is not None
            prem_s, prem_p, prem_o = prem_n.split(',')
            hyp_s, hyp_p, hyp_o = hyp_n.split(',')
            prem_tsubj = prem_tsubj.replace('_', ' ')
            prem_tobj = prem_tobj.replace('_', ' ')
            hyp_tsubj = hyp_tsubj.replace('_', ' ')
            hyp_tobj = hyp_tobj.replace('_', ' ')

            if aligned_flag is True:
                prem_tsubj = prem_tsubj + ' X'
                prem_tobj = prem_tobj + ' Y'
                hyp_tsubj = hyp_tsubj + ' X'
                hyp_tobj = hyp_tobj + ' Y'
            else:
                prem_tsubj = prem_tsubj + ' X'
                prem_tobj = prem_tobj + ' Y'
                hyp_tsubj = hyp_tsubj + ' Y'
                hyp_tobj = hyp_tobj + ' X'

            prem = ', '.join([prem_tsubj, prem_p.strip(' '), prem_tobj])
            hyp = ', '.join([hyp_tsubj, hyp_p.strip(' '), hyp_tobj])

            prem_hyps.append((prem, hyp, label_n, aligned_flag))
    return prem_hyps


def load_typed_entries_for_pseudowords(in_fn):
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
                _, hyp_subj_typed, hyp_obj_typed = hyp_r.split(' ')
                hyp_subj, hyp_tsubj = hyp_subj_typed.split('::')
                hyp_obj, hyp_tobj = hyp_obj_typed.split('::')
            except ValueError:
                hyp_tsubj = None
                hyp_tobj = None
                hyp_subj = None
                hyp_obj = None
            try:
                _, prem_subj_typed, prem_obj_typed = prem_r.split(' ')
                prem_subj, prem_tsubj = prem_subj_typed.split('::')
                prem_obj, prem_tobj = prem_obj_typed.split('::')
            except ValueError:
                prem_tsubj = None
                prem_tobj = None
                prem_subj = None
                prem_obj = None

            # Figure out the aligned flag: when we know the args are not aligned, we set the flag to False;
            # Otherwise, if we know it to be aligned, or we don't know, we both set the flag to True
            if prem_subj is not None and hyp_obj is not None and prem_subj == hyp_obj:
                aligned_flag = False
            elif prem_obj is not None and hyp_subj is not None and prem_obj == hyp_subj:
                aligned_flag = False
            else:
                aligned_flag = True

            if hyp_tsubj is None or hyp_tobj is None:
                assert aligned_flag is True
                hyp_tsubj = prem_tsubj
                hyp_tobj = prem_tobj
            if prem_tsubj is None or prem_tobj is None:
                assert aligned_flag is True
                prem_tsubj = hyp_tsubj
                prem_tobj = hyp_tobj
            assert hyp_tsubj is not None and hyp_tobj is not None and prem_tsubj is not None and prem_tobj is not None
            prem_ns, prem_p, prem_no = prem_n.split(',')
            hyp_ns, hyp_p, hyp_no = hyp_n.split(',')
            prem_tsubj = prem_tsubj.replace('_', ' ')
            prem_tobj = prem_tobj.replace('_', ' ')
            hyp_tsubj = hyp_tsubj.replace('_', ' ')
            hyp_tobj = hyp_tobj.replace('_', ' ')
            prem_hyps.append((prem_p, prem_subj, prem_obj, prem_tsubj, prem_tobj, hyp_p, label_n, aligned_flag))
    return prem_hyps


def load_pseudoent_entries(in_fn):
    with open(in_fn, 'r', encoding='utf8') as fp:
        entries = []
        for line in fp:
            if len(line) < 2:
                continue
            dct = json.loads(line)
            entries.append(dct)
    return entries


# Borrowed from S&S
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
