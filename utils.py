

def load_entries(dir_fn):
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


def load_typed_entries(dir_fn):
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
