import json
from collections import defaultdict


def get_rel_lines(path_n, path_r):
    ret_lines = []
    with open(path_n, 'r', encoding='utf8') as nfp, \
            open(path_r, 'r', encoding='utf8') as rfp:
        for n_line, r_line in zip(nfp, rfp):
            if len(n_line) < 2 or len(r_line) < 2:
                assert len(n_line) < 2 and len(r_line) < 2
                continue
            nlst = n_line.split('\t')
            if len(nlst) == 3:
                n_hyp, n_prm, n_lbl = nlst
            elif len(nlst) == 4:
                n_hyp, n_prm, n_lbl, _ = nlst
            else:
                raise ValueError
            h_subj_n, h_pred_n, h_obj_n = n_hyp.split(',')
            p_subj_n, p_pred_n, p_obj_n = n_prm.split(',')

            r_hyp, r_prm, r_lbl = r_line.split('\t')
            try:
                h_pred_r, h_subj_r, h_obj_r = r_hyp.split(' ')
                h_tsubj_r = h_subj_r.split('::')[1]
                h_tobj_r = h_obj_r.split('::')[1]
                hyp_ccg_arg_positions = [int(c) for c in h_pred_r if c.isdigit()]
                if h_pred_n.endswith(' by') and '.by' not in h_pred_r and hyp_ccg_arg_positions and \
                        hyp_ccg_arg_positions[0] == 1:
                    h_pred_flipped = True
                else:
                    h_pred_flipped = False
            except ValueError:
                h_pred_r, h_subj_r, h_obj_r, h_tsubj_r, h_tobj_r = None, None, None, None, None
                h_pred_flipped = None

            try:
                p_pred_r, p_subj_r, p_obj_r = r_prm.split(' ')
                p_tsubj_r = p_subj_r.split('::')[1]
                p_tobj_r = p_obj_r.split('::')[1]
                prem_ccg_arg_positions = [int(c) for c in p_pred_r if c.isdigit()]
                if p_pred_n.endswith(' by') and '.by' not in p_pred_r and prem_ccg_arg_positions and \
                        prem_ccg_arg_positions[0] == 1:
                    p_pred_flipped = True
                else:
                    p_pred_flipped = False
            except ValueError:
                p_pred_r, p_subj_r, p_obj_r, p_tsubj_r, p_tobj_r = None, None, None, None, None
                p_pred_flipped = None

            ret_lines.append({
                'h_pred_r': h_pred_r,
                'h_pred_n': h_pred_n,
                'p_pred_r': p_pred_r,
                'p_pred_n': p_pred_n,
                'h_tsubj_r': h_tsubj_r,
                'h_tobj_r': h_tobj_r,
                'p_tsubj_r': p_tsubj_r,
                'p_tobj_r': p_tobj_r,
                'h_subj_r': h_subj_r.split('::')[0] if h_subj_r is not None else None,
                'h_obj_r': h_obj_r.split('::')[0] if h_obj_r is not None else None,
                'h_flipped': h_pred_flipped,
                'p_flipped': p_pred_flipped,
            })
    return ret_lines


input_lines = []
with open('./test_randprem.txt', 'r', encoding='utf8') as ifp:
    for iline in ifp:
        if len(iline) < 2:
            continue
        i_hyp, i_prm, i_lbl = iline.strip().split('\t')
        input_lines.append((i_hyp, i_prm, i_lbl))

source_lines_d = get_rel_lines('../../MoNTEE_Levy_Holt/full/dev_s2.txt',
                             '../../MoNTEE_Levy_Holt/full/dev_rels.txt')
source_lines_t = get_rel_lines('../../MoNTEE_Levy_Holt/full/test_s2.txt',
                               '../../MoNTEE_Levy_Holt/full/test_rels.txt')
source_lines = source_lines_d + source_lines_t
ref_lines = get_rel_lines('../with_type/test_ordered.txt',
                          '../with_type/test_rels.txt')

ofp = open('./test_randprem_rels.txt', 'w', encoding='utf8')

for lidx, line in enumerate(input_lines):
    i_hyp, i_prm, i_lbl = line
    ih_subj_n, ih_pred_n, ih_obj_n = i_hyp.split(',')
    h_flipped_flag = ref_lines[lidx]['h_flipped']
    h_pred_r = ref_lines[lidx]['h_pred_r']
    h_pred_n = ref_lines[lidx]['h_pred_n']
    h_subj_r = ref_lines[lidx]['h_subj_r']
    h_obj_r = ref_lines[lidx]['h_obj_r']
    h_tsubj_r = ref_lines[lidx]['h_tsubj_r']
    h_tobj_r = ref_lines[lidx]['h_tobj_r']
    assert ih_pred_n.strip(' ') == h_pred_n.strip(' '), f"{ih_pred_n} != {h_pred_n}"

    ip_subj_n, ip_pred_n, ip_obj_n = i_prm.split(',')
    ip_pred_r = None
    ip_flipped = None
    ip_unparsed_found = False
    for s_entry in source_lines:
        if s_entry['h_pred_n'].strip(' ') == ip_pred_n.strip(' '):
            if s_entry['h_pred_r'] is None:
                ip_unparsed_found = True
                pass
            # if {s_entry['h_tsubj_r'], s_entry['h_tobj_r']}.difference({h_tsubj_r, h_tobj_r, 'thing'}) and \
            #         {h_tsubj_r, h_tobj_r}.difference({s_entry['h_tsubj_r'], s_entry['h_tobj_r'], 'thing'}):
            #     print('mismatched types')
            #     print(s_entry['h_tsubj_r'], s_entry['h_tobj_r'])
            #     print(h_tsubj_r, h_tobj_r)
            #     pass
            # else:
            ip_pred_r = s_entry['h_pred_r']
            ip_flipped = s_entry['h_flipped']
            break
        elif s_entry['p_pred_n'].strip(' ') == ip_pred_n.strip(' '):
            if s_entry['p_pred_r'] is None:
                ip_unparsed_found = True
                pass
            # if {s_entry['p_tsubj_r'], s_entry['p_tobj_r']}.difference({h_tsubj_r, h_tobj_r, 'thing'}) and \
            #         {h_tsubj_r, h_tobj_r}.difference({s_entry['p_tsubj_r'], s_entry['p_tobj_r'], 'thing'}):
            #     print('mismatched types')
            #     print(s_entry['p_tsubj_r'], s_entry['p_tobj_r'])
            #     print(h_tsubj_r, h_tobj_r)
            #     pass
            # else:
            ip_pred_r = s_entry['p_pred_r']
            ip_flipped = s_entry['p_flipped']
            break
        else:
            pass
    assert ip_flipped is not None or ip_unparsed_found, f"{ip_pred_n} not found"

    if h_flipped_flag != ip_flipped:
        p_subj_r = h_obj_r
        p_obj_r = h_subj_r
        p_tsubj_r = h_tsubj_r
        p_tobj_r = h_tobj_r
    else:
        p_subj_r = h_subj_r
        p_obj_r = h_obj_r
        p_tsubj_r = h_tsubj_r
        p_tobj_r = h_tobj_r
    if ip_unparsed_found and ip_flipped is None:
        print(f"unparsed: {ip_pred_n} {ip_subj_n}::{ip_obj_n}")
        r_outline = f"{h_pred_r} {h_subj_r}::{h_tsubj_r} {h_obj_r}::{h_tobj_r}\t\t{i_lbl}\n"
    else:
        r_outline = f"{h_pred_r} {h_subj_r}::{h_tsubj_r} {h_obj_r}::{h_tobj_r}\t{ip_pred_r} {p_subj_r}::{p_tsubj_r} {p_obj_r}::{p_tobj_r}\t{i_lbl}\n"
    ofp.write(r_outline)



