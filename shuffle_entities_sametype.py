import json
import argparse
import random
import sys
import os


def add_entity_to_pool(entity_type, entity_name, pool):
    # Duplications are considered here, so that when we sample from the pool, more frequent entities will be sampled more often.
    if entity_type not in pool:
        pool[entity_type] = []
    pool[entity_type].append(entity_name)
    return


def build_entity_pool(args) -> dict:
    entity_pool = {}
    rel_fp = open(f'./levyholt_files/{args.subset}_files/with_type/{args.split}_rels.txt', 'r', encoding='utf8')
    text_fp = open(f'./levyholt_files/{args.subset}_files/with_original/{args.split}_ordered.txt', 'r', encoding='utf8')
    for r_line, t_line in zip(rel_fp, text_fp):
        if len(r_line) < 2:
            continue
        hyp_r, prem_r, label_r = r_line.rstrip().split('\t')
        hyp_t, prem_t, label_t, _ = t_line.rstrip().split('\t')
        try:
            hyp_pred, hyp_r_subj, hyp_r_obj = hyp_r.split(' ')
            hyp_tsubj = hyp_r_subj.split('::')[1]
            hyp_tobj = hyp_r_obj.split('::')[1]
            hyp_textual_subj, hyp_textual_pred, hyp_textual_obj = hyp_t.split(',')
            add_entity_to_pool(hyp_tsubj, hyp_textual_subj, entity_pool)
            add_entity_to_pool(hyp_tobj, hyp_textual_obj, entity_pool)
        except ValueError as e:
            print(e)
            print(f"{hyp_r}; {hyp_t}")
            pass
        try:
            prem_pred, prem_r_subj, prem_r_obj = prem_r.split(' ')
            prem_tsubj = prem_r_subj.split('::')[1]
            prem_tobj = prem_r_obj.split('::')[1]
            prem_textual_subj, prem_textual_pred, prem_textual_obj = prem_t.split(',')
            add_entity_to_pool(prem_tsubj, prem_textual_subj, entity_pool)
            add_entity_to_pool(prem_tobj, prem_textual_obj, entity_pool)
        except ValueError as e:
            print(e)
            print(f"{prem_r}; {prem_t}")
            pass
    rel_fp.close()
    text_fp.close()

    entity_pool = {k: v for k, v in sorted(entity_pool.items(), key=lambda item: item[0])}  # sort by key
    print(f"Unique entity types: {len(entity_pool)}")
    print(f"Number of unique entities per type:")
    for _type in entity_pool:
        print(f"{_type}: {len(set(entity_pool[_type]))}")

    return entity_pool


def get_text_line_dict(args):
    tfp = open(f'./levyholt_files/{args.subset}_files/with_original/{args.split}.txt', 'r', encoding='utf8')
    text_lines = []
    for line in tfp:
        if len(line) < 2:
            continue
        hyp, prm, label, lang = line.strip().split('\t')
        entry = {
            'hyp': hyp,
            'prm': prm,
            'label': label,
            'lang': lang,
            'hyp_subj': hyp.split(',')[0].strip(' ').lower(),
            'hyp_pred': hyp.split(',')[1].strip(' ').lower(),
            'hyp_obj': hyp.split(',')[2].strip(' ').lower(),
            'prm_subj': prm.split(',')[0].strip(' ').lower(),
            'prm_pred': prm.split(',')[1].strip(' ').lower(),
            'prm_obj': prm.split(',')[2].strip(' ').lower(),
        }
        text_lines.append(entry)
    return text_lines


def get_types_from_relpair(hyp_r, prem_r, label_r):
    try:
        hyp_pred, hyp_subj, hyp_obj = hyp_r.split(' ')
        hyp_subj_type, hyp_obj_type = hyp_subj.split('::')[1], hyp_obj.split('::')[1]
    except ValueError as e:
        hyp_subj_type, hyp_obj_type = None, None

    try:
        prem_pred, prem_subj, prem_obj = prem_r.split(' ')
        prem_subj_type, prem_obj_type = prem_subj.split('::')[1], prem_obj.split('::')[1]
    except ValueError as e:
        prem_subj_type, prem_obj_type = None, None

    if (hyp_subj_type == prem_subj_type) and (hyp_obj_type == prem_obj_type):
        pass
    elif (hyp_subj_type == prem_obj_type) and (hyp_obj_type == prem_subj_type):
        pass
    elif hyp_subj_type is None or hyp_obj_type is None or prem_subj_type is None or prem_obj_type is None:
        pass
    else:
        # print('Error: the premise and the hypothesis are not Nones, and not of the same types:', file=sys.stderr)
        # print(hyp_r, prem_r, label_r, file=sys.stderr)
        if hyp_subj_type == prem_subj_type and hyp_obj_type != prem_obj_type:
            if hyp_obj_type == 'thing':
                hyp_obj_type = prem_obj_type
            else:
                prem_obj_type = hyp_obj_type
        elif hyp_subj_type == prem_obj_type and hyp_obj_type != prem_subj_type:
            if hyp_obj_type == 'thing':
                hyp_obj_type = prem_subj_type
            else:
                prem_subj_type = hyp_obj_type
        elif hyp_obj_type == prem_subj_type and hyp_subj_type != prem_obj_type:
            if hyp_subj_type == 'thing':
                hyp_subj_type = prem_obj_type
            else:
                prem_obj_type = hyp_subj_type
        elif hyp_obj_type == prem_obj_type and hyp_subj_type != prem_subj_type:
            if hyp_subj_type == 'thing':
                hyp_subj_type = prem_subj_type
            else:
                prem_subj_type = hyp_subj_type
        # print(f"We take the non-thing types as gold, otherwise we take the hypothesis types as gold.", file=sys.stderr)
    return hyp_subj_type, hyp_obj_type, prem_subj_type, prem_obj_type


def fuzzy_match(a, b):
    if a in b or b in a:
        return True
    a = a.split(' ')
    b = b.split(' ')
    a = [x for x in a if x not in ['the', 'a', 'an']]
    b = [x for x in b if x not in ['the', 'a', 'an']]
    ainb_flag = True
    for x in a:
        if x in b or x.rstrip('s') in b or x.rstrip('es') in b:
            continue
        else:
            ainb_flag = False
            break
    if ainb_flag:
        return True
    bina_flag = True
    for x in b:
        if x in a or x.rstrip('s') in a or x.rstrip('es') in a:
            continue
        else:
            bina_flag = False
            break
    if bina_flag:
        return True
    return False


def match_typed_entry_to_raw_entry(query_hyp, query_prem, query_label, raw_entries):
    query_hyp_subj, query_hyp_pred, query_hyp_obj = query_hyp.split(',')
    query_prem_subj, query_prem_pred, query_prem_obj = query_prem.split(',')
    query_hyp_subj = query_hyp_subj.strip(' ').lower()
    query_hyp_pred = query_hyp_pred.strip(' ').lower()
    query_hyp_obj = query_hyp_obj.strip(' ').lower()
    query_prem_subj = query_prem_subj.strip(' ').lower()
    query_prem_pred = query_prem_pred.strip(' ').lower()
    query_prem_obj = query_prem_obj.strip(' ').lower()

    shortlist = []
    fuzzy_matches = []

    for entry in raw_entries:
        if entry['hyp_pred'] == query_hyp_pred and entry['prm_pred'] == query_prem_pred:
            if entry['label'] != query_label:
                # print(f"Warning: the label of the raw entry and the typed entry do not match: {entry['label']} vs. {query_label}", file=sys.stderr)
                continue
            shortlist.append(entry)
            if (entry['hyp_subj'] == query_hyp_subj and entry['prm_subj'] == query_prem_subj) or \
                    (entry['hyp_obj'] == query_hyp_obj and entry['prm_obj'] == query_prem_obj):
                return entry
            elif (fuzzy_match(entry['hyp_subj'], query_hyp_subj) and fuzzy_match(entry['prm_subj'], query_prem_subj)) or \
                    (fuzzy_match(entry['hyp_obj'], query_hyp_obj) and fuzzy_match(entry['prm_obj'], query_prem_obj)):
                fuzzy_matches.append(entry)
            else:
                pass
        else:
            pass

    # If programme ever runs through here, it means exact match was not found.
    if fuzzy_matches:
        return random.choice(fuzzy_matches)

    if shortlist:
        return random.choice(shortlist)
    else:
        raise AssertionError


def shuffle_with_other_ents(ent_type, ent_textual_name, pool):
    candidates = [x for x in pool[ent_type] if x != ent_textual_name]
    if len(candidates) == 0:
        return ent_textual_name
    else:
        return random.choice(candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='full')
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--seed', type=int, default=4242)
    args = parser.parse_args()
    random.seed(args.seed)

    entity_pool = build_entity_pool(args)

    # Reload rel_fp from the beginning, as well as the input, output files.
    rel_fp = open(f'./levyholt_files/{args.subset}_files/with_type/{args.split}_rels.txt', 'r', encoding='utf8')
    text_fp = open(f'./levyholt_files/{args.subset}_files/with_original/{args.split}_ordered.txt', 'r', encoding='utf8')
    typed_text_fp = open(f'./levyholt_files/{args.subset}_files/with_type/{args.split}.txt', 'r', encoding='utf8')
    os.makedirs(f'./levyholt_files/{args.subset}_files/with_shuffled_entities/', exist_ok=True)
    out_fp = open(f'./levyholt_files/{args.subset}_files/with_shuffled_entities/{args.split}.txt', 'w', encoding='utf8')

    shuffle_partly_failed_idxes = []
    shuffle_full_failed_idxes = []
    lidx = 0
    for lidx, (r_line, rt_line, t_line) in enumerate(zip(rel_fp, typed_text_fp, text_fp)):
        if lidx % 1000 == 0:
            print(f'Processed {lidx} lines.')
        if len(r_line) < 2:
            continue
        hyp_rel, prem_rel, label_rel = r_line.rstrip().split('\t')
        hyp_ttext, prem_ttext, label_ttext = rt_line.rstrip().split('\t')
        hyp_t, prm_t, label_t, lang = t_line.rstrip().split('\t')
        assert label_rel == label_ttext == label_t
        hyp_subj_type, hyp_obj_type, prem_subj_type, prem_obj_type = get_types_from_relpair(hyp_rel, prem_rel, label_rel)

        # We still need to check the argument alignemnts, because type alignments are not always indicative of argument alignments.
        hyp_textual_subj, hyp_textual_pred, hyp_textual_obj = hyp_t.split(',')
        prem_textual_subj, prem_textual_pred, prem_textual_obj = prm_t.split(',')
        if fuzzy_match(hyp_textual_subj.lower(), prem_textual_subj.lower()) or fuzzy_match(hyp_textual_obj.lower(), prem_textual_obj.lower()):
            align_flag = True
        elif fuzzy_match(hyp_textual_subj.lower(), prem_textual_obj.lower()) or fuzzy_match(hyp_textual_obj.lower(), prem_textual_subj.lower()):
            align_flag = False
        else:
            print('Error: the premise and the hypothesis have different arguments:', file=sys.stderr)
            print(hyp_t, prm_t, label_t, file=sys.stderr)
            print(f"We take it that they should've been aligned.", file=sys.stderr)
            align_flag = True

        # the exact same entities are avoided in the shuffled version as much as possible.
        if hyp_subj_type is not None and hyp_obj_type is not None:
            hyp_shuffled_subj = shuffle_with_other_ents(hyp_subj_type, hyp_textual_subj, entity_pool)
            hyp_shuffled_obj = shuffle_with_other_ents(hyp_obj_type, hyp_textual_obj, entity_pool)
            if align_flag is True:
                prem_shuffled_subj, prem_shuffled_obj = hyp_shuffled_subj, hyp_shuffled_obj
            else:
                assert align_flag is False
                prem_shuffled_subj, prem_shuffled_obj = hyp_shuffled_obj, hyp_shuffled_subj
        elif prem_subj_type is not None and prem_obj_type is not None:
            prem_shuffled_subj = shuffle_with_other_ents(prem_subj_type, prem_textual_subj, entity_pool)
            prem_shuffled_obj = shuffle_with_other_ents(prem_obj_type, prem_textual_obj, entity_pool)
            if align_flag is True:
                hyp_shuffled_subj, hyp_shuffled_obj = prem_shuffled_subj, prem_shuffled_obj
            else:
                assert align_flag is False
                hyp_shuffled_subj, hyp_shuffled_obj = prem_shuffled_obj, prem_shuffled_subj
        else:
            print(f"Error: both premise and hypothesis have no types assigned in REL file.", file=sys.stderr)
            print(hyp_t, prm_t, label_t, file=sys.stderr)
            hyp_shuffled_subj, hyp_shuffled_obj = hyp_textual_subj, hyp_textual_obj
            prem_shuffled_subj, prem_shuffled_obj = prem_textual_subj, prem_textual_obj

        hyp_shuffled = f'{hyp_shuffled_subj},{hyp_textual_pred},{hyp_shuffled_obj}'
        prem_shuffled = f'{prem_shuffled_subj},{prem_textual_pred},{prem_shuffled_obj}'
        if hyp_shuffled_subj == hyp_textual_subj or hyp_shuffled_obj == hyp_textual_obj or \
                prem_shuffled_subj == prem_textual_subj or prem_shuffled_obj == prem_textual_obj:
            shuffle_partly_failed_idxes.append(lidx)
        if hyp_shuffled == hyp_t or prem_shuffled == prm_t:
           shuffle_full_failed_idxes.append(lidx)

        out_fp.write(f'{hyp_shuffled}\t{prem_shuffled}\t{label_t}\t{lang}\n')

    print(f"Shuffle partly failed for {len(shuffle_partly_failed_idxes)} / {lidx} lines.", file=sys.stderr)
    print(f"Shuffle full failed for {len(shuffle_full_failed_idxes)} / {lidx} lines.", file=sys.stderr)
    print(f"lidxes where shuffle partly failed:", file=sys.stderr)
    print(shuffle_partly_failed_idxes, file=sys.stderr)
    print(f"lidxes where shuffle full failed:", file=sys.stderr)
    print(shuffle_full_failed_idxes, file=sys.stderr)
