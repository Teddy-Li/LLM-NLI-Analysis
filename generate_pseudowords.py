import sys
import wuggy
from random_word import RandomWords
import json
import random
import os
import argparse
from nltk.corpus import wordnet as wn
from utils import load_typed_entries_for_pseudowords


def get_pseudos_from_phrase(phrase, g, r, ncandidates_per_sequence=1, max_search_time_per_sequence=20):
    if phrase is None:
        for i in range(5):
            w = r.get_random_word()
            try:
                matches = g.generate_classic(input_sequences=[w.lower()], ncandidates_per_sequence=50,
                                             max_search_time_per_sequence=max_search_time_per_sequence,
                                             match_subsyllabic_segment_length=False,
                                             match_letter_length=False, output_mode='plain')
                candidate_pseudo = matches[0]['pseudoword']
                return candidate_pseudo, True
            except Exception as ee:
                continue
        return 'Caplus', True

    phrase_list = phrase.split(' ')
    pseudo_phrase_list = []
    miss_flag = False
    for word in phrase_list:
        if word.lower() != word:
            capitalized = True
        else:
            capitalized = False
        word = word.lower()
        if word in ['a', 'an', 'the']:
            pseudo_phrase_list.append(word.capitalize() if capitalized else word)
        else:
            try:
                matches = g.generate_classic(input_sequences=[word], ncandidates_per_sequence=ncandidates_per_sequence,
                                             max_search_time_per_sequence=max_search_time_per_sequence,
                                             match_subsyllabic_segment_length=False,
                                             match_letter_length=False, output_mode='plain')
                candidate_pseudo = matches[0]['pseudoword']
                pseudo_phrase_list.append(candidate_pseudo.capitalize() if capitalized else candidate_pseudo)
            except Exception as e:
                print(e, file=sys.stderr)
                miss_flag = True

                replaced_flag = False
                for i in range(5):
                    w = r.get_random_word()
                    try:
                        matches = g.generate_classic(input_sequences=[w.lower()], ncandidates_per_sequence=ncandidates_per_sequence,
                                                     max_search_time_per_sequence=max_search_time_per_sequence,
                                                     match_subsyllabic_segment_length=False,
                                                     match_letter_length=False, output_mode='plain')
                        candidate_pseudo = matches[0]['pseudoword']
                        pseudo_phrase_list.append(candidate_pseudo.capitalize() if capitalized else candidate_pseudo)
                        replaced_flag = True
                        break
                    except Exception as ee:
                        continue
                if not replaced_flag:
                    matches = g.generate_classic(input_sequences=['sequence', 'input', 'generate'], ncandidates_per_sequence=50,
                                                 max_search_time_per_sequence=max_search_time_per_sequence,
                                                 match_subsyllabic_segment_length=False,
                                                 match_letter_length=False, output_mode='plain')
                    match = random.choice(matches)
                    pseudo_phrase_list.append(match['pseudoword'].capitalize() if capitalized else match['pseudoword'])

    return ' '.join(pseudo_phrase_list), miss_flag


def main(args):
    g = wuggy.WuggyGenerator()
    g.load('orthographic_english')
    r = RandomWords()
    total_miss = 0
    total = 0
    data_dirname = f"{args.subset}_files"

    os.makedirs(os.path.join(args.data_root, data_dirname, 'with_pseudoents'), exist_ok=True)
    for split in ['dev', 'test']:
        if args.subset == 'dir':
            split += '_dir'
        else:
            assert args.subset == 'full'
            pass
        in_fpath = os.path.join(args.data_root, data_dirname, 'with_type', split + '%s.txt')
        out_fpath = os.path.join(args.data_root, data_dirname, 'with_pseudoents', split + '.txt')
        prem_hyps = load_typed_entries_for_pseudowords(in_fpath)
        with open(out_fpath, 'w', encoding='utf8') as ofp:
            for lidx, (prem_p, prem_subj, prem_obj, prem_tsubj, prem_tobj, hyp_p, label, aligned_flag) in enumerate(prem_hyps):
                if lidx % 40 == 0:
                    print(f'Processing line {lidx}; total miss: {total_miss}; total: {total}')

                pseudo_subj, miss_flag = get_pseudos_from_phrase(prem_subj, g, r, args.ncandidates_per_sequence,
                                                           args.max_search_time_per_sequence)
                if miss_flag:
                    total_miss += 1
                total += 1

                pseudo_obj, miss_flag = get_pseudos_from_phrase(prem_obj, g, r, args.ncandidates_per_sequence,
                                                          args.max_search_time_per_sequence)
                if miss_flag:
                    total_miss += 1
                total += 1
                prem_pseudo = pseudo_subj + ',' + prem_p + ',' + pseudo_obj
                if aligned_flag:
                    hyp_pseudo = pseudo_subj + ',' + hyp_p + ',' + pseudo_obj
                else:
                    hyp_pseudo = pseudo_obj + ',' + hyp_p + ',' + pseudo_subj

                out_item = {
                    'prem': prem_pseudo,
                    'hyp': hyp_pseudo,
                    'label': label,
                    'aligned': aligned_flag,
                    'prem_tsubj': prem_tsubj,
                    'prem_tobj': prem_tobj
                }

                ofp.write(json.dumps(out_item) + '\n')


def playground():
    g = wuggy.WuggyGenerator()
    g.load('orthographic_english')
    matches = g.generate_classic(input_sequences=['mat', 'man', 'healthy'], ncandidates_per_sequence=3,
                                 max_search_time_per_sequence=20, match_subsyllabic_segment_length=False,
                                 match_letter_length=False, output_mode='plain')
    for match in matches:
        print(match['statistics']['lexicality'], match['pseudoword'])
    match = random.choice(matches)
    print(f"chosen match: {match}")
    synsets = wn.synsets('a')
    print(synsets)
    synsets = wn.synsets('the')
    print(synsets)
    synsets = wn.synsets('cat')
    print(synsets)
    synsets = wn.synsets('man')
    print(synsets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='playground')
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--subset', type=str, default='full')
    parser.add_argument('--ncandidates_per_sequence', type=int, default=1)
    parser.add_argument('--max_search_time_per_sequence', type=int, default=20)
    args = parser.parse_args()

    if args.task == 'playground':
        playground()
    elif args.task == 'convert':
        main(args)
    else:
        raise ValueError('Unknown task: {}'.format(args.task))
