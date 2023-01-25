import json
import os

for subset in ['dev', 'test']:
    full_lines = {}
    dir_idxes = []
    with open(f'./full_files/with_entities/{subset}.txt', 'r', encoding='utf8') as fp:
        for lidx, line in enumerate(fp):
            if len(line) < 2:
                raise AssertionError(f'Empty line at {lidx} in {subset}')
            full_lines[line] = lidx

    with open(f"./dir_files/with_entities/{subset}.txt", 'r', encoding='utf8') as fp:
        for line in fp:
            if line not in full_lines:
                raise AssertionError(f'{line} not in {subset}')
            dir_idxes.append(full_lines[line])

    with open(f'./dir_files/with_entities/{subset}_idxes.json', 'w', encoding='utf8') as fp:
        json.dump(dir_idxes, fp, ensure_ascii=False, indent=4)
