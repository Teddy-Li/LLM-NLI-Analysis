import json
import argparse
from utils import print_metrics, phi_coefficient
from statsmodels.stats.inter_rater import fleiss_kappa


def my_aggregate_raters(raters):
    res = []
    num_entries = len(raters[0])
    for i in range(num_entries):
        curr_res = [raters[j][i] for j in range(len(raters))]
        posis = sum(curr_res)
        res.append([3-posis, posis])
    return res


parser = argparse.ArgumentParser()
parser.add_argument('--use_plhr', type=str, default='original')
parser.add_argument('--llama_model_name', type=str, default='llama-65b-hf')

args = parser.parse_args()

gpt_honly_res_path = f'./results/levyholt_results/gpt_results/gpt3_text-davinci-003_res_dir_text_test_{args.use_plhr}_icl=lbl_trinary_1_hyponly.json'
llama_honly_res_path = f'./results/levyholt_results/llama_results/llama_{args.llama_model_name}_res_dir_text_test_{args.use_plhr}_icl=lbl_hypOnly.json'

with open(gpt_honly_res_path, 'r', encoding='utf-8') as f:
    gpt_honly_binaries = []
    gpt_honly_scores = []
    gpt_honly_gold_labels = []
    for line in f:
        item = json.loads(line)
        assert len(item['preds']) == 1
        if item['preds'][0] > 0.5:
            gpt_honly_binaries.append(True)
        else:
            gpt_honly_binaries.append(False)
        gpt_honly_scores.append(item['preds'][0])
        gpt_honly_gold_labels.append(item['gold'])

with open(llama_honly_res_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    assert isinstance(data, dict)
    llama_honly_predictions = data['predictions']
    llama_honly_scores = data['scores']
    assert len(llama_honly_predictions) == 1 and len(llama_honly_scores) == 1
    llama_honly_predictions = llama_honly_predictions[0]
    llama_honly_scores = llama_honly_scores[0]
    llama_honly_binaries = []
    for pred in llama_honly_predictions:
        assert pred in ['A', 'B', 'C']
        if pred == 'A':
            llama_honly_binaries.append(True)
        else:
            llama_honly_binaries.append(False)

phi_gpt_llama_honly = phi_coefficient(gpt_honly_binaries, llama_honly_binaries)
print(f'phi_gpt_llama_honly: {phi_gpt_llama_honly}')

agg_for_fleiss = my_aggregate_raters([gpt_honly_binaries, llama_honly_binaries])
kappa_honly = fleiss_kappa(agg_for_fleiss)

print(f'kappa_honly: {kappa_honly}')

intersect_honly_binaries = []
union_honly_binaries = []
vote_honly_binaries = []
for i in range(len(gpt_honly_binaries)):
    intersect_honly_binaries.append(gpt_honly_binaries[i] and llama_honly_binaries[i])
    union_honly_binaries.append(gpt_honly_binaries[i] or llama_honly_binaries[i])
    vote_honly_binaries.append(sum([gpt_honly_binaries[i], llama_honly_binaries[i]]) >= 1)

strict_honly_binaries = []
for i in range(len(gpt_honly_binaries)):
    if intersect_honly_binaries[i] is True:
        strict_honly_binaries.append(True)
    elif union_honly_binaries[i] is False:
        strict_honly_binaries.append(False)
    else:
        strict_honly_binaries.append(None)

phi_intersect_honly = phi_coefficient(intersect_honly_binaries, gpt_honly_gold_labels)
phi_union_honly = phi_coefficient(union_honly_binaries, gpt_honly_gold_labels)
print(f"phi_intersect_honly: {phi_intersect_honly}")
print(f"phi_union_honly: {phi_union_honly}")

print(f"intersect honly positives / negatives: {sum(intersect_honly_binaries)} / {len(intersect_honly_binaries)-sum(intersect_honly_binaries)}")
print(f"union honly positives / negatives: {sum(union_honly_binaries)} / {len(union_honly_binaries)-sum(union_honly_binaries)}")
print(f"vote honly positives / negatives: {sum(vote_honly_binaries)} / {len(vote_honly_binaries)-sum(vote_honly_binaries)}")

with open(f'./results/levyholt_results/polled_honly/llama_gpt_intersection_honly_{args.use_plhr}_binaries.json', 'w', encoding='utf-8') as f:
    json.dump(intersect_honly_binaries, f, indent=4)

with open(f'./results/levyholt_results/polled_honly/llama_gpt_union_honly_{args.use_plhr}_binaries.json', 'w', encoding='utf-8') as f:
    json.dump(union_honly_binaries, f, indent=4)

with open(f'./results/levyholt_results/polled_honly/llama_gpt_strict_honly_{args.use_plhr}_binaries.json', 'w', encoding='utf-8') as f:
    json.dump(strict_honly_binaries, f, indent=4)

with open(f'./results/levyholt_results/polled_honly/llama_gpt_vote_honly_{args.use_plhr}_binaries.json', 'w', encoding='utf-8') as f:
    json.dump(vote_honly_binaries, f, indent=4)


