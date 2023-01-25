# GPT3 Experiments with Levy/Holt

## Dataset

In these experiments we use the Levy/Holt dataset. Levy/Holt dataset is a dataset focused on predicate entailment. It was
collected by converting factoid questions into triples as hypotheses, then retrieving potentially relevant propositions 
from non-fictional corpora as candidate premises (answers). The relation between these premises and hypotheses is annotated
by crowd annotators as a binary classification task.

Specifically, in [full_files](full_files) directory is the full Levy/Holt
Dataset, and in [dir_files](dir_files) directory is the directional subset of Levy/Holt Dataset, where each premise-hypothesis
pair is either a directional entailment (p entails q, q does not entail p) or a directional contradiction (p does not entail q, q entails p).

Each dataset comes in three forms: 
- `with_entities`: the original entities are used as arguments in the propositions;
- `with_types`: the original entities are replaced with their types as arguments in the propositions;
- `with_pseudoents`: the original entities are replaced with random pseudo-words generated from the original entity names.

For the full dataset, there are 5486 entries in the dev set (of which 1085 positive, 19.8%) and 12921 entries in the test set (of which 2831 positive, 21.9%);
for the directional subset, there are 630 entries in the dev set and 1784 entries in the test set, where 50% of the entries are positive.

## Running
Before running the experiments, make sure you have your virtual environment ready, and dependencies installed as specified in [requirements.txt](requirements.txt).

For running experiments with entities or types, use the following command:
```python gpt3_general_entailment.py --model_name XXX --use_plhr [none/type] --in_context [none/lbl/cot] --num_templates 4 --subset [dir/full] --split [toy/dev/test]```

Note that especially for the directional subset with `use_plhr=type`, we need to specify `--typed_in_fn ./%s_files/with_type/%s%s.txt`
to address the naming differences.

The full list of options are:

1. `--model_name` specifies the GPT-3 model to use. The default is `text-davinci-003`;
2. `--max_tokens` specifies the maximum number of tokens to output for each entry. The default is 32;
3. `--temperature` controls the temperature of generation, which is 0.7 by default;
4. `--use_plhr` specifies whether to use placeholders and how. The default is `none`, but can also be `xy` or `type`;
5. `--in_context` specifies whether to do ICL, which is `none` by default, but can also be `lbl` 
(in-context examples with labels only) or `cot` (in-context examples with chain-of-thought);
6. `--num_templates` specifies the number of active templates to try, the current set of all templates are within the 
list `sent_template_to_test` (the full set has 7 templates, usually 4 are activated since the other three are bad, 
activation is controlled by the `sent_template_activate_flags` variable in the function `retrieve_results_main`);
7. `--subset` specifies the subset to evaluate on, either `full` or `dir`. The default is `full`;
8. `--split` specifies the split to run on, either `dev` or `test`; the zero-shot results were reported on test set.

Some example scripts include:

- Dir Subset COT: ` python gpt3_general_entailment.py --model_name text-davinci-003 --use_plhr none --subset dir --split dev --debug --in_context cot --num_templates 1 `
- Dir Subset COT args=type instruct-beta: ` nohup python gpt3_general_entailment.py --model_name davinci-instruct-beta --max_tokens 8 --use_plhr type --subset dir --split dev --in_context cot --num_templates 4 --typed_in_fn ./%s_files/with_type/%s_dir%s.txt > ./logdir/davinci-instruct-beta-dir-cot-type.log & `
- Dir Subset COT args=type Code-Davinci-002: ` nohup python gpt3_general_entailment.py --model_name code-davinci-002 --max_tokens 8 --use_plhr type --subset dir --split dev --in_context cot --num_templates 4 --typed_in_fn ./%s_files/with_type/%s_dir%s.txt --sleep_after_query 30 > ./logdir/code-davinci-002-dir-cot-type.log & `

[//]: # (## Preliminary Results)

[//]: # (| GPT-3 Da Vinci    | Precision &#40;%&#41; | Recall &#40;%&#41; | F1 &#40;%&#41; |)

[//]: # (|-------------------|---------------|------------|--------|)

[//]: # (| Template 0        | 53.81         | 70.40      | 61.00  |)

[//]: # (| Template 1        | 52.99         | 73.43      | 61.56  |)

[//]: # (| Template 2        | 52.78         | 47.87      | 50.21  |)

[//]: # ( | Template 3        | 53.83         | 70.85      | 61.18  |)

[//]: # ( | Template 4        | 100.00        | 0.22       | 0.45   |)

[//]: # ( | Voting            | 53.32         | 61.32      | 57.61  |)

[//]: # ( | Union             | 51.79         | 87.67      | 65.11  |)

[//]: # ( | Majority Baseline | 50.00         | 100.00     | 66.67  |)

[//]: # (Above are the reported test set results from Davinci-003 with 32 tokens, 0.7 temperature, no ICL, and no placeholders.)

## Results
Note: the temperature should be 0.0 to avoid fluctuations in the results due to randomness.

| GPT-3 Davinci                          | Precision (%) | Recall (%) | F1 (%) | AUC (%) | Average Precision (%) |
|----------------------------------------|---------------|------------|--------|---------|-----------------------|
 | ICL=lbl T=1 *Temperature==0.7!!        | 59.49         | 95.56      | 73.33  | 71.10   | 71.20                 |
 | ICL=cot T=0                            | 60.98         | 94.28      | 74.06  | 67.94   | 68.09                 |
 | ICL=cot T=1                            | 61.22         | 95.24      | 74.53  | 74.58   | 74.65                 |
| ICL=cot T=2                            | 48.42         | 87.62      | 62.37  | 48.25   | 48.44                 |
| ICL=cot T=3                            | 56.58         | 95.56      | 71.07  | 65.63   | 65.80                 |
 | ICL=cot args=type T=0                  | 67.07         | 69.84      | 68.43  | 63.73   | 63.94                 |
| ICL=cot args=type T=1                  | 69.97         | 76.19      | 72.95  | 69.89   | 70.07                 |
| ICL=cot args=type T=2                  | 50.43         | 74.29      | 60.08  | 53.55   | 53.78                 |
| ICL=cot args=type T=3                  | 68.73         | 70.48      | 69.59  | 66.87   | 67.13                 |
| ICL=cot args=type Vote                 | 69.30         | 69.52      | 69.41  | -       | -                     |
| ICL=cot args=type Any                  | 53.14         | 88.57      | 66.43  | -       | -                     |
| ICL=cot args=type Concensus            | 68.42         | 57.78      | 62.65  | -       | -                     |
 | ICL=lbl args=type T=0                  | 64.94         | 67.61      | 66.25  | 65.08   | 65.32                 |
 | ICL=lbl args=type T=1                  | 67.72         | 67.94      | 67.83  | 69.80   | 69.98                 |
 | ICL=lbl args=type T=2                  | 49.02         | 71.75      | 58.25  | 50.18   | 50.37                 |
 | ICL=lbl args=type T=3                  | 64.35         | 67.62      | 65.94  | 66.69   | 66.87                 |
 | ICL=lbl args=type Vote                 | 65.52         | 66.35      | 65.93  | -       | -                     |
 | ICL=lbl args=type Any                  | 51.47         | 83.17      | 63.59  | -       | -                     |
 | ICL=lbl args=type Concensus            | 67.32         | 54.92      | 66.25  | -       | -                     |
| Zero-shot args=type T=0                | 53.03         | 74.92      | 62.11  | 54.51   | 54.71                 |
 | Zero-shot args=type T=1                | 55.53         | 84.44      | 67.00  | 56.19   | 56.45                 |
 | Zero-shot args=type T=2                | 51.06         | 30.48      | 38.17  | 53.14   | 53.34                 |
 | Zero-shot args=type T=3                | 56.76         | 67.94      | 61.85  | 61.17   | 61.39                 |
 | Zero-shot args=type Vote               | 56.09         | 62.86      | 59.28  | -       | -                     |
 | Zero-shot args=type Any                | 52.90         | 89.84      | 66.59  | -       | -                     |
 | Zero-shot args=type Concensus          | 57.97         | 25.40      | 35.32  | -       | -                     |
| ICL=cot args=type T=0 instruct         | 51.38         | 94.60      | 66.59  | 60.12   | 60.29                 |
| ICL=cot args=type T=1 instruct         | 49.92         | 98.73      | 66.31  | 56.00   | 56.22                 |
| ICL=cot args=type T=2 instruct         | 49.76         | 99.05      | 66.24  | 49.04   | 49.24                 |
| ICL=cot args=type T=3 instruct         | 50.41         | 96.83      | 66.30  | 58.84   | 59.05                 |
| ICL=cot args=type Vote instruct        | 50.33         | 97.14      | 66.31  | -       | -                     |
| ICL=cot args=type Any instruct         | 50.00         | 100.00     | 66.67  | -       | -                     |
| ICL=cot args=type Concensus instruct   | 51.30         | 93.65      | 66.29  | -       | -                     |
| ICL=cot args=type T=0 code-002         | 69.47         | 49.84      | 58.04  | 64.22   | 64.34                 |
| ICL=cot args=type T=1 code-002         | 74.63         | 64.44      | 69.17  | 71.40   | 71.46                 |
| ICL=cot args=type T=2 code-002         | 62.08         | 47.30      | 53.69  | 58.30   | 58.59                 |
| ICL=cot args=type T=3 code-002         | 71.55         | 52.70      | 60.69  | 65.54   | 65.64                 |
| ICL=cot args=type Vote code-002        | 73.24         | 49.52      | 59.09  | -       | -                     |
| ICL=cot args=type Any code-002         | 65.04         | 72.06      | 68.37  | -       | -                     |
| ICL=cot args=type Concensus code-002   | 70.75         | 33.02      | 45.02  | -       | -                     |
| ICL=cot args=pseudo T=0 text-003       | 49.47         | 59.0(47)   | 53.84  | 52.22   | 52.39                 |
| ICL=cot args=pseudo T=1 text-003       | 51.20         | 67.62      | 58.28  | 53.99   | 54.16                 |
| ICL=cot args=pseudo T=2 text-003       | 47.88         | 25.08      | 32.92  | 49.94   | 50.12                 |
| ICL=cot args=pseudo T=3 text-003       | 50.00         | 50.79      | 50.39  | 50.98   | 51.19                 |
| ICL=cot args=pseudo Vote text-003      | 48.94         | 44.13      | 46.41  | -       | -                     |
| ICL=cot args=pseudo Any text-003       | 49.90         | 75.87      | 60.20  | -       | -                     |
| ICL=cot args=pseudo Concensus text-003 | 49.30         | 22.54      | 30.94  | -       | -                     |
