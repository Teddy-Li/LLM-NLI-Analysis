# GPT3 Experiments with Levy/Holt Directional Subset

## Dataset

The dataset is the directional subset of the Levy/Holt dataset. It has 64 pairs of true directional entailments for dev set,
and 892 pairs of true directional entailments for test set. (due to some quirks of the dataset, the numbers of pairs between dev and test set vary vastly.)

The dataset create false entailment entries by swapping the order of the premise and hypothesis in the true directional entailments.
So in total there are 64 * 2 = 128 entries for dev set, and 892 * 2 = 1784 entries for test set.

## Running

```python gpt3_dirent.py```

There are a few options:

1. `--model_name` specifies the GPT-3 model to use. The default is `text-davinci-003`;
2. `--max_tokens` specifies the maximum number of tokens to output for each entry. The default is 32;
3. `--temperature` controls the temperature of generation, which is 0.7 by default;
4. `--use_plhr` specifies whether to use placeholders and how. The default is `none`, but can also be `xy` or `type`;
5. `--in_context` specifies whether to do ICL, which is `False` by default, context examples are between line 31-48;
6. `--num_templates` specifies the number of templates to try, the current set of all templates are within the list `sent_template_to_test`;
7. `--split` specifies the split to run on, either `dev` or `test`; the zero-shot results were reported on test set.

## Preliminary Results

| GPT-3 Da Vinci  | Precision (%) | Recall (%) | F1 (%) |
|-----------------|---------------|------------|--------|
| Template 0      | 53.81         | 70.40      | 61.00  |
| Template 1      | 52.99         | 73.43      | 61.56  |
| Template 2      | 52.78         | 47.87      | 50.21  |
 | Template 3      | 53.83         | 70.85      | 61.18  |
 | Template 4      | 100.00        | 0.22       | 0.45   |
 | Majority        | 53.32         | 61.32      | 57.61  |
 | Union           | 51.79         | 87.67      | 65.11  |
 | Random Baseline | 50.00         | 100.00     | 66.67  |

Above are the reported test set results from Davinci-003 with 32 tokens, 0.7 temperature, no ICL, and no placeholders.
