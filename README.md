# Sources of Hallucination by Large Language Models on Inference Tasks

## Overview
This repository contains the code and data for the paper [Sources of Hallucination by Large Language Models on Inference Tasks](https://arxiv.org/abs/2305.14552).

## Data & Results Files

The data used in the paper can be found in the [levyholt_files](levyholt_files) directory. The directory contains the following sub-folders:

- with_original: contains the directional Levy/Holt entries with their original context (i.e., the original entities as arguments).
- with_type: contains the directional Levy/Holt entries with their type context (i.e., with typed placeholders of the entities as arguments). These files should be processed with 
- randprem_files: contains the random premise files used in Section 5/7 of the paper.
- swapped_entities: contains the files with entity-swapped entries used in the Section 6 of the paper.
- with_pseudoents_legacy: legacy files for a previous version of experiments with wugged words. These files are not used in the paper.

The model outputs for GPT models and LLaMA models are stored in the [results](./results/levyholt_results) directory. The directory contains the following sub-folders:

- gpt_results: contains the GPT model outputs (`text-davinci-003` and `gpt-4-0314`) for the directional Levy/Holt entries; original, entity-swap, random-premise and hypothesis-only results are included for `text-davinci-003`.
- llama_results: contains the LLaMA model outputs (`llama-65b`) for the directional Levy/Holt entries, original, entity-swap, random-premise and hypothesis-only results are included.

## Scripts

### Running Models on Natural Language Inference Tasks

- `gpt3_inference.py` contains the code to acquire natural language inference results with GPT-3/4 models.
  - Example Script: 
    - Dev set: `python -u gpt3_inference.py --model_name text-davinci-003 --use_plhr [original/type/lowfreq/highfreq/randprem-orig/randprem-type] --split dev --in_context cot --num_templates 4`
    - Test set: `python -u gpt3_inference.py --model_name text-davinci-003 --use_plhr original --split dev --in_context cot --tplt_id 1 --num_templates 1`
    - Hypothesis-only: `python -u gpt3_inference.py --model_name text-davinci-003 --use_plhr original --split test --in_context lbl --num_templates 1 --hypothesis-only`
- `llama_inference.py` contains the code for calculating the LLaMA results.
  - Example Script:
    - Dev set: `python -u llama_inference.py --model_root [YOUR_PATH_TO_MODEL] --model_name llama-65b-hf --use_plhr [original/type/lowfreq/highfreq/randprem-orig/randprem-type] --split dev --in_context cot --task lh`
    - Test set: `python -u llama_inference.py --model_root [YOUR_PATH_TO_MODEL] --model_name llama-65b-hf --use_plhr original --split test --in_context cot --task lh --tplt_id 1`
    - Hypothesis-only: `python -u llama_inference.py --model_root [YOUR_PATH_TO_MODEL] --model_name llama-65b-hf --use_plhr original --split test --in_context lbl --task lh --single_statement h`
  - Note: LLaMA is an LLM published by Meta, please visit [their website](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) to gain access to the model
- Attestation bias can be measured with the hyp-only setting in the two scripts above.
- Attestation results can be polled using the `poll_attestation.py` script.
- Relative frequency prior can be measured using the `get_frequencies_ngram.py` script.

### Controlled Experiments
- `randprem_experiments.py` contains the code to calculate the probabilities of predicting `Entail` conditioned on attestation or relative frequency.
- `attestation_controlled_experiment.py` contains the code to calculate the model performance on attestation-consistent / adversarial subsets of Levy/Holt.
- `frequency_controlled_experiments.py` contains the code to calculate the model performance on frequency-consistent / adversarial subsets of Levy/Holt.

