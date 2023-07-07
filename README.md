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



