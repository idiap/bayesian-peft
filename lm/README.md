# Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting: language modeling experiments

The scripts under `language-modeling` and `text-classification` in this repository are adapted from the Hugging Face [Transformers](https://github.com/huggingface/transformers/tree/v4.34.0/examples/pytorch) version 4.34.0.

# Setup

1. Follow instructions from [Transformers](https://github.com/huggingface/transformers) to setup the python envrionment.
2. Install the customized peft package.

## Hessian estimation

Prepare data following `scripts/preprocess.py`.

Scripts for Hessian estimation are in  `scripts/compute_hessian`.

## Fine-tuning

Scripts for fine-tuning are in `scripts/glue_nt` for text classification and `scripts/clm_nt` for causal language modeling.

## Evaluation

Evaluation for the target task are conducted automatically after fine-tuning.

Scripts for evaluating perplexity are in `scripts/eval_perplexity`.
