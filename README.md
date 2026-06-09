# CausalPhys

### Causal Scaffolding for Physical Reasoning: A Benchmark for Causally-Informed Physical World Understanding in VLMs

> 🎉 **Accepted at KDD 2026 (Datasets and Benchmarks Track).**

[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-causalphys-2ea44f.svg)](https://haorentang.github.io/CausalPhys/)
[![Paper](https://img.shields.io/badge/📄%20Paper-2606.05966-b31b1b.svg)](https://arxiv.org/abs/2606.05966)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-haorentang%2Fcausalphys-yellow)](https://huggingface.co/datasets/haorentang/causalphys)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**CausalPhys** is a benchmark of 3,000+ image- and video-based questions for evaluating the *causal* physical reasoning of vision-language models (VLMs). Each question is paired with a ground-truth **causal graph**, enabling mechanism-level evaluation along four metrics — **ACC** (Accuracy), **EF** (Entity Faithfulness), **RA** (Relation Awareness), and **DC** (Description Correctness). We further provide **CRFT** (Causal Rationale Fine-Tuning), which guides VLMs to reason through causal mechanisms rather than surface correlations.

## Setup

```bash
pip install -r requirements.txt

# Download the benchmark
huggingface-cli download haorentang/causalphys --repo-type dataset --local-dir dataset

# API keys for hosted models (do NOT commit them)
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export GOOGLE_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Evaluation

Run the full causal-graph-grounded evaluation (CoT generation → answer verification → EF/RA/DC judging):

```bash
# 1. Generate model responses (rationale + answer)
python evaluation/generate_responses.py --dataset_dir ./dataset --models gpt-4o-mini internvl-3-78b

# 2. Score responses with the causal-graph-grounded LLM judge
python evaluation/evaluate_responses.py --baseline_path evaluation_results/gpt-4o-mini --use_judge

# 3. Aggregate & compare results
python evaluation/analyze_results.py --compare --models gpt-4o-mini internvl-3-78b
```

Or run the end-to-end evaluation in one command:

```bash
bash scripts/run_causal_eval.sh
```

## SFT (answer-only baseline)

```bash
bash scripts/run_train_sft_answer_only.sh
```

## CRFT (Causal Rationale Fine-Tuning)

```bash
# 1. Generate gold causal rationales with a teacher LLM (e.g., GPT-4o)
bash scripts/run_generate_rationales.sh

# 2. Fine-tune the VLM on rationale + answer
bash scripts/run_train_rationale_from_checkpoint.sh
```

## Citation

```bibtex
@inproceedings{causalphys2026,
  title     = {Causal Scaffolding for Physical Reasoning: A Benchmark for
               Causally-Informed Physical World Understanding in VLMs},
  author    = {CausalPhys Authors},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining (KDD), Datasets and Benchmarks Track},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
