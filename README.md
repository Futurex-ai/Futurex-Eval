# FutureX Evaluation Scripts

This repository contains the evaluation code for the FutureX benchmark from the paper
[FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction](https://arxiv.org/abs/2508.11987).

It provides the core scoring functions used in FutureX evaluation:

- Level 1/2 (Type-A style): set/multi-label F1 scoring.
- Level 3/4 (Type-B style): LLM-as-judge scoring for numeric, string, and ranking answers.

## Environment Setup

Use Python 3.9+ and install required packages:

```bash
pip install openai tenacity
```

Set the LLM judge environment variables:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_API_BASE="YOUR_API_BASE"   # optional for official OpenAI
export OPENAI_API_MODEL="YOUR_API_MODEL" # e.g. gpt-4.1 / volc-deepseek-v3
```

## Main Files

- `eval.py`: public scoring entrypoints and evaluation helpers.
- `llm_judge_level_34.py`: L3/L4 judge implementation.
- `utils.py`: number parsing helpers (`to_float`, `is_number`).

## Usage

### Level 1/2

```python
from eval import estimate_type_a_score

y_true = [set(["A"]), set(["A", "B"])]
y_pred = [set(["A"]), set(["B"])]
scores, avg = estimate_type_a_score(y_true, y_pred)
print(scores, avg)
```

### Level 3/4

```python
from eval import estimate_type_b_score

questions = ["Q1", "Q2"]
y_true = [[3.2], ["Beijing"]]
y_pred = [["3.1"], ["北京市"]]
stds = [0.5, 1.0]
scores, avg = estimate_type_b_score(questions, y_true, y_pred, stds)
print(scores, avg)
```

## Notes On Level-3/4 Logic

- Numeric GT is normalized before scoring (including single-string numbers).
- Numeric `std` is auto-derived for numeric GT (`0.05 * gt`, with `0 -> 0.01`).
- Multi-choice ranking answers (e.g. `["A", "B", "C"]`) support order-insensitive exact match shortcut.
- Non-exact ranking answers use overlap-based partial credit.