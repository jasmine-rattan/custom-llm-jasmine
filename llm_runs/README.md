# LLM Runs

This directory holds all model weights, eval results, and training artifacts.

## run_001 — Starter corpus (classroom sentences only)
- `model_untrained.pt` — weights before any training
- `model.pt` — weights after 5,000 training steps
- `eval_results_untrained.csv` / `eval_summary_untrained.json` — result set 1
- `eval_results_trained.csv` / `eval_summary_trained.json` — result set 2
- `training_curves.svg`, `history.json`, `training_summary.json`
- `config.json`, `corpus.txt`, `vocabulary_report.json`, `tokenization.json`
- `inspection.json`, `checkpoint.json`, `temperature_comparison.json`
- `eval_separation.json`

## run_002 — Extended corpus (classroom + negation + categories + everyday knowledge)
- Same file structure as run_001
- `eval_results_untrained.csv` / `eval_summary_untrained.json` — result set 3
- `eval_results_trained.csv` / `eval_summary_trained.json` — result set 4

Populate each folder by downloading the ZIP from Colab after each experiment.
