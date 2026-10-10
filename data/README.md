# Data guide

Put a chosen real CSV in `data/raw/`. The default filename is `hinglish_sentiment.csv`; change `paths.raw_data` in `config.yaml` or pass `--input`.

Required logical columns are `text` and `label`; their names are configurable. Labels may be positive/negative/neutral or numeric values that are explicitly declared in `config.yaml`.

Possible sources are SemEval-2020 Task 9 (SentiMix Hinglish), GLUECoS Hinglish sentiment, a Kaggle Hinglish dataset, or a custom self-labelled CSV. Check the source licence, label definitions, and split policy first.

## Bundled sample

`sample/hinglish_smoke_sample.csv` is 30 hand-written examples for pipeline smoke tests only. It is not representative, not a benchmark, and must never be cited as experimental data. It is only loaded with `--sample`.
