# 📊 LLM Eval Framework

> Benchmarking, automated regression testing, and hallucination detection at scale.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![RAGAS](https://img.shields.io/badge/RAGAS-0.1-orange) ![DeepEval](https://img.shields.io/badge/DeepEval-0.21-green) ![W&B](https://img.shields.io/badge/W%26B-0.17-yellow)

---

## Overview

LLM Eval Framework is a comprehensive evaluation harness for LLMs. It provides benchmarking across standard datasets, automated regression testing in CI/CD, hallucination scoring, and rich reporting via Weights & Biases.

## Features

- **Multi-metric evaluation**: Faithfulness, relevancy, hallucination score, toxicity, BLEU, ROUGE
- **Benchmark datasets**: HotpotQA, TriviaQA, NaturalQuestions, GSM8K, MMLU, HumanEval
- **Hallucination detection**: NLI-based + LLM-as-judge dual approach
- **Regression testing**: pytest integration with metric threshold assertions
- **W&B integration**: Automatic logging, comparison tables, alert triggers
- **CI/CD ready**: GitHub Actions integration with PR comment reports

## Metrics & Achievements

| Benchmark | Score |
|-----------|-------|
| HotpotQA (F1) | **0.74** |
| TriviaQA (EM) | **0.81** |
| GSM8K (accuracy) | **0.88** |
| MMLU (5-shot) | **0.79** |
| HumanEval (pass@1) | **0.82** |
| Hallucination rate (NLI) | **6.2%** detected |
| Eval throughput | **340 samples/min** |

## Quick Start

```bash
pip install -r requirements.txt
cp configs/.env.example .env

# Run benchmark suite
python -m src.runners.benchmark --suite full --model gpt-4o --output metrics/

# Run regression tests
pytest tests/regression/ --tb=short

# Run hallucination detection
python -m src.detectors.hallucination --input data/model_outputs.jsonl --output metrics/hallucination_report.json
```

## Project Structure

```
LLMEval/
├── src/
│   ├── benchmarks/
│   │   ├── hotpotqa.py
│   │   ├── gsm8k.py
│   │   ├── mmlu.py
│   │   └── humaneval.py
│   ├── detectors/
│   │   ├── hallucination.py    # NLI + LLM judge
│   │   └── toxicity.py
│   ├── reporters/
│   │   ├── wandb_reporter.py
│   │   └── html_reporter.py
│   └── runners/
│       └── benchmark.py
├── tests/
│   └── regression/             # pytest regression suite
├── metrics/
└── configs/
```

## License

MIT
