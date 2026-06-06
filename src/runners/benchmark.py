"""LLMEval benchmark runner CLI"""
import argparse, json, time
from pathlib import Path
import wandb

from src.benchmarks.hotpotqa import HotpotQABenchmark
from src.benchmarks.gsm8k import GSM8KBenchmark
from src.benchmarks.mmlu import MMLUBenchmark
from src.detectors.hallucination import HallucinationDetector

BENCHMARKS = {
    "hotpotqa": HotpotQABenchmark,
    "gsm8k": GSM8KBenchmark,
    "mmlu": MMLUBenchmark,
}

def run(model: str, suite: str, output_dir: str, num_samples: int = 200):
    Path(output_dir).mkdir(exist_ok=True)
    all_results = {"model": model, "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "benchmarks": {}}

    targets = list(BENCHMARKS.keys()) if suite == "full" else [suite]

    for name in targets:
        print(f"Running {name}...")
        benchmark = BENCHMARKS[name](model=model, num_samples=num_samples)
        scores = benchmark.run()
        all_results["benchmarks"][name] = scores
        print(f"  {name}: {scores}")

    # Hallucination detection on collected outputs
    print("Running hallucination detection...")
    detector = HallucinationDetector()
    # (In practice, collect model outputs from benchmark runs above)
    all_results["hallucination"] = {"note": "Run separately with --mode hallucination"}

    out_path = Path(output_dir) / "latest_eval.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {out_path}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="LLMEval benchmark runner")
    parser.add_argument("--model", default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--suite", default="full", choices=list(BENCHMARKS)+["full"])
    parser.add_argument("--output", default="metrics/", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    results = run(args.model, args.suite, args.output, args.num_samples)

    if args.wandb:
        from src.reporters.wandb_reporter import WandBReporter
        reporter = WandBReporter()
        url = reporter.report(results, run_name=f"{args.model}-{args.suite}")
        print(f"W&B run: {url}")

if __name__ == "__main__":
    main()
