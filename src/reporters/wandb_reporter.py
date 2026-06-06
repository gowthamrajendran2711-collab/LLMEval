"""Weights & Biases reporter for LLM eval results"""
import wandb
from pathlib import Path
import json

class WandBReporter:
    def __init__(self, project: str = "llm-eval", entity: Optional[str] = None):
        self.project = project
        self.entity = entity

    def report(self, metrics: dict, run_name: str = "eval", config: dict = None) -> str:
        """Log metrics to W&B and return the run URL."""
        run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config or {},
        )
        # Log all metrics
        wandb.log(metrics)

        # Create comparison table
        if "benchmarks" in metrics:
            table = wandb.Table(
                columns=["benchmark", "metric", "score"],
                data=[
                    [bench, metric, score]
                    for bench, scores in metrics["benchmarks"].items()
                    for metric, score in scores.items()
                    if isinstance(score, (int, float))
                ]
            )
            wandb.log({"benchmark_results": table})

        run_url = run.get_url()
        run.finish()
        return run_url
