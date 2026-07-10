
from datasets import load_dataset
from openai import OpenAI
import re

class GSM8KBenchmark:
    def __init__(self, model: str = "gpt-4o", num_samples: int = 200):
        self.model = model
        self.num_samples = num_samples
        self.client = OpenAI()

    def run(self) -> dict:
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{self.num_samples}]")
        correct = 0
        for item in ds:
            pred = self._solve(item["question"])
            gold = self._extract_answer(item["answer"])
            if pred == gold:
                correct += 1
        accuracy = correct / len(ds)
        return {"accuracy": accuracy, "correct": correct, "total": len(ds)}

    def _solve(self, question: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Solve the math problem step by step. End with: The answer is <number>."},
                {"role": "user", "content": question}
            ], temperature=0)
        return self._extract_answer(r.choices[0].message.content)

    def _extract_answer(self, text: str) -> str:
        m = re.search(r"####\s*([\d,\.]+)", text) or re.search(r"answer is ([\d,\.]+)", text, re.I)
        return m.group(1).replace(",", "") if m else ""
