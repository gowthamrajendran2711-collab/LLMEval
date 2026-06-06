"""HotpotQA multi-hop reasoning benchmark"""
from datasets import load_dataset
from openai import OpenAI

class HotpotQABenchmark:
    def __init__(self, model: str = "gpt-4o", num_samples: int = 200):
        self.model = model
        self.num_samples = num_samples
        self.client = OpenAI()

    def run(self) -> dict:
        ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{self.num_samples}]")
        f1_scores, em_scores = [], []
        for item in ds:
            context = " ".join([s for ctx in item["context"]["sentences"] for s in ctx])
            pred = self._answer(item["question"], context)
            gold = item["answer"]
            em_scores.append(1.0 if pred.strip().lower() == gold.strip().lower() else 0.0)
            f1_scores.append(self._f1(pred, gold))
        return {"f1": sum(f1_scores)/len(f1_scores), "exact_match": sum(em_scores)/len(em_scores), "num_samples": len(ds)}

    def _answer(self, question: str, context: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer the question based on the context. Be concise (1-5 words)."},
                {"role": "user", "content": f"Context: {context[:3000]}\n\nQuestion: {question}"}
            ], temperature=0, max_tokens=50)
        return r.choices[0].message.content.strip()

    def _f1(self, pred: str, gold: str) -> float:
        p_tokens = set(pred.lower().split())
        g_tokens = set(gold.lower().split())
        if not p_tokens or not g_tokens: return 0.0
        common = p_tokens & g_tokens
        if not common: return 0.0
        p = len(common)/len(p_tokens)
        r = len(common)/len(g_tokens)
        return 2*p*r/(p+r)
