"""
Hallucination Detector: Dual approach using NLI model + LLM-as-judge
"""
from transformers import pipeline
from openai import OpenAI
import json

class HallucinationDetector:
    def __init__(self, nli_model: str = "cross-encoder/nli-deberta-v3-base"):
        self.nli = pipeline("text-classification", model=nli_model, device=-1)
        self.llm = OpenAI()
        self.nli_weight = 0.5
        self.llm_weight = 0.5

    def score(self, claim: str, context: str) -> dict:
        """Score hallucination probability for a claim given context."""
        nli_score = self._nli_score(claim, context)
        llm_score = self._llm_judge_score(claim, context)
        combined = self.nli_weight * nli_score + self.llm_weight * llm_score
        return {
            "hallucination_score": combined,
            "is_hallucinated": combined > 0.5,
            "nli_score": nli_score,
            "llm_score": llm_score,
        }

    def _nli_score(self, claim: str, context: str) -> float:
        result = self.nli(f"{context} [SEP] {claim}", top_k=None)
        scores = {r["label"].lower(): r["score"] for r in result}
        # If "contradiction" probability is high, likely hallucinated
        return scores.get("contradiction", 0.0)

    def _llm_judge_score(self, claim: str, context: str) -> float:
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Given this context:
{context}

Does this claim hallucinate (make things up not in context)?
Claim: {claim}

Respond with JSON: {{"hallucinated": true/false, "confidence": 0.0-1.0, "reason": "..."}}"""
            }],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return result["confidence"] if result["hallucinated"] else 1 - result["confidence"]

    def batch_score(self, samples: list[dict]) -> list[dict]:
        return [{"id": s.get("id"), **self.score(s["claim"], s["context"])} for s in samples]
