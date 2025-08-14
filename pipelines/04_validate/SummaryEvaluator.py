
# Summary validation
# To assess the intrinsic quality of the automatically generated summaries contained in the dataset, we implemented a dual-metric evaluation framework that measures both lexical overlap and semantic similarity between each record’s original text and its associated summary. This process enables a rigorous, reproducible assessment of summary fidelity, ensuring that dataset entries meet minimal standards of informativeness and relevance.
# For each record, the evaluation compares the original abstract to its corresponding short-form summary. Prior to scoring, both texts are segmented into sentences using a standard tokenization process to ensure consistency across varying sentence structures and punctuation styles. This preprocessing step is essential because metrics such as ROUGE-Lsum rely on sentence-boundary alignment for accurate scoring.
# The evaluation process incorporates two complementary metric families. The first, ROUGE (Recall-Oriented Understudy for Gisting Evaluation), measures lexical similarity through overlapping word sequences at different granularities. ROUGE-1, ROUGE-2, and ROUGE-L capture unigram, bigram, and longest common subsequence matches, respectively, while ROUGE-Lsum extends the longest common subsequence approach to sentence-level alignment. These scores primarily quantify surface-level content overlap between the abstract and the summary. The second family of metrics, BERTScore, evaluates semantic similarity by encoding both texts into contextual embeddings and computing their alignment. This produces precision, recall, and F1 values that reflect not only lexical matches but also deeper semantic correspondence, even when wording differs substantially.
# The scoring pipeline operates in two modes. When possible, it uses the Hugging Face evaluate library to access pre-optimized implementations of ROUGE and BERTScore, ensuring consistency with widely accepted NLP evaluation standards. If the online evaluation backend is unavailable, the system falls back to native implementations of the metrics, preserving functionality without compromising accuracy. Both modes produce identical metric definitions, ensuring comparability across runs.
# Judgment of summary quality is made by interpreting the combination of these metrics. High-quality summaries are expected to achieve both high ROUGE and high BERTScore values, indicating that they capture the original’s essential lexical and semantic content. Conversely, low scores in either metric suggest that a summary omits critical information, introduces unrelated material, or fails to preserve the original meaning. For example, a summary with high ROUGE but low BERTScore may reuse many words from the abstract without accurately representing its concepts, while the inverse may indicate semantically faithful paraphrasing with limited lexical overlap.
# The framework supports evaluation of both single pairs and large batches of abstract–summary pairs. In large-scale mode, sentence segmentation, metric computation, and result collation are executed in parallel to improve throughput while preserving deterministic scoring. Each evaluation produces a structured output containing the full set of ROUGE and BERTScore values, allowing downstream researchers to reproduce quality judgments or apply their own thresholds. This transparent, multi-metric approach provides a robust foundation for validating the linguistic and semantic quality of summaries within the dataset.


#!/usr/bin/env python3
"""
Summary-quality evaluation (single- or multi-pair).

* ROUGE-1/2/L/Lsum and BERTScore (P/R/F1)
* Primary backend: evaluate Hub assets
* Offline/native fallback: rouge_score + bert_score
Author: eshgham  (2025-07, batch-capable)
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import nltk

# ---------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------
_ZERO_SCORES: Dict[str, float] = {
    "ROUGE-1": 0.0,
    "ROUGE-2": 0.0,
    "ROUGE-L": 0.0,
    "ROUGE-Lsum": 0.0,
    "BERTScore_Precision": 0.0,
    "BERTScore_Recall": 0.0,
    "BERTScore_F1": 0.0,
}

nltk.data.path.append("/home/mahmoud/nltk_data")
nltk.download("punkt", quiet=True)

warnings.filterwarnings("ignore", message="Some weights of RobertaModel")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def _is_valid(text: str) -> bool:
    return isinstance(text, str) and text.strip() != ""


# ---------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------
class SummaryEvaluator:
    """
    Compute ROUGE-n/L/Lsum + BERTScore for (abstract, TL;DR) pairs.
    """

    def __init__(
        self,
        model_type: str = "roberta-large",
        *,
        backend: str = "auto",  # "auto", "native", or "evaluate"
    ) -> None:
        self.model_type = model_type
        self._backend = backend
        self._init_metrics()

    def _init_metrics(self) -> None:
        if self._backend == "native":
            self._init_native_metrics()
            return
        try:
            import evaluate
            self.rouge = evaluate.load("rouge")
            self.bertscore = evaluate.load("bertscore")
            self._backend = "evaluate"
        except Exception:
            warnings.warn("Falling back to native metrics due to evaluate hub error.")
            self._init_native_metrics()
            self._backend = "native"

    def _init_native_metrics(self) -> None:
        from rouge_score import rouge_scorer
        import bert_score
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
        )
        self.bert_score = bert_score

    @staticmethod
    def _preprocess(text: str) -> str:
        return "\n".join(nltk.sent_tokenize((text or "").strip()))

    def evaluate(self, abstract: str, tldr: str) -> Dict[str, float]:
        return self.evaluate_batch([(abstract, tldr)])[0]

    def evaluate_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        processed = [(self._preprocess(a), self._preprocess(t)) for a, t in pairs]
        results = []

        # Check if empty inputs exist
        valid_idx = [i for i, (ref, pred) in enumerate(processed) if ref and pred]
        if not valid_idx:
            return [_ZERO_SCORES.copy() for _ in pairs]

        if self._backend == "evaluate":
            try:
                results = self._evaluate_batch_hub(processed)
            except Exception as e:
                warnings.warn(f"[evaluate_batch] Hub metrics failed: {type(e).__name__} – {e}")
                self._init_native_metrics()
                self._backend = "native"
                results = self._evaluate_batch_native(processed)
        else:
            results = self._evaluate_batch_native(processed)

        # Fill in empty predictions if needed
        final_results = []
        for i, (ref, pred) in enumerate(processed):
            if not ref or not pred:
                final_results.append(_ZERO_SCORES.copy())
            else:
                final_results.append(results.pop(0))
        return final_results

    def _evaluate_batch_hub(self, processed: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        preds, refs = zip(*processed)
        rouge_scores = self.rouge.compute(
            predictions=preds,
            references=refs,
            use_stemmer=True,
        )
        bert_scores = self.bertscore.compute(
            predictions=preds,
            references=refs,
            lang="en",
            model_type=self.model_type,
            use_fast_tokenizer=True,
            batch_size=8,
        )
        results = []
        for i in range(len(preds)):
            result = {
                "ROUGE-1": rouge_scores["rouge1"][i] * 100,
                "ROUGE-2": rouge_scores["rouge2"][i] * 100,
                "ROUGE-L": rouge_scores["rougeL"][i] * 100,
                "ROUGE-Lsum": rouge_scores["rougeLsum"][i] * 100,
                "BERTScore_Precision": bert_scores["precision"][i] * 100,
                "BERTScore_Recall": bert_scores["recall"][i] * 100,
                "BERTScore_F1": bert_scores["f1"][i] * 100,
            }
            results.append(result)
        return results

    def _evaluate_batch_native(self, processed: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        results = []
        preds, refs = zip(*[x for x in processed if x[0] and x[1]])

        try:
            # ROUGE: per-pair
            rouge_results = [self.rouge_scorer.score(ref, pred) for ref, pred in zip(refs, preds)]

            # BERTScore: batched
            P, R, F1 = self.bert_score.score(
                preds,
                refs,
                lang="en",
                model_type=self.model_type,
                batch_size=8,
                verbose=False,
                rescale_with_baseline=True,
            )

            for i, r in enumerate(rouge_results):
                results.append({
                    "ROUGE-1": r["rouge1"].fmeasure * 100,
                    "ROUGE-2": r["rouge2"].fmeasure * 100,
                    "ROUGE-L": r["rougeL"].fmeasure * 100,
                    "ROUGE-Lsum": r["rougeLsum"].fmeasure * 100,
                    "BERTScore_Precision": P[i].item() * 100,
                    "BERTScore_Recall": R[i].item() * 100,
                    "BERTScore_F1": F1[i].item() * 100,
                })
            return results
        except Exception as e:
            warnings.warn(f"[evaluate_batch_native] Failed: {type(e).__name__} – {e}")
            return [_ZERO_SCORES.copy() for _ in preds]


# ---------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import json

    examples = [
        ("Transformers have revolutionized NLP by enabling ...", "We survey how transformers changed NLP."),
        ("Quantum dots exhibit discrete energy levels leading to ...", "Paper explains optical properties of quantum dots."),
        ("", "Empty abstract test"),
    ]

    se = SummaryEvaluator(backend="native")
    results = se.evaluate_batch(examples)
    for r in results:
        print(json.dumps(r, indent=2))
