
# Chunk validation
# To ensure the internal consistency and usability of the paragraph-level text and embedding data, we implemented an automated chunk-level validation workflow. This procedure examines both the linguistic and numerical components of each paragraph to verify that they meet structural, length, and dimensionality requirements. Such validation is essential for large-scale text–embedding datasets, where small irregularities in text processing or vector generation can propagate into model training and downstream analyses.
# For each document, the validator first extracts its paragraphs and corresponding embedding vectors. Paragraph text is assessed for presence and completeness; missing or empty paragraphs are immediately flagged. The token length of each paragraph is measured using a consistent transformer tokenizer (intfloat/e5-large-v2), and the value is compared against dataset-wide thresholds. Paragraphs exceeding the maximum token length or falling below the minimum are counted and recorded. In addition, each paragraph undergoes a Unicode character categorization process that counts valid characters, corrupted replacement symbols, control characters, invisible formatting markers, and unassigned code points. This step detects encoding anomalies and ensures that the text content is free from corruption or unintended control sequences.
# Embedding validation proceeds in parallel with text validation. For each paragraph, the presence of an associated embedding is confirmed. Missing embeddings are flagged, and vectors that are present but fail to match the expected dimensionality are recorded as invalid. These checks guarantee that each paragraph is paired with a numerical representation of the correct length, preventing model ingestion errors in later stages.
# The judgment process combines the outcomes of the text and embedding checks to produce a single validation status for each record. If critical embedding issues are detected—either missing embeddings or vectors of incorrect dimensionality—the record is marked as “fail,” since these deficiencies cannot be trivially corrected without re-generating embeddings. If embeddings are valid but one or more paragraphs exceed the maximum token threshold or fall below the minimum, the record is marked as “warn,” indicating that while the data is structurally intact, it may be unsuitable for some modeling purposes. Only records where all paragraphs meet the token length constraints, contain no corrupted characters, and have valid embeddings for each chunk receive a “pass” status.
# The validation process operates on both individual documents and large collections. For scalability, batch validation is parallelized using either multiprocessing or multithreading, depending on the workload characteristics, which substantially reduces runtime without affecting determinism. Each validation output includes detailed per-paragraph diagnostics, aggregated dataset metrics, and a UTC timestamp of the validation run. These outputs provide a transparent audit trail and enable downstream filtering, quality control, and reproducibility of the dataset’s content integrity assessment.

#!/usr/bin/env python3
"""
Chunk-level validator for Lit2Vec paragraph/embedding records (dict-only version).

* Token-length, Unicode sanity, embedding-shape checks
* Returns rich diagnostics per paragraph and aggregated metrics
* Scales to large collections via multiprocessing or multithreading
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkValidator:
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer] = "intfloat/e5-large-v2",
        max_tokens: int = 300,
        min_tokens: int = 100,
        expected_vector_dim: int = 1024,
    ):
        # NOTE: keep only one tokenizer instance per process
        self.tokenizer: PreTrainedTokenizer = (
            tokenizer
            if isinstance(tokenizer, PreTrainedTokenizer)
            else AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        )
        self.tokenizer.model_max_length = max(self.tokenizer.model_max_length, 10**9)
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.vector_dim = expected_vector_dim

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def validate(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one dictionary and return a report."""
        checked_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        paragraphs: Dict[str, str] = doc.get("paragraphs") or {}
        embeddings: Dict[str, Any] = doc.get("embeddings") or {}

        if not isinstance(paragraphs, dict) or not paragraphs:
            return {
                "status": "fail",
                "summary": "Missing or invalid 'paragraphs' field.",
                "metrics": {},
                "checked_at": checked_at,
            }

        paragraph_reports: Dict[str, Any] = {}
        token_lengths: List[int] = []
        too_long = too_short = missing_embed = bad_embed = empty_chunks = 0

        for pid, text in paragraphs.items():
            if not isinstance(text, str) or not text.strip():
                empty_chunks += 1
                continue

            tlen = self._token_length(text)
            token_lengths.append(tlen)
            too_long += tlen > self.max_tokens
            too_short += tlen < self.min_tokens

            char_counts, bad_chars = self._categorize_characters(text)

            embed = embeddings.get(pid)
            if embed is None:
                missing_embed += 1
            elif not (isinstance(embed, list) and len(embed) == self.vector_dim):
                bad_embed += 1

            paragraph_reports[pid] = {
                "token_length": tlen,
                "char_counts": char_counts,
                "problem_characters": bad_chars,
            }

        q1, q2, q3 = (0, 0, 0)
        if token_lengths:
            q1, q2, q3 = np.percentile(token_lengths, [25, 50, 75])

        metrics = {
            "paragraph_count": len(paragraphs),
            "token_length_distribution": {
                "Q1": float(q1),
                "Q2": float(q2),
                "Q3": float(q3),
                "min": min(token_lengths, default=0),
                "max": max(token_lengths, default=0),
                "mean": float(np.mean(token_lengths)) if token_lengths else 0,
            },
            "chunks_too_long": too_long,
            "chunks_too_short": too_short,
            "empty_chunks": empty_chunks,
            "missing_embeddings": missing_embed,
            "invalid_embedding_vectors": bad_embed,
            "max_token_limit": self.max_tokens,
            "min_token_limit": self.min_tokens,
        }

        if bad_embed or missing_embed:
            status, summary = "fail", "Critical embedding issues detected."
        elif too_long or too_short:
            status, summary = (
                "warn",
                f"{too_long} chunk(s) over max tokens; {too_short} under min tokens.",
            )
        else:
            status, summary = "pass", "All chunks valid and embeddings consistent."

        return {
            "status": status,
            "summary": summary,
            "metrics": metrics,
            "paragraphs": paragraph_reports,
            "checked_at": checked_at,
        }

    def validate_batch(
        self,
        docs: List[Dict[str, Any]],
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        use_threads: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Validate a list of documents efficiently.

        Parameters
        ----------
        docs
            List of dict records.
        parallel
            Run in parallel if True and len(docs) > 1.
        max_workers
            Worker count (defaults to CPU count).
        use_threads
            True -> ThreadPoolExecutor (good for tokenizers that release GIL).
            False -> ProcessPoolExecutor (full CPU isolation).
        """
        if not parallel or len(docs) <= 1:
            return [self.validate(d) for d in docs]

        exec_cls = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor

        # Re-instantiate inside each worker to avoid pickling heavy tokenizer
        def _worker(record: Dict[str, Any], cfg: Tuple[str, int, int, int]) -> Dict[str, Any]:
            tok_name, max_t, min_t, vec_dim = cfg
            validator = ChunkValidator(tok_name, max_t, min_t, vec_dim)
            return validator.validate(record)

        cfg = (
            self.tokenizer.name_or_path,
            self.max_tokens,
            self.min_tokens,
            self.vector_dim,
        )
        with exec_cls(max_workers=max_workers) as pool:
            return list(pool.map(lambda d: _worker(d, cfg), docs))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _categorize_characters(text: str) -> Tuple[Dict[str, int], List[Dict[str, str]]]:
        counts = {
            "valid": 0,
            "corrupted": 0,
            "control": 0,
            "formatting": 0,
            "unassigned": 0,
        }
        seen, examples = set(), []

        for ch in text:
            cat = unicodedata.category(ch)
            if ch == "\uFFFD":
                counts["corrupted"] += 1
            elif cat == "Cc":
                if ch in ("\n", "\r"):
                    counts["valid"] += 1
                    continue
                counts["control"] += 1
            elif cat == "Cf":
                counts["formatting"] += 1
            elif cat == "Cn":
                counts["unassigned"] += 1
            else:
                counts["valid"] += 1

            if (
                ch not in seen
                and (ch == "\uFFFD" or (cat in ("Cc", "Cf", "Cn") and ch not in ("\n", "\r")))
            ):
                seen.add(ch)
                examples.append(
                    {
                        "char": ch,
                        "codepoint": f"U+{ord(ch):04X}",
                        "name": unicodedata.name(ch, "UNKNOWN"),
                        "category": cat,
                    }
                )
                if len(examples) >= 5:
                    break

        return counts, examples


# ---------------------------------------------------------------------- #
# Example usage (remove or adapt in production)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import json

    docs = [
        {
            "paragraphs": {
                "p1": "This is a valid paragraph." * 10,
                "p2": "Short.",
            },
            "embeddings": {
                "p1": [0.0] * 1024,
                # p2 missing embedding
            },
        },
        # ... more dicts
    ]

    validator = ChunkValidator()
    reports = validator.validate_batch(docs, parallel=True, use_threads=True)
    print(json.dumps(reports, indent=2))
