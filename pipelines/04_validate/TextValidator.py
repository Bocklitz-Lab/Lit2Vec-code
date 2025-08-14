# Text Validation
# To verify the structural integrity, linguistic quality, and semantic consistency of the dataset’s textual components, we implemented an automated text validation pipeline. This procedure evaluates both the abstract and the full-text content for each record, ensuring that the textual data is sufficiently complete, correctly formatted, linguistically coherent, and semantically aligned between components. The validator combines length-based thresholds, character-level diagnostics, language identification, semantic similarity scoring, and optional embedding-based checks to produce a comprehensive quality assessment.
# For each record, the validator first checks the length of the abstract and the full text, applying minimum character and sentence thresholds to identify incomplete or truncated content. It then performs a character integrity scan that detects corrupted characters, invisible control codes, and unusual Unicode formatting symbols. This scan also records examples of problematic characters, which aids in diagnosing encoding or extraction issues. Whitespace density and ASCII letter share are measured to detect formatting anomalies such as collapsed whitespace or excessive non-alphabetic content, which often indicate parsing errors or inclusion of unrelated material.
# Sentence counts are calculated to ensure that both the abstract and the full text contain a plausible number of complete sentences, consistent with expected scholarly writing patterns. For the full text, the validator also detects heading markers, which serve as structural indicators of well-formatted sections and subsections. Their absence may suggest that the document is improperly segmented or missing structural metadata.
# Language detection is performed on the initial portion of the full text to confirm that the content matches the expected primary language and meets a minimum probability threshold. This step guards against inclusion of documents in other languages or those with insufficient linguistic coherence. The semantic relationship between the abstract and the full text is then evaluated using ROUGE-1 recall, computed between the abstract and the first 2,000 characters of the full text. Low recall values may indicate a mismatch between the summary and the body of the work.
# If an abstract embedding vector is present, its L2 norm is calculated and compared against an expected magnitude to ensure numerical consistency. Deviations from the expected norm or missing embeddings are flagged, as they can indicate upstream processing errors or incompatibilities in the embedding generation process.
# A record is classified as a “pass” when it meets all length, character integrity, language, semantic overlap, and embedding consistency requirements. Records with issues that are important but not fundamentally invalidating are marked as “warn,” while those failing critical checks—such as extremely short full texts or missing embeddings—are assigned a “fail” status. Each validation report includes the measured metrics, identified flags, problematic character examples, and an ISO 8601 timestamp indicating when the validation occurred. For large datasets, validation is parallelized across multiple processes to accelerate execution without affecting reproducibility.


#!/usr/bin/env python3
"""
Abstract / full-text validator for Lit2Vec records.

• Presence & minimum-length checks
• Unicode / control-character sweep
• Whitespace-collapse ratio
• ASCII-letter share
• Language detection (langdetect; optional fastText)
• Sentence-terminator count
• Heading markers in full-text
• Abstract ↔ full-text overlap via ROUGE-1 (first 2 k chars)
• Embedding-norm sanity (if abstract_embedding present)

Returns:
    {
        "status": "pass" | "warn" | "fail",
        "summary": "...",
        "metrics": {...},
        "checked_at": "ISO-8601",
    }
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
from langdetect import detect_langs, LangDetectException
from rouge_score import rouge_scorer
from functools import partial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextValidator:
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        min_abs_chars: int = 100,
        min_full_chars: int = 1000,
        min_abs_sentences: int = 2,
        min_full_sentences: int = 50,
        min_abs_ws_ratio: float = 0.75,
        min_full_ws_ratio: float = 0.83,
        min_abs_ascii_ratio: float = 0.7,
        min_full_ascii_ratio: float = 0.75,
        language: str = "en",
        lang_prob_threshold: float = 0.90,
        rouge_min_recall: float = 0.5,
        expected_emb_norm: float = 1.0,
        emb_norm_tol: float = 1e-3,
    ):
        self.min_abs_chars = min_abs_chars
        self.min_full_chars = min_full_chars
        self.min_abs_sentences = min_abs_sentences
        self.min_full_sentences = min_full_sentences
        self.min_abs_ws_ratio = min_abs_ws_ratio
        self.min_full_ws_ratio = min_full_ws_ratio
        self.min_abs_ascii_ratio = min_abs_ascii_ratio
        self.min_full_ascii_ratio = min_full_ascii_ratio
        self.language = language
        self.lang_prob_threshold = lang_prob_threshold
        self.rouge_min_recall = rouge_min_recall
        self.expected_emb_norm = expected_emb_norm
        self.emb_norm_tol = emb_norm_tol
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def validate(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        checked_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        abstract = doc.get("abstract", "")
        fulltext = doc.get("fulltext", "")

        metrics: Dict[str, Any] = {
            "abstract_chars": len(abstract),
            "fulltext_chars": len(fulltext),
            "abstract_bad_chars": {},
            "fulltext_bad_chars": {},
            "abstract_ws_ratio": None,
            "fulltext_ws_ratio": None,
            "abstract_ascii_ratio": None,
            "fulltext_ascii_ratio": None,
            "abstract_sentence_count": None,
            "fulltext_sentence_count": None,
            "fulltext_heading_markers": None,
            "language": "unknown",
            "language_score": 0.0,
            "rouge1_recall": None,
            "abstract_embedding_norm": None,
        }

        flags: Dict[str, bool] = {}
        abstr_ex, full_ex = [], []

        # 1. Length Checks
        flags["abstract_too_short"] = not abstract or len(abstract) < self.min_abs_chars
        flags["fulltext_too_short"] = not fulltext or len(fulltext) < self.min_full_chars

        # 2. Unicode character checks
        try:
            metrics["abstract_bad_chars"], abstr_ex = self._categorize_characters(abstract)
            metrics["fulltext_bad_chars"], full_ex = self._categorize_characters(fulltext)
            flags["abstract_has_corrupted_chars"] = metrics["abstract_bad_chars"].get("corrupted", 0) > 0
            flags["fulltext_has_corrupted_chars"] = metrics["fulltext_bad_chars"].get("corrupted", 0) > 0
        except Exception:
            abstr_ex, full_ex = [], []
            flags["character_scan_failed"] = True

        # 3. Whitespace ratio
        try:
            abs_ws_ratio = self._whitespace_ratio(abstract)
            full_ws_ratio = self._whitespace_ratio(fulltext)
            metrics["abstract_ws_ratio"] = abs_ws_ratio
            metrics["fulltext_ws_ratio"] = full_ws_ratio
            flags["abstract_low_whitespace_ratio"] = abs_ws_ratio < self.min_abs_ws_ratio
            flags["fulltext_low_whitespace_ratio"] = full_ws_ratio < self.min_full_ws_ratio
        except Exception:
            flags["whitespace_ratio_check_failed"] = True

        # 4. ASCII ratio
        try:
            abs_ascii_ratio = self._ascii_ratio(abstract)
            full_ascii_ratio = self._ascii_ratio(fulltext)
            metrics["abstract_ascii_ratio"] = abs_ascii_ratio
            metrics["fulltext_ascii_ratio"] = full_ascii_ratio
            flags["abstract_low_ascii_ratio"] = abs_ascii_ratio < self.min_abs_ascii_ratio
            flags["fulltext_low_ascii_ratio"] = full_ascii_ratio < self.min_full_ascii_ratio
        except Exception:
            flags["ascii_ratio_check_failed"] = True

        # 5. Sentence count
        try:
            abs_sent = self._sentence_count(abstract)
            full_sent = self._sentence_count(fulltext)
            metrics["abstract_sentence_count"] = abs_sent
            metrics["fulltext_sentence_count"] = full_sent
            flags["abstract_low_sentence_count"] = abs_sent < self.min_abs_sentences
            flags["fulltext_low_sentence_count"] = full_sent < self.min_full_sentences
        except Exception:
            flags["sentence_count_failed"] = True

        # 6. Heading markers
        try:
            heading_count = self._heading_marker_count(fulltext)
            metrics["fulltext_heading_markers"] = heading_count
            flags["fulltext_missing_heading_markers"] = heading_count == 0
        except Exception:
            flags["heading_marker_count_failed"] = True

        # 7. Language detection
        try:
            langs = detect_langs(fulltext[:1000])
            if langs:
                lang_obj = langs[0]
                metrics["language"] = lang_obj.lang
                metrics["language_score"] = lang_obj.prob
                flags["language_mismatch_or_low_confidence"] = (
                    lang_obj.lang != self.language or lang_obj.prob < self.lang_prob_threshold
                )
            else:
                flags["language_detection_empty"] = True
        except Exception:
            metrics["language"] = "error"
            metrics["language_score"] = 0.0
            flags["language_detection_failed"] = True

        # 8. ROUGE overlap
        try:
            if abstract and fulltext:
                rouge1 = self.scorer.score(abstract, fulltext[:2000])["rouge1"].recall
                metrics["rouge1_recall"] = rouge1
                flags["low_rouge1_overlap"] = rouge1 < self.rouge_min_recall
        except Exception:
            flags["rouge_scoring_failed"] = True

        # 9. Embedding norm check
        emb = doc.get("abstract_embedding")
        if isinstance(emb, list) and emb:
            try:
                norm = float(np.linalg.norm(np.asarray(emb, dtype=np.float32)))
                metrics["abstract_embedding_norm"] = norm
                flags["abstract_embedding_norm_off"] = abs(norm - self.expected_emb_norm) > self.emb_norm_tol
                flags["abstract_embedding_missing_or_invalid"] = False
            except Exception:
                flags["abstract_embedding_missing_or_invalid"] = True
                flags["abstract_embedding_norm_off"] = False
        else:
            flags["abstract_embedding_missing_or_invalid"] = True
            flags["abstract_embedding_norm_off"] = False

        # Determine status
        CRITICAL_FLAGS = [
            "fulltext_too_short",
            "abstract_embedding_missing_or_invalid",
        ]

        has_critical = any(flags.get(f, False) for f in CRITICAL_FLAGS)
        has_warnings = any(v for k, v in flags.items() if k not in CRITICAL_FLAGS and v)

        if has_critical:
            status = "fail"
        elif has_warnings:
            status = "warn"
        else:
            status = "pass"

        return {
            "status": status,
            "flags": flags,
            "metrics": metrics,
            "examples": {
                "abstract_problem_chars": abstr_ex,
                "fulltext_problem_chars": full_ex,
            },
            "checked_at": checked_at,
        }



    def validate_batch(
        self,
        docs: List[Dict[str, Any]],
        *,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> List[Dict[str, Any]]:
        if not parallel or len(docs) <= 1:
            return [self.validate(d) for d in docs]

        cfg = self._export_cfg()
        with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(partial(_text_worker, cfg_tuple=cfg), docs))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _export_cfg(self) -> Tuple:
        """Serialize config for child processes."""
        return (
            self.min_abs_chars,
            self.min_full_chars,
            self.min_abs_sentences,
            self.min_full_sentences,
            self.min_abs_ws_ratio,
            self.min_full_ws_ratio,
            self.min_abs_ascii_ratio,
            self.min_full_ascii_ratio,
            self.language,
            self.lang_prob_threshold,
            self.rouge_min_recall,
            self.expected_emb_norm,
            self.emb_norm_tol,
        )

    @classmethod
    def _from_cfg(cls, cfg: Tuple) -> "TextValidator":
        return cls(
            min_abs_chars=cfg[0],
            min_full_chars=cfg[1],
            min_abs_sentences=cfg[2],
            min_full_sentences=cfg[3],
            min_abs_ws_ratio=cfg[4],
            min_full_ws_ratio=cfg[5],
            min_abs_ascii_ratio=cfg[6],
            min_full_ascii_ratio=cfg[7],
            language=cfg[8],
            lang_prob_threshold=cfg[9],
            rouge_min_recall=cfg[10],
            expected_emb_norm=cfg[11],
            emb_norm_tol=cfg[12],
        )

    @staticmethod
    def _categorize_characters(text: str) -> Tuple[Dict[str, int], List[Dict[str, str]]]:
        """
        Count character types in the input text and extract examples of unusual ones.

        Returns:
            counts: Dict with keys: valid, corrupted, control, formatting, unassigned, whitespace
            examples: Up to 5 problematic characters with codepoint and Unicode name
        """
        counts = {
            "valid": 0,
            "corrupted": 0,    # U+FFFD replacement char
            "control": 0,      # Excludes \n and \r
            "formatting": 0,   # Invisible Unicode formatting
            "unassigned": 0,   # Unused codepoints
            "whitespace": 0    # Includes space, tabs, newlines
        }
        examples, seen = [], set()

        for ch in text:
            if ch.isspace():
                counts["whitespace"] += 1
                continue

            cat = unicodedata.category(ch)
            if ch == "\uFFFD":
                key = "corrupted"
            elif cat == "Cc":
                if ch in ("\n", "\r"):  # Acceptable in Markdown
                    key = "valid"
                else:
                    key = "control"
            elif cat == "Cf":
                key = "formatting"
            elif cat == "Cn":
                key = "unassigned"
            else:
                key = "valid"

            counts[key] += 1

            if key != "valid" and ch not in seen:
                seen.add(ch)
                examples.append({
                    "char": ch,
                    "codepoint": f"U+{ord(ch):04X}",
                    "name": unicodedata.name(ch, "UNKNOWN"),
                    "category": cat
                })
                if len(examples) >= 5:
                    break

        return counts, examples

    @staticmethod
    def _whitespace_ratio(text: str) -> float:
        non_ws = sum(1 for ch in text if not ch.isspace())
        return non_ws / (len(text) or 1)

    @staticmethod
    def _ascii_ratio(text: str) -> float:
        ascii_letters = sum(ch.isalpha() and ord(ch) < 128 for ch in text)
        return ascii_letters / (len(text) or 1)

    @staticmethod
    def _sentence_count(text: str) -> int:
        return text.count(".") + text.count("!") + text.count("?")

    @staticmethod
    def _heading_marker_count(text: str) -> int:
        import re

        return len(re.findall(r"^#{1,6}\s|\n\d+(\.\d+)*\s", text, flags=re.MULTILINE))

def _text_worker(record: Dict[str, Any], cfg_tuple: Tuple) -> Dict[str, Any]:
    validator = TextValidator._from_cfg(cfg_tuple)
    return validator.validate(record)
# ---------------------------------------------------------------------- #
# Example usage (remove or adapt in production)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import json

    sample = {
        "abstract": "This study develops a new method. Results are promising.",
        "fulltext": "# Introduction\nThis is the body of the article.\n\n## Methods\nLots of details...",
        "abstract_embedding": [0.0] * 1024,
    }
    validator = TextValidator()
    report = validator.validate(sample)
    print(json.dumps(report, indent=2))
