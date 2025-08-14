
# Embedding validation
# To guarantee the structural integrity and semantic reliability of the vector representations included in the dataset, we implemented an automated embedding validation framework. This framework is designed for paragraph-level text embeddings generated in a Lit2Vec-style format and stored alongside their source text. The objective is to ensure that each embedding is correctly aligned with its source paragraph, has the expected numeric properties, and remains reproducible when re-encoded with the reference model.
# The validation process begins by retrieving the paragraphs and embeddings fields from each record. Both fields must conform to a dictionary structure in which keys are paragraph identifiers and values are either strings (for the text) or numerical sequences (for the embeddings). Records that do not conform to this schema are immediately flagged as invalid. The validator then checks that every paragraph identifier has a corresponding embedding and vice versa, identifying cases where an embedding is missing, an embedding exists for a non-existent paragraph, or the stored vector does not satisfy the required format. Each embedding must have exactly the expected number of dimensions, consist of finite floating-point values, and be approximately unit-normalized within a tolerance that accounts for floating-point precision differences.
# Once structural checks are complete, the validator assesses reproducibility by spot-checking a subset of paragraphs against their stored embeddings. A small, randomly selected sample of paragraph identifiers is re-encoded using the reference Sentence-Transformer model specified in the validation configuration. These freshly computed vectors are compared to the stored embeddings using cosine similarity. The process calculates the mean and minimum cosine similarity values across the sample, as well as the maximum deviation from perfect alignment. If all sampled pairs exceed the pre-defined similarity threshold, the record is considered to have passed the reproducibility test; otherwise, it fails. This approach ensures that stored embeddings were generated with the intended model and have not been corrupted, reordered, or mismatched with their text.
# A record receives a “pass” status only if all alignment checks succeed, no extra or missing entries are found, all embeddings meet the required shape and normalization criteria, and the reproducibility check confirms agreement within the defined tolerance. Any failure in these conditions results in a “fail” status. For large-scale datasets, the framework supports both sequential and parallel processing modes, using either thread-based or process-based execution depending on resource constraints and GPU sharing considerations. Parallel execution significantly reduces runtime without altering the deterministic nature of the results.
# Each validation run produces a structured output containing the final status, a summary of the validation outcome, detailed metrics on alignment and reproducibility, and a UTC timestamp of when the validation was performed. This output format facilitates downstream filtering, quality auditing, and reproducibility verification for embedding data in the dataset.

#!/usr/bin/env python3
"""
EmbeddingValidator 3.0  –  Dictionary-only, parallel Lit2Vec validator

• Validates paragraph ↔ embedding alignment
• Verifies vector shape, dtype, finite values, and (optionally) unit-norm
• Spot-checks that stored vectors match fresh Sentence-Transformer encodings
• Batch mode with Thread/Process pools for high throughput
"""

from __future__ import annotations

import concurrent.futures as cf
import numbers
import random
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------  constants  -------------------------------- #
PARAGRAPHS_KEY = "paragraphs"
EMBEDDINGS_KEY = "embeddings"

DEFAULT_SEED = 42
DEFAULT_ATOL_COS = 1e-2          # drift tolerance (fp16 / GPU)
DEFAULT_EXPECTED_DIM = 1024


# ---------------------------  helper functions  ----------------------------- #
def _set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Best-effort determinism for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# ---------------------------  validator class  ------------------------------ #
class EmbeddingValidator:
    """
    Validate Lit2Vec-style records that look like:

    {
        "paragraphs": { "p0": "text ...", ... },
        "embeddings": { "p0": [... 1024 floats ...], ... },
        ...
    }
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        expected_dim: int = DEFAULT_EXPECTED_DIM,
        atol_cos: float = DEFAULT_ATOL_COS,
        seed: int = DEFAULT_SEED,
        device: str | None = None,
    ) -> None:
        _set_seeds(seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name: str = model_name
        self.expected_dim: int = expected_dim
        self.atol_cos: float = atol_cos
        self.device: str = device

        self.model: SentenceTransformer = SentenceTransformer(
            model_name, device=self.device
        )

    # ------------------------------------------------------------------------- #
    def validate(self, doc: Dict[str, Any], *, n_compare: int = 10) -> Dict[str, Any]:
        """Validate one Lit2Vec record dictionary."""
        checked_at = datetime.utcnow().isoformat()

        paragraphs = doc.get(PARAGRAPHS_KEY, {})
        embeddings = doc.get(EMBEDDINGS_KEY, {})

        if not isinstance(paragraphs, dict) or not isinstance(embeddings, dict):
            return {
                "status": "fail",
                "summary": "Invalid structure for paragraphs or embeddings",
                "metrics": {
                    "paragraph_count": len(paragraphs) if isinstance(paragraphs, dict) else None,
                    "embedding_count": len(embeddings) if isinstance(embeddings, dict) else None,
                },
                "checked_at": checked_at,
            }

        align = self._alignment_check(paragraphs, embeddings)

        common_ok = set(paragraphs) - set(align["missing"]) - set(align["bad"])
        sample_ids = (
            random.sample(list(common_ok), min(n_compare, len(common_ok)))
            if common_ok
            else []
        )
        determinism = (
            self._compare_to_stored(sample_ids, paragraphs, embeddings)
            if sample_ids
            else {"passed": False}
        )

        status = (
            "pass"
            if not align["missing"] and not align["extra"] and not align["bad"] and determinism["passed"]
            else "fail"
        )

        return {
            "status": status,
            "summary": "Embedding validation complete",
            "metrics": {
                "paragraphs": len(paragraphs),
                "embeddings": len(embeddings),
                "alignment": align,
                "determinism": determinism,
            },
            "checked_at": checked_at,
        }

    # ------------------------------------------------------------------------- #
    def validate_batch(
        self,
        docs: List[Dict[str, Any]],
        *,
        n_compare: int = 10,
        parallel: bool = True,
        max_workers: int | None = None,
        use_threads: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        High-throughput validation.

        Parameters
        ----------
        docs         : list of record dicts
        n_compare    : #paragraphs to re-encode per record
        parallel     : run in parallel (default True)
        max_workers  : size of thread/process pool (default = CPU cores)
        use_threads  : True = ThreadPool (shared GPU); False = ProcessPool (CPU-safe)
                       ⚠️ If running inside Jupyter, threads can be unstable. Use processes.
        """
        if not parallel or len(docs) <= 1:
            return [self._safe_validate(d, n_compare) for d in docs]

        if use_threads:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(lambda d: self._safe_validate(d, n_compare), docs))

        def _worker(args: Tuple[Dict[str, Any], int, str, int, float]) -> Dict[str, Any]:
            d, n, model_name, exp_dim, atol = args
            sub = EmbeddingValidator(
                model_name=model_name, expected_dim=exp_dim, atol_cos=atol, device="cpu"
            )
            return sub._safe_validate(d, n)

        task_params = [
            (d, n_compare, self.model_name, self.expected_dim, self.atol_cos)
            for d in docs
        ]
        with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_worker, task_params))

    # ------------------------------------------------------------------------- #
    def _safe_validate(self, d: Dict[str, Any], n_compare: int) -> Dict[str, Any]:
        try:
            return self.validate(d, n_compare=n_compare)
        except Exception as e:
            return {
                "status": "error",
                "summary": f"Validation failed: {str(e)}",
                "checked_at": datetime.utcnow().isoformat()
            }

    def _is_vector(self, vec: Sequence[Any]) -> bool:
        if not isinstance(vec, (list, np.ndarray)):
            return False
        if len(vec) != self.expected_dim:
            return False
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            return False
        if not np.isfinite(arr).all():
            return False
        if not np.issubdtype(arr.dtype, np.floating):
            return False
        norm = np.linalg.norm(arr)
        if norm == 0 or not np.isclose(norm, 1.0, atol=0.05):
            return False
        return True

    def _alignment_check(
        self, paragraphs: Dict[str, str], embeddings: Dict[str, Sequence[float]]
    ) -> Dict[str, List[str]]:
        para_ids = set(paragraphs)
        emb_ids = set(embeddings)

        missing = sorted(para_ids - emb_ids)
        extra = sorted(emb_ids - para_ids)
        bad = [pid for pid in para_ids & emb_ids if not self._is_vector(embeddings[pid])]
        return {"missing": missing, "extra": extra, "bad": bad}

    def _compare_to_stored(
        self,
        sample_ids: List[str],
        paragraphs: Dict[str, str],
        embeddings: Dict[str, Sequence[float]],
    ) -> Dict[str, Any]:
        texts = [paragraphs[i] for i in sample_ids]
        fresh_vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False  # crucial for thread safety
        )
        stored_vecs = np.asarray([embeddings[i] for i in sample_ids], dtype=np.float32)

        cos = np.diag(cosine_similarity(fresh_vecs, stored_vecs))
        delta = 1.0 - cos
        passed = bool(np.allclose(cos, 1.0, atol=self.atol_cos))

        return {
            "mean_cos": float(cos.mean()),
            "min_cos": float(cos.min()),
            "max_delta": float(delta.max()),
            "threshold": self.atol_cos,
            "passed": passed,
        }


# --------------------------------------------------------------------- #
# Example usage:
# validator = EmbeddingValidator()
# single_result = validator.validate(record)
# batch_results = validator.validate_batch(records, parallel=True, use_threads=False)
