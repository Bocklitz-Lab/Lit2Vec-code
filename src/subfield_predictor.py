"""Predict chemistry subfields from abstract text using the Lit2Vec HF classifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

REPO_ID = "Bocklitz-Lab/lit2vec-subfield-classifier-model"
EMBED_MODEL = "intfloat/e5-large-v2"
TEXT_PREFIX = {"abstract": "abstract: ", "summary": "summary: "}
DEFAULT_THRESHOLD = 0.5


def _require_dependency(name: str, import_error: Exception) -> None:
    raise ImportError(
        f"Missing optional dependency '{name}'. Install the subfield inference stack before use."
    ) from import_error


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, dict[int, str]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        _require_dependency("huggingface_hub", exc)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        _require_dependency("sentence-transformers", exc)

    try:
        from tensorflow import keras
    except ImportError as exc:
        _require_dependency("tensorflow-cpu", exc)

    model_path = hf_hub_download(REPO_ID, filename="mlp_model.h5")
    label_map_path = hf_hub_download(REPO_ID, filename="label_mapping.json")

    with open(label_map_path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    index_to_label = {int(k): v for k, v in mapping["index_to_label"].items()}
    model = keras.models.load_model(model_path, compile=False)
    encoder = SentenceTransformer(EMBED_MODEL, device=os.getenv("LIT2VEC_ENCODER_DEVICE", "cpu"))
    return model, encoder, index_to_label


@dataclass(slots=True)
class SubfieldPredictor:
    """Run multilabel subfield inference for abstracts or summaries."""

    threshold: float = DEFAULT_THRESHOLD
    text_type: str = "abstract"

    def encode_text(self, text: str) -> np.ndarray:
        _, encoder, _ = _load_artifacts()
        prefix = TEXT_PREFIX.get(self.text_type, "")
        embedding = encoder.encode([prefix + text], normalize_embeddings=True)
        return np.asarray(embedding, dtype="float32")

    def predict_scores(self, text: str) -> dict[str, float]:
        text = (text or "").strip()
        if not text:
            return {}

        model, _, index_to_label = _load_artifacts()
        embeddings = self.encode_text(text)
        probabilities = model.predict(embeddings, verbose=0)[0]
        predictions = {
            index_to_label[i]: float(probabilities[i])
            for i in range(len(probabilities))
            if float(probabilities[i]) >= self.threshold
        }
        return dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

    def predict_payload(self, text: str) -> dict[str, dict[str, float]]:
        return {"predicted_subfield": self.predict_scores(text)}


def _read_text_argument(args: argparse.Namespace) -> str:
    if args.abstract:
        return args.abstract
    if args.abstract_file:
        with open(args.abstract_file, "r", encoding="utf-8") as handle:
            content = handle.read()
        return _extract_abstract_from_content(content, args.abstract_file)
    if not os.isatty(0):
        return sys.stdin.read()
    raise ValueError("Provide --abstract, --abstract-file, or pipe abstract text via stdin.")


def _extract_abstract_from_content(content: str, source_name: str) -> str:
    stripped = content.strip()
    if not stripped:
        return ""

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return content

    if isinstance(parsed, dict):
        abstract = parsed.get("abstract")
        if isinstance(abstract, str):
            return abstract
        raise ValueError(f"No string 'abstract' field found in {source_name}.")

    raise ValueError(f"Unsupported abstract file format in {source_name}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict Lit2Vec chemistry subfields from abstract text."
    )
    parser.add_argument("--abstract", help="Abstract text to classify.")
    parser.add_argument("--abstract-file", help="Path to a text file containing the abstract.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum score required to include a label in predicted_subfield.",
    )
    parser.add_argument(
        "--text-type",
        choices=sorted(TEXT_PREFIX),
        default="abstract",
        help="Prefix type used during embedding generation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    abstract = _read_text_argument(args)

    predictor = SubfieldPredictor(
        threshold=args.threshold,
        text_type=args.text_type,
    )
    payload = predictor.predict_payload(abstract)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
