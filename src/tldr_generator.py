"""Generate TL;DR summaries from abstracts using the Lit2Vec BART model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

REPO_ID = "Bocklitz-Lab/lit2vec-tldr-bart-model"
DEFAULT_MAX_INPUT_TOKENS = 1024


def _require_dependency(name: str, import_error: Exception) -> None:
    raise ImportError(
        f"Missing optional dependency '{name}'. Install the TL;DR inference stack before use."
    ) from import_error


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            GenerationConfig,
        )
    except ImportError as exc:
        _require_dependency("transformers", exc)

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(REPO_ID)
    generation_config = GenerationConfig.from_pretrained(REPO_ID).to_dict()
    return model, tokenizer, generation_config


@dataclass(slots=True)
class TldrGenerator:
    """Generate a chemistry TL;DR from an abstract."""

    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS

    def generate(self, abstract: str) -> str:
        text = (abstract or "").strip()
        if not text:
            return ""

        model, tokenizer, generation_config = _load_artifacts()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        summary_ids = model.generate(**inputs, **generation_config)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def generate_payload(self, abstract: str) -> dict[str, str]:
        return {"tldr": self.generate(abstract)}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Lit2Vec TL;DR summary from abstract text."
    )
    parser.add_argument("--abstract", help="Abstract text to summarize.")
    parser.add_argument("--abstract-file", help="Path to a text or JSON file containing the abstract.")
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Maximum number of input tokens passed to the tokenizer.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    abstract = _read_text_argument(args)

    generator = TldrGenerator(max_input_tokens=args.max_input_tokens)
    payload = generator.generate_payload(abstract)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
