
# Schma validation
# A schema-level validation pipeline was developed to ensure that all records in the dataset conform to a consistent structural and semantic format. This process is essential for maintaining data integrity, enabling downstream processing, and ensuring interoperability with external tools and workflows. The validation system is based on a formal JSON Schema definition that describes the complete structure of a Lit2Vec-style record, including both mandatory and optional fields, as well as precise constraints on value types, formats, and array lengths.
# Each record is evaluated against this schema using a standards-compliant Draft 2020-12 JSON Schema validator with full format-checking capabilities. The schema enforces strict typing rules for all properties, such as requiring numerical arrays of fixed length for embeddings, enforcing the presence of core text and metadata fields, and constraining numeric ranges for temporal and bibliometric attributes. Properties that must exist, such as schema_version, corpus_id, metadata, abstract, paragraphs, and embeddings, are explicitly marked as required, while additional or unexpected properties are disallowed. This approach prevents the inclusion of undocumented or inconsistent fields.
# When a record fails validation, the system produces a comprehensive set of diagnostic outputs. Each detected violation is classified according to the relevant schema rule and is mapped to a Boolean flag indicating the specific nature of the error. For example, missing required properties trigger missing_<field> flags, type mismatches generate type_mismatch_<field> flags, and violations of array length constraints result in too_short_<field> or too_long_<field> flags. These flags, along with the original human-readable error messages, are collected into a structured metrics object. This design allows both machine-driven filtering and human-readable error analysis, enabling rapid identification of recurrent schema issues.
# The system determines a “pass” status only when a record satisfies all schema constraints without generating any validation errors. Records that trigger any violations are assigned a “fail” status, along with a count of the errors and their corresponding flags. This binary classification simplifies integration with data ingestion pipelines, where only fully conforming records are eligible for inclusion in the curated dataset.
# The validator is optimized for both single-record checks and large-scale batch processing. For large datasets, the process can be parallelized using either multi-threading or multi-processing to reduce runtime without affecting reproducibility. Each validation result is timestamped to record the exact moment of verification, which supports reproducibility and long-term auditability of validation outcomes. By enforcing strict structural compliance at the schema level, this validation step ensures that the dataset remains consistent, reliable, and immediately usable for downstream research and analysis tasks.


#!/usr/bin/env python3
"""
Schema-level validator for Lit2Vec-style full-document records
with comprehensive Boolean flags for every validation failure.
"""

from __future__ import annotations

import concurrent.futures as cf
from datetime import datetime
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, FormatChecker


class SchemaValidator:
    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, max_errors: int = 50):
        self.max_errors = max_errors

        # -- keep your original schema here ----------------------------- #
        self.schema: Dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "schema_version",
                "corpus_id",
                "metadata",
                "abstract",
                "paragraphs",
                "embeddings",
                "tldr",
                "abstract_embedding",
                "predicted_subfield",
            ],
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string"},
                "corpus_id": {"type": ["string", "integer"]},
                # ---------------- metadata ---------------- #
                "metadata": {
                    "type": "object",
                    "required": ["title", "authors", "venue", "year"],
                    "additionalProperties": False,
                    "properties": {
                        "_id": {"type": "string"},
                        "corpusid": {"type": ["string", "integer"]},
                        "externalids": {"type": "object"},
                        "url": {"type": "string", "format": "uri"},
                        "title": {"type": "string"},
                        "authors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name"],
                                "additionalProperties": False,
                                "properties": {
                                    "authorId": {"type": ["string", "integer", "null"]},
                                    "name": {"type": "string"},
                                },
                            },
                        },
                        "venue": {"type": "string"},
                        "publicationvenueid": {"type": ["string", "null"]},
                        "year": {"type": "integer", "minimum": 1800, "maximum": 2100},
                        "referencecount": {"type": ["integer", "null"], "minimum": 0},
                        "citationcount": {"type": ["integer", "null"], "minimum": 0},
                        "influentialcitationcount": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                        },
                        "isopenaccess": {"type": "boolean"},
                        "publicationdate": {
                            "type": ["string", "null"],
                            "format": "date",
                        },
                        "s2fieldsofstudy": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "publicationtypes": {"type": ["array", "null"]},
                        "journal": {"type": ["object", "null"]},
                    },
                },
                # ---------------- core text --------------- #
                "abstract": {"type": "string"},
                "fulltext": {"type": "string"},
                "paragraphs": {
                    "type": "object",
                    "patternProperties": {r"^\d+P\d+$": {"type": "string"}},
                    "additionalProperties": False,
                },
                # ---------------- embeddings --------------- #
                "embeddings": {
                    "type": "object",
                    "patternProperties": {
                        r"^\d+P\d+$": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 1024,
                            "maxItems": 1024,
                        }
                    },
                    "additionalProperties": False,
                },
                "abstract_embedding": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1024,
                    "maxItems": 1024,
                },
                # ---------------- misc --------------------- #
                "tldr": {"type": "string"},
                "predicted_subfield": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "unpaywall_license": {"type": ["object", "null"]},
                "crossref_license": {"type": ["object", "null"]},
                "openalex_license": {"type": ["object", "null"]},
            },
        }

        self._validator = Draft202012Validator(
            self.schema, format_checker=FormatChecker()
        )

    # ------------------------------------------------------------------ #
    # Flag helpers                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm(path: str) -> str:
        """Path like 'metadata/authors' → 'metadata_authors'."""
        return path.replace("/", "_") if path else "root"

    @classmethod
    def _flag_from_error(cls, validator: str, field: str, msg: str) -> str:
        field_norm = cls._norm(field)

        if validator == "required":
            missing = msg.split("'")[1]  # get the prop name
            return f"missing_{missing}"

        if validator == "type":
            return f"type_mismatch_{field_norm}"

        if validator == "minItems":
            return f"too_short_{field_norm}"

        if validator == "maxItems":
            return f"too_long_{field_norm}"

        if validator in {"minimum", "exclusiveMinimum"}:
            return f"value_below_minimum_{field_norm}"

        if validator in {"maximum", "exclusiveMaximum"}:
            return f"value_above_maximum_{field_norm}"

        if validator == "format":
            return f"invalid_format_{field_norm}"

        if validator == "additionalProperties":
            prop = msg.split("'")[1]
            return f"additional_property_{prop}"

        if validator == "enum":
            return f"enum_error_{field_norm}"

        if validator == "const":
            return f"const_error_{field_norm}"

        if validator == "pattern":
            return f"pattern_violation_{field_norm}"

        if validator == "minLength":
            return f"too_short_string_{field_norm}"

        if validator == "maxLength":
            return f"too_long_string_{field_norm}"

        # Default / fallback:
        return f"{validator}_error_{field_norm}"

    # ------------------------------------------------------------------ #
    # Validate a single document                                         #
    # ------------------------------------------------------------------ #
    def validate(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        checked_at = datetime.utcnow().isoformat()
        errors: List[Dict[str, Any]] = []
        flags: Dict[str, bool] = {}

        for idx, err in enumerate(self._validator.iter_errors(doc)):
            if idx >= self.max_errors:
                errors.append({"message": "... additional errors truncated ..."})
                break

            field_path = "/".join(str(p) for p in err.path) or ""
            errors.append(
                {
                    "message": err.message,
                    "field": field_path or "root",
                    "schema_path": "/".join(map(str, err.schema_path)),
                    "validator": err.validator,
                }
            )

            flag_name = self._flag_from_error(err.validator, field_path, err.message)
            flags[flag_name] = True  # mark flag as hit

        status = "pass" if not errors else "fail"
        summary = (
            "Document conforms to the JSON schema."
            if status == "pass"
            else f"{len(errors)} schema validation error(s) found."
        )

        return {
            "status": status,
            "summary": summary,
            "metrics": {
                "validation_errors": errors,  # original error list
                "flags": flags,               # NEW boolean flags
            },
            "checked_at": checked_at,
        }

    # ------------------------------------------------------------------ #
    # Batch validation (API unchanged)                                   #
    # ------------------------------------------------------------------ #
    def validate_batch(
        self,
        docs: List[Dict[str, Any]],
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        use_threads: bool = False,
    ) -> List[Dict[str, Any]]:
        if not parallel or len(docs) <= 1:
            return [self.validate(d) for d in docs]

        Executor = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor

        def _worker(d: Dict[str, Any]) -> Dict[str, Any]:
            # fresh validator per worker to avoid pickling issues
            return SchemaValidator(self.max_errors).validate(d)

        with Executor(max_workers=max_workers) as pool:
            return list(pool.map(_worker, docs))


# ---------------------------------------------------------------------- #
# Example usage                                                          #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import json, random

    good_doc = {
        "schema_version": "1.0",
        "corpus_id": 123,
        "metadata": {
            "title": "A Valid Paper",
            "authors": [{"name": "Alice"}],
            "venue": "Nature",
            "year": 2024,
        },
        "abstract": "text",
        "paragraphs": {"0P0": "para"},
        "embeddings": {"0P0": [random.random() for _ in range(1024)]},
        "abstract_embedding": [random.random() for _ in range(1024)],
        "tldr": "Good summary",
        "predicted_subfield": {"Chemistry": 0.8},
    }

    bad_doc = {
        "schema_version": "1.0",
        "corpus_id": 124,
        "metadata": {"title": "Bad Paper", "authors": [{"name": "Bob"}]},
        "abstract": "oops",
        "paragraphs": {},
        "embeddings": {},
        "abstract_embedding": [],
        "tldr": "bad",
        "predicted_subfield": {},
        "validation_info": {}  # extra field → triggers additionalProperties
    }

    sv = SchemaValidator(max_errors=20)
    result = sv.validate_batch([good_doc, bad_doc])
    print(json.dumps(result, indent=2))
