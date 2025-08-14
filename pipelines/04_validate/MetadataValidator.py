
# Metadata validation
# To ensure structural integrity, completeness, and semantic correctness of the dataset’s bibliographic records, we implemented a strict metadata validation pipeline. This process is designed to detect missing information, incorrect data types, inconsistent dates, malformed author entries, and other structural anomalies that could compromise downstream analyses.
# Each record is expected to conform to a fixed schema version, currently set to 1.0, and to contain a set of required metadata fields, including the title, authors, publication year, venue, publication date, external identifiers, field-of-study annotations, and open-access status. For each field, the validator first confirms that it is present and of the correct type. If a field is missing, a warning is issued and recorded in the diagnostics. If a field is present but of the wrong type, the validator registers a hard error, as type mismatches indicate schema violations. Fields that are structurally correct but empty are also flagged as warnings, since they may still represent incomplete metadata.
# Beyond schema conformity, the validator performs content-based checks. The title is examined for minimal length to detect incomplete or placeholder entries. Author entries are inspected to ensure they are dictionaries containing both a non-empty name and a valid author identifier; any authors failing these conditions are marked as malformed. Publication dates are parsed with timezone awareness, and the year component is compared to the provided publication year; discrepancies between the two are flagged. The validator also checks for future-dated publications, which may indicate data-entry errors.
# The year field itself is validated to ensure that it is an integer within a plausible range, currently bounded between 1800 and one year beyond the current year. The list of fields of study is examined to confirm the presence of at least one entry categorised as “Chemistry,” in line with the dataset’s topical focus. External identifiers, such as DOIs, are checked to ensure at least one non-empty value is present. Publication type entries are verified to be non-empty strings to avoid including incomplete or incorrectly formatted type information.
# For records marked as open access, the validator also checks for the presence of license information in at least one of the recognized license fields (unpaywall_license, crossref_license, openalex_license). Missing license information in such cases is flagged, as it represents a discrepancy between the declared access status and the supporting metadata.
# The validator distinguishes between errors, which indicate definitive schema or content violations, and warnings, which represent potential issues that may require curation but do not necessarily invalidate the record. A record is marked as “fail” if any errors are present, “warn” if only warnings are found, and “pass” if neither are detected.
# The pipeline supports both individual record validation and large-scale batch processing. For large datasets, multiprocessing is used to parallelize the validation process, ensuring efficient runtime performance without sacrificing reproducibility. Each validation report contains the final status, a summary, detailed counts of errors and warnings, the list of detected issues, structured diagnostics describing the type and location of each issue, and a UTC timestamp marking the time of validation. This structured reporting facilitates transparent auditing, enables targeted data cleaning, and ensures that the dataset meets the structural and content standards necessary for long-term reuse.

#!/usr/bin/env python3
"""
Strict validator for Lit2Vec-style metadata records (dict-only version).

* Deep field/type/content checks (authors, externalids, FOS, etc.)
* Safe timezone-aware date handling
* Distinguishes errors vs warnings and returns rich diagnostics
* Fast batch mode via multiprocessing
"""

from __future__ import annotations

import concurrent.futures as cf
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Tuple

from dateutil.parser import parse as parse_date


class MetadataValidator:
    # ------------------------------------------------------------------ #
    # Static configuration
    # ------------------------------------------------------------------ #
    REQ_FIELDS: ClassVar[Dict[str, type]] = {
        "title": str,
        "authors": list,
        "year": int,
        "venue": str,
        "publicationdate": str,
        "externalids": dict,
        # "publicationtypes": list,
        "s2fieldsofstudy": list,
        "isopenaccess": bool,
    }

    LICENSE_KEYS: ClassVar[List[str]] = [
        "unpaywall_license",
        "crossref_license",
        "openalex_license",
    ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def validate(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single record dict and return a rich report."""
        checked_at = datetime.utcnow().isoformat()
        return self._validate_doc(doc, checked_at)

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

        exec_cls = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor

        with exec_cls(max_workers=max_workers) as pool:
            if use_threads:
                return list(pool.map(self.validate, docs))
            else:
                return list(pool.map(_metadata_worker, docs))


    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_doc(self, doc: Dict[str, Any], checked_at: str) -> Dict[str, Any]:
        schema_version = doc.get("schema_version")
        meta = doc.get("metadata")

        if schema_version != "1.0" or meta is None:
            return {
                "status": "fail",
                "summary": "Missing or incompatible top-level fields",
                "metrics": {
                    "schema_version": schema_version,
                    "has_metadata": bool(meta),
                },
                "checked_at": checked_at,
            }

        errors, warnings, diagnostic = self._validate_metadata(meta, doc)

        status = "pass"
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"

        return {
            "status": status,
            "summary": "Metadata field & content validation",
            "metrics": {
                "error_count": len(errors),
                "warning_count": len(warnings),
                "errors": errors,
                "warnings": warnings,
                "diagnostics": diagnostic,
            },
            "checked_at": checked_at,
        }

    def _validate_metadata(
        self, meta: Dict[str, Any], doc: Dict[str, Any]
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        errors: List[str] = []
        warnings: List[str] = []

        diag: Dict[str, Any] = {
            "missing_fields": [],
            "wrong_type_fields": [],
            "empty_fields": [],
            "malformed_authors": [],
            "title_short": False,
            "year_mismatch": False,
            "future_date": False,
            "date_issue": None,
            "no_chemistry_field": False,
            "oa_missing_licenses": [],
        }

        # --- Field presence/type/emptiness -------------------------------- #
        for field, expected_type in self.REQ_FIELDS.items():
            value = meta.get(field, None)

            if value is None:
                warnings.append(f"missing:{field}")
                diag["missing_fields"].append(field)
                continue

            if not isinstance(value, expected_type):
                errors.append(
                    f"type:{field}:{type(value).__name__}->{expected_type.__name__}"
                )
                diag["wrong_type_fields"].append(field)
                continue

            if (
                (expected_type is str and not value.strip())
                or (expected_type is list and not value)
                or (expected_type is dict and not value)
            ):
                warnings.append(f"empty:{field}")
                diag["empty_fields"].append(field)

        # --- Title length check ------------------------------------------- #
        title = meta.get("title", "")
        if isinstance(title, str) and len(title.strip()) < 5:
            warnings.append("title_short")
            diag["title_short"] = True

        # --- Authors validation ------------------------------------------- #
        bad_author_indices: List[int] = []
        for idx, author in enumerate(meta.get("authors", [])):
            if not (
                isinstance(author, dict)
                and isinstance(author.get("name"), str)
                and author["name"].strip()
                and isinstance(author.get("authorId"), (str, int))
                and str(author["authorId"]).strip()
            ):
                bad_author_indices.append(idx)

        if bad_author_indices:
            warnings.append("authors_malformed")
            diag["malformed_authors"] = bad_author_indices

        # --- Date validation ---------------------------------------------- #
        pub_date_raw = meta.get("publicationdate")
        if pub_date_raw:
            try:
                dt = parse_date(pub_date_raw).date()  # extract only the date part
                now = datetime.utcnow().date()        # current date, no timezone issues

                if isinstance(meta.get("year"), int) and dt.year != meta["year"]:
                    warnings.append("year_vs_date")
                    diag["year_mismatch"] = True

                if dt > now:
                    warnings.append("date_in_future")
                    diag["future_date"] = True
            except Exception:
                warnings.append("date_bad_format")
                diag["date_issue"] = "bad_format"

        year = meta.get("year")
        this_year = datetime.utcnow().year
        if not isinstance(year, int) or not (1800 <= year <= this_year + 1):
            warnings.append("year_out_of_range")


        # --- Chemistry FOS check ------------------------------------------ #
        if not any(
            isinstance(fos, dict) and fos.get("category") == "Chemistry"
            for fos in meta.get("s2fieldsofstudy", [])
        ):
            warnings.append("fos_no_chemistry")
            diag["no_chemistry_field"] = True

        # --- External ID check ------------------------------------------- #
        if not any(
            v for v in (meta.get("externalids") or {}).values() if v not in {None, ""}
        ):
            warnings.append("externalids_empty")

        # --- Publication type items -------------------------------------- #
        publicationtypes = meta.get("publicationtypes") or []
        if any(not isinstance(t, str) or not t.strip() for t in publicationtypes):
            warnings.append("pubtypes_bad_item")

        # --- OA license presence ----------------------------------------- #
        if meta.get("isopenaccess", False):
            missing = [k for k in self.LICENSE_KEYS if not doc.get(k)]
            if missing:
                warnings.append("oa_missing_licenses")
                diag["oa_missing_licenses"] = missing

        return errors, warnings, diag

def _metadata_worker(document: Dict[str, Any]) -> Dict[str, Any]:
    validator = MetadataValidator()
    return validator.validate(document)

# ---------------------------------------------------------------------- #
# Example usage (remove or adapt in production)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import json

    docs = [
        {
            "schema_version": "1.0",
            "metadata": {
                "title": "A Study of Stuff",
                "authors": [{"name": "Alice A.", "authorId": "1"}],
                "year": 2025,
                "venue": "ArXiv",
                "publicationdate": "2025-05-10",
                "externalids": {"doi": "10.1000/xyz123"},
                "publicationtypes": ["Journal"],
                "s2fieldsofstudy": [{"category": "Chemistry"}],
                "isopenaccess": True,
            },
            "unpaywall_license": "cc-by",
        },
        # Add more dicts here...
    ]

    validator = MetadataValidator()
    # Parallel by default
    reports = validator.validate_batch(docs)
    print(json.dumps(reports, indent=2))
