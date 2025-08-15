
# License validation
# To ensure that every record in the dataset contains clear and legally valid usage rights, we developed an automated license validation and agreement-checking pipeline. This process is essential for datasets intended for redistribution and reuse because licensing information in bibliographic metadata is often incomplete, inconsistent, or expressed in non-standard forms across different metadata providers.
# For each record, licensing information is retrieved from three independent sources: Crossref, Unpaywall, and OpenAlex. Since these metadata fields can appear as strings, lists, or nested dictionaries, the validator incorporates a flexible parsing strategy capable of extracting the most explicit license statement available. Once extracted, each license value is normalized to a controlled vocabulary of canonical identifiers, such as cc-by, cc-by-nc-sa, cc0, or public-domain. The normalization step applies both direct mappings for exact textual matches and regular expression mappings for common URL patterns and textual variants of Creative Commons, public domain, and free/open-source licenses. Ambiguous or vague descriptors, such as “other-oa” or “publisher-specific-oa,” are assigned to a non-informative category and excluded from further agreement checks.
# After normalization, the validator compares the three source-specific licenses to determine a final resolved license. If only one source contains an informative license, that license is accepted as the resolved value. When two sources provide informative licenses that agree, the license is accepted and both sources are recorded as the decision basis. If two sources provide different informative licenses, the record is flagged as having a two-way conflict, and the resolved license is recorded as a conflict expression, such as conflict:cc-by_vs_closed. In cases where all three sources agree, the license is accepted with high confidence. However, if all three provide different informative values, the record is flagged as having a multi-way conflict, which is explicitly recorded in the output. Throughout this process, non-informative values are excluded from the agreement assessment, ensuring that agreement is counted only when the matching licenses are both informative and identical.
# The validator assigns a “pass” status to a record when the resolved license is recognized as an open-access license from a curated list, when it is confirmed by at least two independent sources, and when no conflicts are detected between sources. All other cases are assigned a “fail” status, including records with missing data, inconsistent licenses, or licenses that impose restrictive conditions, such as closed, cc-by-nd, or cc-by-nc-nd.
# The pipeline is capable of processing both individual records and large datasets. For large-scale validation tasks, multiprocessing is employed to process records in parallel, which significantly reduces runtime while producing deterministic and reproducible results. Each validation output includes the resolved license, the contributing sources, the raw normalized licenses from each source, a conflict indicator when applicable, and a UTC timestamp indicating when the validation was performed. This structured output enables transparent auditing of the license decision process and allows downstream users to reliably filter records based on compliance status or to investigate specific conflicts.

# SPDX-License-Identifier: MIT
"""
Robust licence validator for Lit2Vec / S2ORC-style records (dict-only).

* Accepts raw Crossref / Unpaywall / OpenAlex payloads (dict, str, list, None)
* Normalises every source to a canonical id (e.g. "cc-by-nc-sa")
* Resolves the three sources into one final decision
* Flags restrictive or conflicting cases
* Fast batch mode via multiprocessing / threading
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Mapping, MutableMapping, Optional, Pattern, Tuple, Union

RAW = Union[str, Mapping[str, Any], List[Any], None]


class LicenseValidator:
    # ------------------------------------------------------------------ #
    # Treat like 'missing' inside _resolve
    NON_INFORMATIVE: ClassVar[set[str]] = {
        "missing",
        "unknown",
        "publisher-specific-oa",
        "other-oa",
    }
    # ------------------------------------------------------------------ #
    # 1. Canonical licence identifiers
    # ------------------------------------------------------------------ #
    CANONICAL: ClassVar[Dict[str, str]] = {
        # Creative Commons
        "cc-by": "cc-by",
        "cc-by-sa": "cc-by-sa",
        "cc-by-nd": "cc-by-nd",
        "cc-by-nc": "cc-by-nc",
        "cc-by-nc-sa": "cc-by-nc-sa",
        "cc-by-nc-nd": "cc-by-nc-nd",
        "cc0": "cc0",
        "public-domain": "public-domain",
        # Free-/FOSS-style licences
        "mit": "mit",
        "bsd": "bsd",
        "gpl-v3": "gpl-v3",
        "agpl-v3": "agpl-v3",
        "elastic-v2": "elastic-v2",
        # Catch-alls / placeholders
        "publisher-specific-oa": "publisher-specific-oa",
        "other-oa": "other-oa",
        "missing": "missing",
        "unknown": "unknown",
    }

    # ------------------------------------------------------------------ #
    # 2. Regex patterns → canonical ids
        # ------------------------------------------------------------------ #
    REGEX_MAP: ClassVar[List[Tuple[Pattern[str], str]]] = [
        # Creative Commons — most specific first
        (re.compile(r"creativecommons\.org/(?:licenses|licences)/by(?:-|/)nc(?:-|/)sa(?:/[\d.]+)?", re.I), "cc-by-nc-sa"),
        (re.compile(r"creativecommons\.org/(?:licenses|licences)/by(?:-|/)nc(?:-|/)nd(?:/[\d.]+)?", re.I), "cc-by-nc-nd"),
        (re.compile(r"creativecommons\.org/(?:licenses|licences)/by(?:-|/)nc(?:/[\d.]+)?", re.I), "cc-by-nc"),
        (re.compile(r"creativecommons\.org/(?:licenses|licences)/by(?:-|/)sa(?:/[\d.]+)?", re.I), "cc-by-sa"),
        (re.compile(r"creativecommons\.org/(?:licenses|licences)/by(?:-|/)nd(?:/[\d.]+)?", re.I), "cc-by-nd"),
        # Generic CC-BY fallback (only if no NC/SA/ND matched above)
        (re.compile(r"creativecommons\.org/(?:licenses|licences)/by(?:/|-)(?:[234]\.\d+)?", re.I), "cc-by"),

        # CC0 / Public domain
        (re.compile(r"creativecommons\.org/(?:publicdomain|public-domain|licenses)/zero(?:/1\.0)?", re.I), "cc0"),
        (re.compile(r"public\.?domain", re.I), "public-domain"),

        # FOSS
        (re.compile(r"gnu\.org/licenses/gpl", re.I), "gpl-v3"),
        (re.compile(r"gnu\.org/licenses/agpl", re.I), "agpl-v3"),
        (re.compile(r"elastic-license", re.I), "elastic-v2"),
        (re.compile(r"mit[-_]license", re.I), "mit"),
    ]


    # ------------------------------------------------------------------ #
    # 3. Licence policy
    # ------------------------------------------------------------------ #
    OPEN_LICENSES: ClassVar[set[str]] = {
        "cc0",
        "public-domain",
        "cc-by",
        "cc-by-sa",
        "cc-by-nc",
        "cc-by-nc-sa",
        # Enable / disable more as needed:
        # "cc-by-nd", "cc-by-nc-nd", "mit", ...
    }
    RESTRICTIVE: ClassVar[set[str]] = {"closed", "cc-by-nd", "cc-by-nc-nd"}

    # ================================================================== #
    # Public API
    # ================================================================== #
    def validate(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Validate licence information for a single record dict."""
        checked_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        cr_raw = self._extract_license(doc.get("crossref_license"))
        upw_raw = self._extract_license(doc.get("unpaywall_license"))
        oa_raw = self._extract_license(doc.get("openalex_license"))

        result = self._resolve(cr_raw, upw_raw, oa_raw)
        result["checked_at"] = checked_at
        return result

    def validate_batch(
        self,
        docs: List[Dict[str, Any]],
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        use_threads: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Validate many records efficiently.
        """
        if not parallel or len(docs) <= 1:
            return [self.validate(d) for d in docs]

        exec_cls = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor

        with exec_cls(max_workers=max_workers) as pool:
            if use_threads:
                return list(pool.map(self.validate, docs))
            else:
                return list(pool.map(_license_worker, docs))


    # ================================================================== #
    # Normalisation + extraction helpers
    # ================================================================== #
    def _normalise(self, raw: RAW) -> str:
        """Convert a raw licence hint into a canonical id."""
        if raw is None:
            return "missing"

        if isinstance(raw, (dict, list)):
            raw = json.dumps(raw, ensure_ascii=False)

        txt = str(raw).strip().lower().replace("‐", "-").rstrip("/")

        # Direct mapping
        if txt in self.CANONICAL:
            return self.CANONICAL[txt]

        # Regex mapping
        for pattern, ident in self.REGEX_MAP:
            if pattern.search(txt):
                return ident

        # Generic CC fallback
        if "creativecommons" in txt:
            return "other-oa"

        return "unknown"

    def _extract_license(self, val: RAW) -> str:
        """Walk common Crossref / Unpaywall / OA structures & return raw string."""
        if val is None:
            return "missing"

        if isinstance(val, str):
            return val

        # Lists (Crossref licence list)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, Mapping):
                    lic = item.get("license") or item.get("rightsUri") or item.get("rights-uri")
                    if lic:
                        return lic
            return "missing"

        # Dicts
        if isinstance(val, Mapping):
            # Direct string value
            if isinstance(val.get("license"), str):
                return val["license"]

            # Crossref nested list
            if isinstance(val.get("license"), list):
                return self._extract_license(val["license"])

            # Unpaywall
            if (loc := val.get("best_oa_location")) and isinstance(loc.get("license"), str):
                return loc["license"]

            # OpenAlex
            if (loc := val.get("primary_location")) and isinstance(loc.get("license"), str):
                return loc["license"]

            # Fallback: any key looking like licence / rights
            for k, v in val.items():
                if isinstance(v, str) and re.search(r"(licen[cs]e|rights)", k, re.I):
                    return v

        return "missing"

    # ================================================================== #
    # Resolution logic
    # ================================================================== #
    def _resolve(self, cr: RAW, upw: RAW, oa: RAW) -> Dict[str, Any]:
        cr_n, upw_n, oa_n = (self._normalise(cr), self._normalise(upw), self._normalise(oa))
        licenses = {"crossref": cr_n, "unpaywall": upw_n, "openalex": oa_n}

        # --- ignore non-informative values --------------------------------
        non_inf = self.NON_INFORMATIVE
        non_missing = [v for v in licenses.values() if v not in non_inf]
        # ------------------------------------------------------------------

        conflict = False
        resolved = "missing"
        source = "none"

        if len(non_missing) == 1:               # one concrete value
            resolved = non_missing[0]
            source = next(k for k, v in licenses.items() if v not in non_inf)

        elif len(non_missing) == 2:             # two concrete values
            items = [(k, v) for k, v in licenses.items() if v not in non_inf]
            (k1, v1), (k2, v2) = items
            if v1 == v2:                        # they agree
                resolved = v1
                source = f"{k1}+{k2}"
            else:                               # they disagree
                resolved = f"conflict:{v1}_vs_{v2}"
                source = f"{k1}+{k2}"
                conflict = True

        elif len(non_missing) == 3:             # all three concrete
            a, b, c = licenses.values()
            if a == b == c:
                resolved = a
                source = "all"
            # elif a == b:  # majority wins
            #     resolved = a
            #     source = "crossref+unpaywall"
            # elif a == c:
            #     resolved = a
            #     source = "crossref+openalex"
            # elif b == c:
            #     resolved = b
            #     source = "unpaywall+openalex"
            else:                               # three-way clash
                resolved = f"conflict:{a}_vs_{b}_vs_{c}"
                source = "mixed"
                conflict = True

        passed = (resolved in self.OPEN_LICENSES) and not conflict and len(non_missing) > 1
        status = "pass" if passed else "fail"

        return {
            "status": status,
            "summary": "Checks licence consistency and open-access compliance.",
            "metrics": {
                "resolved_license": resolved,
                "license_source": source,
                "license_conflict": conflict,
                "input_licenses": licenses,
            },
        }

def _license_worker(record: Dict[str, Any]) -> Dict[str, Any]:
    return LicenseValidator().validate(record)

# ---------------------------------------------------------------------- #
# Quick example – remove or adapt in production
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    sample_records = [
        {
            "crossref_license": [
                {"license": "https://creativecommons.org/licenses/by/4.0"}
            ],
            "unpaywall_license": "cc-by",
            "openalex_license": None,
        },
        {
            "crossref_license": None,
            "unpaywall_license": "closed",
            "openalex_license": "https://creativecommons.org/licenses/by-nc/4.0",
        },
    ]

    lv = LicenseValidator()
    reports = lv.validate_batch(sample_records)  # parallel by default
    print(json.dumps(reports, indent=2))
