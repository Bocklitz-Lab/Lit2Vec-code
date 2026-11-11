#!/usr/bin/env python3
"""Build the Lit2Vec record-level manifest from the published validation reports.

Reads the validation archives (validation/data_part_*.tar.gz from the Lit2Vec-dataset
release) and writes ONE file, lit2vec_manifest.csv.gz, with one row per record:

  corpus_id, doi, s2ag_release,
  license_crossref, license_unpaywall, license_openalex,   (normalised per-source values)
  resolved_license, license_source, license_status, license_checked_at,
  n_paragraphs, sha256_metadata, sha256_fulltext, sha256_paragraphs,
  sha256_crossref_license, sha256_unpaywall_license, sha256_openalex_license

No text or embeddings are read or written. Only the Python standard library is needed.

Usage:
  python build_manifest.py validation/            # folder with data_part_*.tar.gz
  python build_manifest.py validation/ --out lit2vec_manifest.csv.gz
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

RELEASE = "2024-12-31"
HASHED = ["metadata", "fulltext", "paragraphs",
          "crossref_license", "unpaywall_license", "openalex_license"]
COLUMNS = (["corpus_id", "doi", "s2ag_release",
            "license_crossref", "license_unpaywall", "license_openalex",
            "resolved_license", "license_source", "license_status", "license_checked_at",
            "n_paragraphs"] + [f"sha256_{f}" for f in HASHED])


def row_from_report(name, report):
    v = report.get("validation_info", report)
    c = v["checks"]
    lic = c.get("license_validation", {})
    lm = lic.get("metrics", {})
    inp = lm.get("input_licenses", {})
    idm = c.get("id_validation", {}).get("metrics", {})
    fh = c.get("record_hash", {}).get("field_hashes", {})
    ids = idm.get("corpus_ids", {})
    cid = ids.get("corpus_id") or ids.get("filename_id") or Path(name).stem
    return {
        "corpus_id": str(cid),
        "doi": ";".join(sorted(d.lower() for d in idm.get("doi_set", []) if d)),
        "s2ag_release": RELEASE,
        "license_crossref": inp.get("crossref", ""),
        "license_unpaywall": inp.get("unpaywall", ""),
        "license_openalex": inp.get("openalex", ""),
        "resolved_license": lm.get("resolved_license", ""),
        "license_source": lm.get("license_source", ""),
        "license_status": lic.get("status", ""),
        "license_checked_at": (lic.get("checked_at") or "")[:10],
        "n_paragraphs": idm.get("para_count", ""),
        **{f"sha256_{f}": fh.get(f, {}).get("hash", "") for f in HASHED},
    }


def read_archive(path):
    rows, errors = [], []
    import tarfile
    with tarfile.open(path) as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".json")):
                continue
            try:
                rows.append(row_from_report(m.name, json.load(tar.extractfile(m))))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{Path(path).name}:{m.name}\t{type(e).__name__}: {e}")
    return rows, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Folder with data_part_*.tar.gz (or a single archive).")
    ap.add_argument("--out", default="lit2vec_manifest.csv.gz")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    p = Path(a.input)
    archives = sorted(p.glob("*.tar.gz")) if p.is_dir() else [p]
    if not archives:
        sys.exit(f"no .tar.gz archives found in {p}")

    rows, errors = [], []
    with Pool(min(a.workers, len(archives))) as pool:
        for r, e in pool.imap(read_archive, archives):
            rows += r
            errors += e
            print(f"  {len(rows):,} records", file=sys.stderr, flush=True)

    rows.sort(key=lambda r: int(r["corpus_id"]) if r["corpus_id"].isdigit() else r["corpus_id"])
    with gzip.open(a.out, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    ids = Counter(r["corpus_id"] for r in rows)
    lic = Counter(r["resolved_license"] for r in rows)
    n = len(rows) or 1
    digest = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(json.dumps({
        "records": len(rows),
        "duplicate_corpus_ids": sum(1 for v in ids.values() if v > 1),
        "records_without_doi": sum(1 for r in rows if not r["doi"]),
        "paragraphs_total": sum(int(r["n_paragraphs"] or 0) for r in rows),
        "license_status": dict(Counter(r["license_status"] for r in rows)),
        "resolved_license_pct": {k: round(100 * v / n, 2) for k, v in lic.most_common()},
        "unreadable_reports": len(errors),
        "output": a.out,
        "sha256": digest,
    }, indent=2))
    if errors:
        Path(a.out + ".errors.txt").write_text("\n".join(errors) + "\n")


if __name__ == "__main__":
    main()
