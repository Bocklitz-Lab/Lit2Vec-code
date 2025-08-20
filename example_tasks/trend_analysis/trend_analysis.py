

#!/usr/bin/env python3

import os, re, glob, sys, csv
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse as parse_dt
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import orjson as jsonlib  
except Exception:
    import json as jsonlib

YEAR_MIN, YEAR_MAX = 1980, datetime.now().year

# ----------- Regex (compiled once per process) -----------
PAT_SERS   = re.compile(r"\bSERS\b", re.I)
PAT_RAMAN  = re.compile(r"\braman\b", re.I)

PAT_GOLD   = re.compile(r"\b(gold|au|auNPs?|gold(-|\s)?nano(particle|rod|sphere)s?)\b", re.I)
PAT_SILVER = re.compile(r"\b(silver|ag|agNPs?|silver(-|\s)?nano(particle|rod|sphere)s?)\b", re.I)

USAGE_TERMS = re.compile(
    r"\b(used|employed|measured|recorded|acquired|collected|excited|excitation|laser|diode|"
    r"λ(?:\s*[_\-]?\s*)?(ex|exc)|instrument|spectrometer|setup|apparatus)\b",
    re.I
)

# Section header recognizers (line-anchored)
SECTION_ANY = re.compile(
    r'^\s*(\d+(\.\d+)*)?[\s.:–-]*'
    r'(abstract|introduction|background|related work|literature review|previous work|review|'
    r'state of the art|methods?|materials and methods|methodology|experimental(?: section)?|'
    r'instrumentation|apparatus|setup|characterization|measurement[s]?|'
    r'results(?: and discussion)?|discussion|conclusion[s]?|summary|'
    r'references|bibliography|acknowledg(?:e)?ments?|supplementary(?: information)?|appendix)'
    r'\s*$',
    re.I | re.M
)
SECTION_ALLOWED = re.compile(
    r'^\s*(\d+(\.\d+)*)?[\s.:–-]*'
    r'(methods?|materials and methods|methodology|experimental(?: section)?|'
    r'instrumentation|apparatus|setup|characterization|measurement[s]?|'
    r'results(?: and discussion)?)\s*$',
    re.I | re.M
)
SECTION_EXCLUDED = re.compile(
    r'^\s*(\d+(\.\d+)*)?[\s.:–-]*'
    r'(abstract|introduction|background|related work|literature review|previous work|review|'
    r'state of the art|conclusion[s]?|summary|references|bibliography|'
    r'acknowledg(?:e)?ments?|supplementary(?: information)?|appendix)\s*$',
    re.I | re.M
)

def extract_relevant_text(text: str) -> str:
    """Keep Methods/Experimental/Instrumentation/Results blocks; exclude Intro/Related/Refs/etc."""
    if not text:
        return ""
    matches = list(SECTION_ANY.finditer(text))
    if not matches:
        return text  # no structure detected
    pieces = []
    for i, m in enumerate(matches):
        hdr = m.group(0)
        start = m.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        if SECTION_ALLOWED.match(hdr) and not SECTION_EXCLUDED.match(hdr):
            pieces.append(block)
    return "\n".join(pieces) if pieces else text  # fallback to full text if we failed

def nm_pat(nm):
    nm = str(nm)
    um = f"0.{nm[:1]}{nm[1:]}" if len(nm) == 3 else None
    parts = [fr"{nm}\s*nm", fr"{nm}\s*-\s*nm", fr"{nm}nm"]
    if um:
        parts += [fr"{um}\s*µm", fr"{um}\s*micron", fr"{um}\s*μm", fr"{um}µm"]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.I)

PAT_785 = nm_pat(785)
PAT_532 = nm_pat(532)

# Quick substring prefilters (applied to relevant text)
PREFILTERS = {
    "SERS_with_gold": ("sers", "gold| au"),
    "SERS_with_silver": ("sers", "silver| ag"),
    "Raman_785nm": ("785",),
    "Raman_532nm": ("532",),
}

def normalize(text: str) -> str:
    if not text: return ""
    return text.replace("\u00b5", "µ").replace("μ", "µ").replace("–", "-").replace("—", "-")

def sers_with_metal_sentence_only(text, metal_pat):
    """Require SERS and metal to appear in the SAME sentence (avoid loose co-mentions)."""
    if not (PAT_SERS.search(text) and metal_pat.search(text)):
        return False
    for sent in re.split(r"[.!?]\s+", text):
        if PAT_SERS.search(sent) and metal_pat.search(sent):
            return True
    return False

def wavelength_with_raman(text, wl_pat):
    """Count wavelength only when near 'Raman' and usage-like terms."""
    # Sentence-level first
    for sent in re.split(r"[.!?]\s+", text):
        if wl_pat.search(sent) and PAT_RAMAN.search(sent) and USAGE_TERMS.search(sent):
            return True
    # Window fallback ±80 chars
    for m in wl_pat.finditer(text):
        s, e = m.span()
        window = text[max(0, s-80): min(len(text), e+80)]
        if PAT_RAMAN.search(window) and USAGE_TERMS.search(window):
            return True
    return False

def extract_year(md, data, fulltext_head):
    y = (md or {}).get("year")
    if isinstance(y, int) and YEAR_MIN <= y <= YEAR_MAX:
        return y
    pubdate = (md or {}).get("publicationdate")
    if pubdate:
        try:
            y = parse_dt(str(pubdate), fuzzy=True).year
            if YEAR_MIN <= y <= YEAR_MAX: return y
        except Exception:
            pass
    for key in ("year","pub_year","publication_year","date","published","publicationDate"):
        v = data.get(key)
        if v:
            try:
                y = parse_dt(str(v), fuzzy=True).year
                if YEAR_MIN <= y <= YEAR_MAX: return y
            except Exception:
                try:
                    yi = int(str(v)[:4])
                    if YEAR_MIN <= yi <= YEAR_MAX: return yi
                except Exception:
                    pass
    years = [int(y) for y in re.findall(r"\b(19[89]\d|20\d{2}|21\d{2})\b", fulltext_head)]
    years = [y for y in years if YEAR_MIN <= y <= YEAR_MAX]
    return max(years) if years else None

def quick_prefilter(lc_text):
    flags = {}
    for k, needles in PREFILTERS.items():
        present = True
        for needle in needles:
            sub_present = any(n.strip() in lc_text for n in needle.split("|"))
            if not sub_present:
                present = False; break
        flags[k] = present
    return flags

def process_file(path):
    # RETURNS: None (skip) OR tuple of values for CSV row
    try:
        with open(path, "rb") as f:
            raw = f.read()
        data = jsonlib.loads(raw)
    except Exception:
        return None

    fulltext = normalize((data.get("fulltext") or ""))
    if not fulltext.strip():
        return None

    md = data.get("metadata") or {}
    head = (fulltext[:4000])  # enough for year & quick matches
    year = extract_year(md, data, head)
    if not (year and YEAR_MIN <= year <= YEAR_MAX):
        return None

    # Keep only sections describing what was DONE (not related work/refs)
    relevant = extract_relevant_text(fulltext)
    if not relevant.strip():
        return None

    # Domain flags
    has_raman = bool(PAT_RAMAN.search(relevant))
    has_sers  = bool(PAT_SERS.search(relevant))

    # Quick prefilters on relevant portion
    lc = relevant.lower()
    pf = quick_prefilter(lc)

    # Compute showcases with tighter logic
    sers_gold   = sers_with_metal_sentence_only(relevant, PAT_GOLD)   if pf["SERS_with_gold"]   else False
    sers_silver = sers_with_metal_sentence_only(relevant, PAT_SILVER) if pf["SERS_with_silver"] else False
    r785 = wavelength_with_raman(relevant, PAT_785) if pf["Raman_785nm"] else False
    r532 = wavelength_with_raman(relevant, PAT_532) if pf["Raman_532nm"] else False

    ext = md.get("externalids") or {}
    return (
        os.path.basename(path),
        year,
        md.get("title"),
        md.get("venue"),
        ext.get("DOI"),
        ext.get("PubMed"),
        ext.get("PubMedCentral"),
        bool(md.get("isopenaccess")),
        int(has_raman), int(has_sers),
        int(bool(sers_gold)),
        int(bool(sers_silver)),
        int(bool(r785)),
        int(bool(r532)),
    )

def iter_files(folder):
    # glob non-recursively; switch to **/*.json for nested
    yield from glob.iglob(str(Path(folder) / "*.json"))

def main(folder, out_csv="doc_level_flags.csv", workers=None, chunksize=200):
    fields = [
        "file","year","title","venue","doi","pubmed","pmcid","is_open_access",
        "has_raman","has_sers",
        "SERS_with_gold","SERS_with_silver","Raman_785nm","Raman_532nm"
    ]
    paths = list(iter_files(folder))
    total = len(paths)
    if total == 0:
        print("No JSON files found.")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)  # leave 1 core free

    print(f"Processing {total:,} files with {workers} processes ...")

    written = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(fields)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for i in range(0, total, chunksize):
                batch = paths[i:i+chunksize]
                for p in batch:
                    futures.append(ex.submit(process_file, p))
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        w.writerow(res)
                        written += 1
                futures.clear()

    print(f"Wrote {written:,} rows to {out_csv}")

    # ---- Aggregate with DuckDB (fast, memory-light) ----
    try:
        import duckdb
        con = duckdb.connect(database=":memory:")
        con.execute(f"""
            CREATE TABLE docs AS
            SELECT * FROM read_csv_auto('{out_csv}', header=true, sample_size=-1);
        """)
        agg = con.execute("""
            SELECT
                year::INT AS year,
                COUNT(*)                   AS total_docs,
                SUM(has_raman)             AS total_raman_docs,
                SUM(has_sers)              AS total_sers_docs,
                SUM(SERS_with_gold)        AS SERS_with_gold,
                SUM(SERS_with_silver)      AS SERS_with_silver,
                SUM(Raman_785nm)           AS Raman_785nm,
                SUM(Raman_532nm)           AS Raman_532nm,

                -- overall shares (per all docs)
                SUM(SERS_with_gold)   * 1.0 / COUNT(*) AS SERS_with_gold_share_overall,
                SUM(SERS_with_silver) * 1.0 / COUNT(*) AS SERS_with_silver_share_overall,
                SUM(Raman_785nm)      * 1.0 / COUNT(*) AS Raman_785nm_share_overall,
                SUM(Raman_532nm)      * 1.0 / COUNT(*) AS Raman_532nm_share_overall,

                -- domain shares (more meaningful)
                CASE WHEN SUM(has_sers)  > 0 THEN SUM(SERS_with_gold) *1.0 / SUM(has_sers)  END AS SERS_gold_share_in_SERS,
                CASE WHEN SUM(has_sers)  > 0 THEN SUM(SERS_with_silver)*1.0 / SUM(has_sers)  END AS SERS_silver_share_in_SERS,
                CASE WHEN SUM(has_raman) > 0 THEN SUM(Raman_785nm)    *1.0 / SUM(has_raman) END AS Raman_785_share_in_Raman,
                CASE WHEN SUM(has_raman) > 0 THEN SUM(Raman_532nm)    *1.0 / SUM(has_raman) END AS Raman_532_share_in_Raman,

                -- domain shift index
                CASE WHEN SUM(has_raman) > 0
                     THEN (SUM(Raman_785nm)*1.0 - SUM(Raman_532nm)) / SUM(has_raman)
                END AS domain_shift_index
            FROM docs
            GROUP BY year
            ORDER BY year
        """).df()
        agg.to_csv("yearly_trends.csv", index=False)
        print("Saved yearly_trends.csv")
    except Exception as e:
        print(f"[WARN] DuckDB aggregation skipped ({e}). You can aggregate later with pandas/Polars.")
        return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trend_analysis.py /path/to/json_folder [optional_workers]")
        sys.exit(1)
    folder = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(folder, workers=workers)





