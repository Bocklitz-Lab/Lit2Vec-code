#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import math
import json
import pandas as pd
import numpy as np

# Optional: agreement metrics
try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError:
    cohen_kappa_score = None
    confusion_matrix = None

# ---------- Helpers ----------

def _coerce_rating(x):
    """Coerce rating to float; return NaN if empty/invalid."""
    if pd.isna(x):
        return np.nan
    try:
        # some CSVs can store numbers as strings
        s = str(x).strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def _nonempty_text(x):
    """True if text (revision) is non-empty after stripping."""
    if pd.isna(x):
        return False
    return str(x).strip() != ""

def _count_tokens(s):
    """Rough token/word count for summaries/revisions."""
    if pd.isna(s):
        return 0
    return len(str(s).split())

def _parse_labels_list(s):
    """Parse list-like strings such as "['A', 'B']" safely to a Python list."""
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v]
        return []
    except Exception:
        # If not a list-like repr, return single item as list if non-empty
        st = str(s).strip()
        return [st] if st else []

def _project_stats(df, expert_name):
    """Compute per-expert stats for Field Classification + Summary."""
    df = df.copy()

    df["field_classification_rating"] = df["field_classification_rating"].apply(_coerce_rating)
    df["summary_rating"] = df["summary_rating"].apply(_coerce_rating)

    df["field_classification_revision_nonempty"] = df["field_classification_revision"].apply(_nonempty_text)
    df["summary_revision_nonempty"] = df["summary_revision"].apply(_nonempty_text)

    # Basic counts
    n = len(df)
    n_field_r = df["field_classification_rating"].notna().sum()
    n_summary_r = df["summary_rating"].notna().sum()

    # Distributions
    def dist(series):
        counts = series.value_counts().reindex([1,2,3,4,5], fill_value=0).to_dict()
        return counts

    stats = {
        "expert": expert_name,
        "n_tasks": n,

        "field_rating_count": n_field_r,
        "field_rating_mean": df["field_classification_rating"].mean(),
        "field_rating_median": df["field_classification_rating"].median(),
        "field_rating_std": df["field_classification_rating"].std(),
        "field_rating_pct_ge4": (df["field_classification_rating"] >= 4).mean() if n_field_r > 0 else np.nan,
        "field_rating_dist": json.dumps(dist(df["field_classification_rating"].dropna())),

        "field_revision_nonempty_count": int(df["field_classification_revision_nonempty"].sum()),
        "field_revision_nonempty_pct": df["field_classification_revision_nonempty"].mean() if n > 0 else np.nan,

        "summary_rating_count": n_summary_r,
        "summary_rating_mean": df["summary_rating"].mean(),
        "summary_rating_median": df["summary_rating"].median(),
        "summary_rating_std": df["summary_rating"].std(),
        "summary_rating_pct_ge4": (df["summary_rating"] >= 4).mean() if n_summary_r > 0 else np.nan,
        "summary_rating_dist": json.dumps(dist(df["summary_rating"].dropna())),

        "summary_revision_nonempty_count": int(df["summary_revision_nonempty"].sum()),
        "summary_revision_nonempty_pct": df["summary_revision_nonempty"].mean() if n > 0 else np.nan,

        # Optional: word counts
        "summary_len_words_mean": df["summary"].apply(_count_tokens).replace(0, np.nan).mean(),
        "summary_revision_len_words_mean": df["summary_revision"].apply(_count_tokens).replace(0, np.nan).mean(),
    }

    return stats

def _align_for_agreement(dfA, dfB):
    """
    Align rows across experts for agreement.
    Prefer matching by task_id; fall back to task_title if no overlap on ids.
    Returns merged DataFrame and key used.
    """
    # Try task_id
    merged = dfA.merge(dfB, on="task_id", suffixes=("_A", "_B"), how="inner")
    if len(merged) > 0:
        return merged, "task_id"
    # Fallback: task_title
    merged = dfA.merge(dfB, on="task_title", suffixes=("_A", "_B"), how="inner")
    return merged, "task_title"

def _agreement_metrics(merged, colA, colB, label=""):
    """
    Compute agreement metrics: Cohen's kappa, correlations, exact match, within-1.
    merged: DataFrame with two numeric columns
    """
    out = {"label": label, "n_overlap": 0,
           "kappa": np.nan, "pearson_r": np.nan, "spearman_r": np.nan,
           "exact_match_pct": np.nan, "within_1pt_pct": np.nan}

    a = pd.to_numeric(merged[colA], errors="coerce")
    b = pd.to_numeric(merged[colB], errors="coerce")
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]
    out["n_overlap"] = int(mask.sum())
    if out["n_overlap"] == 0:
        return out

    if cohen_kappa_score is not None:
        try:
            # Need integers for kappa on ordinal categories
            out["kappa"] = float(cohen_kappa_score(a.astype(int), b.astype(int)))
        except Exception:
            out["kappa"] = np.nan

    try:
        out["pearson_r"] = float(a.corr(b, method="pearson"))
    except Exception:
        pass
    try:
        out["spearman_r"] = float(a.corr(b, method="spearman"))
    except Exception:
        pass

    diffs = (a - b).abs()
    out["exact_match_pct"] = float((diffs == 0).mean())
    out["within_1pt_pct"] = float((diffs <= 1).mean())

    return out

def _confusion(merged, colA, colB):
    """
    Confusion matrix for 1..5 ratings if sklearn is present.
    Returns DataFrame or None.
    """
    if confusion_matrix is None:
        return None
    a = pd.to_numeric(merged[colA], errors="coerce").dropna().astype(int)
    b = pd.to_numeric(merged[colB], errors="coerce").dropna().astype(int)
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) == 0:
        return None
    a = aligned.iloc[:,0]
    b = aligned.iloc[:,1]
    labels = [1,2,3,4,5]
    cm = confusion_matrix(a, b, labels=labels)
    df_cm = pd.DataFrame(cm, index=[f"A={i}" for i in labels], columns=[f"B={i}" for i in labels])
    return df_cm

# ---------- Main ----------

def main():
    # Input files produced by your previous export step
    f_jam = "field_summary_annotations_JAMILE.csv"
    f_sara = "field_summary_annotations_SARA.csv"

    if not os.path.exists(f_jam) and not os.path.exists(f_sara):
        raise SystemExit("CSVs not found. Expected files:\n"
                         f" - {f_jam}\n - {f_sara}\nRun the export step first.")

    dfs = {}
    if os.path.exists(f_jam):
        dfs["JAMILE"] = pd.read_csv(f_jam)
    if os.path.exists(f_sara):
        dfs["SARA"] = pd.read_csv(f_sara)

    # Compute per-expert stats
    per_expert_rows = []
    for expert, df in dfs.items():
        per_expert_rows.append(_project_stats(df, expert))

    df_stats = pd.DataFrame(per_expert_rows)
    df_stats.to_csv("validation_stats_per_expert.csv", index=False)

    # Agreement across experts (if both exist)
    agreement_rows = []
    cm_outputs = []

    if "JAMILE" in dfs and "SARA" in dfs:
        dfA = dfs["JAMILE"].copy()
        dfB = dfs["SARA"].copy()

        # Ensure columns exist
        for col in ["task_id", "task_title", "field_classification_rating", "summary_rating"]:
            for d in (dfA, dfB):
                if col not in d.columns:
                    d[col] = np.nan

        merged, key_used = _align_for_agreement(dfA, dfB)

        # Field classification rating agreement
        agr_field = _agreement_metrics(
            merged,
            "field_classification_rating_A",
            "field_classification_rating_B",
            label=f"Field Classification Rating (aligned on {key_used})"
        )
        agreement_rows.append(agr_field)

        cm_field = _confusion(
            merged,
            "field_classification_rating_A",
            "field_classification_rating_B"
        )
        if cm_field is not None:
            cm_field.to_csv(f"confusion_field_rating_{key_used}.csv")

        # Summary rating agreement
        agr_summary = _agreement_metrics(
            merged,
            "summary_rating_A",
            "summary_rating_B",
            label=f"Summary Rating (aligned on {key_used})"
        )
        agreement_rows.append(agr_summary)

        cm_summary = _confusion(
            merged,
            "summary_rating_A",
            "summary_rating_B"
        )
        if cm_summary is not None:
            cm_summary.to_csv(f"confusion_summary_rating_{key_used}.csv")

        df_agree = pd.DataFrame(agreement_rows)
        df_agree.to_csv("validation_agreement_stats.csv", index=False)
    else:
        df_agree = pd.DataFrame(columns=[
            "label", "n_overlap", "kappa", "pearson_r", "spearman_r",
            "exact_match_pct", "within_1pt_pct"
        ])

    # ---------- Markdown report ----------
    lines = []
    lines.append("# Expert Validation Summary\n")

    lines.append("## Per-Expert Statistics\n")
    if not df_stats.empty:
        lines.append(df_stats.to_markdown(index=False))
        lines.append("\n")
    else:
        lines.append("_No per-expert stats available._\n")

    if not df_agree.empty:
        lines.append("## Cross-Expert Agreement\n")
        lines.append(df_agree.to_markdown(index=False))
        lines.append("\n")
        lines.append("If present, confusion matrices were written to CSV files:\n")
        lines.append("- `confusion_field_rating_task_id.csv` or `_task_title.csv`\n")
        lines.append("- `confusion_summary_rating_task_id.csv` or `_task_title.csv`\n")
    else:
        lines.append("## Cross-Expert Agreement\n")
        lines.append("_Agreement could not be computed (only one expert file found or no overlap)._")

    with open("validation_stats_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✓ Wrote:")
    print(" - validation_stats_per_expert.csv")
    print(" - validation_stats_summary.md")
    if "validation_agreement_stats.csv" in os.listdir("."):
        print(" - validation_agreement_stats.csv")
    for fn in os.listdir("."):
        if fn.startswith("confusion_") and fn.endswith(".csv"):
            print(f" - {fn}")

if __name__ == "__main__":
    main()
