# Example: trend analysis

This example counts, per publication year, how often papers report particular materials and techniques in their experimental sections. It is the implementation behind Section 3.15.3 of the paper (Figure 12). The published figure was computed on the full study corpus of 582,683 papers. On a smaller reconstructed corpus the curves will be noisier.

## What it measures

- SERS with gold or with silver nanoparticles. Both terms have to occur in the same sentence as SERS.
- Raman spectroscopy with 785 nm or with 532 nm excitation, with a Raman and an instrument or laser term nearby.

Only sections that describe what was done are searched (methods, experimental, instrumentation, characterisation, measurements and results). Introductions, related work and references are skipped. The year comes from the record metadata, with a fallback to years found at the start of the full text.

## Running it

The scripts need the packages in the top-level `requirements.txt` (pandas, DuckDB, matplotlib, seaborn, orjson, python-dateutil). They read records that contain `fulltext` and `metadata`.

```bash
# 1. Flag every paper and aggregate the flags by year (optional second argument: number of processes)
python trend_analysis.py /path/to/records 8

# 2. Plot
python plot_yearly_trends.py
```

Both scripts use the current directory for their files. The first step writes `doc_level_flags.csv` (one row per paper) and `yearly_trends.csv`. If DuckDB is missing, the yearly aggregation is skipped with a warning. The second step reads `yearly_trends.csv` and writes `chem_trends_combined_2plots.png`. The plot omits the most recent year and years with fewer than 10 relevant papers, and shows 3-year centred moving averages for the shares.

## Citation

If you use this code or figures derived from it, please cite the paper (see the main [README](../../README.md#citation)).
