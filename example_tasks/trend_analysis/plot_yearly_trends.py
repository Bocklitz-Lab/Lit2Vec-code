import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
sns.set(style="whitegrid", context="talk")

# Load and sort
agg = pd.read_csv("yearly_trends__.csv").sort_values("year")

# Filter: drop latest year and low-document years
min_raman = 10; min_sers = 10
latest = agg["year"].max()
agg = agg[agg["year"] < latest]

ram = agg[agg["total_raman_docs"] >= min_raman].copy()
ser = agg[agg["total_sers_docs"]  >= min_sers].copy()

# Smooth (3-year centered)
for c in ["SERS_gold_share_in_SERS", "SERS_silver_share_in_SERS", "domain_shift_index"]:
    if c in agg:
        agg[c+"_sm"] = agg[c].rolling(3, center=True, min_periods=1).mean()
        ser[c+"_sm"] = ser[c].rolling(3, center=True, min_periods=1).mean()

# Normalized Raman shares (among 785 + 532 only)
ram["Raman_785_norm"] = ram["Raman_785nm"] / (ram["Raman_785nm"] + ram["Raman_532nm"])
ram["Raman_532_norm"] = ram["Raman_532nm"] / (ram["Raman_785nm"] + ram["Raman_532nm"])
ram["Raman_785_norm_sm"] = ram["Raman_785_norm"].rolling(3, center=True, min_periods=1).mean()
ram["Raman_532_norm_sm"] = ram["Raman_532_norm"].rolling(3, center=True, min_periods=1).mean()

# --- Combined 1x2 Subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

# --- SERS subplot (left) ---
color_gold = "#DAA520"
color_silver = "#708090"

# Bar: counts
ax1.plot(ser["year"], ser["SERS_with_gold"], label="SERS + Gold", color=color_gold)
ax1.plot(ser["year"], ser["SERS_with_silver"], label="SERS + Silver", color=color_silver)
ax1.set_ylabel("Document Count", color="black")
ax1.set_xlabel("Year")
ax1.set_title("SERS Trends: Counts and Share")

# Twin axis: share
ax1b = ax1.twinx()
ax1b.plot(ser["year"], ser["SERS_gold_share_in_SERS_sm"], linestyle="--", color=color_gold, label="Gold Share (smoothed)")
ax1b.plot(ser["year"], ser["SERS_silver_share_in_SERS_sm"], linestyle="--", color=color_silver, label="Silver Share (smoothed)")
ax1b.set_ylabel("Share (smoothed)")
ax1b.set_ylim(0.4, 0.6)


# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1b.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# --- Raman subplot (right) ---
color_785 = "#1f77b4"
color_532 = "#2ca02c"

# Bar: counts
ax2.plot(ram["year"], ram["Raman_785nm"], label="785 nm", color=color_785)
ax2.plot(ram["year"], ram["Raman_532nm"], label="532 nm", color=color_532)
ax2.set_ylabel("Document Count", color="black")
ax2.set_xlabel("Year")
ax2.set_title("Raman Trends: Counts and Share")

# Twin axis: share
ax2b = ax2.twinx()
ax2b.plot(ram["year"], ram["Raman_785_norm_sm"], linestyle="--", color=color_785, label="785 nm Share (smoothed)")
ax2b.plot(ram["year"], ram["Raman_532_norm_sm"], linestyle="--", color=color_532, label="532 nm Share (smoothed)")
ax2b.set_ylabel("Share (smoothed)")
ax2b.set_ylim(0, 1)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2b.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Save combined figure
plt.savefig("chem_trends_combined_2plots.png", dpi=200)
print("Saved plot: chem_trends_combined_2plots.png")
