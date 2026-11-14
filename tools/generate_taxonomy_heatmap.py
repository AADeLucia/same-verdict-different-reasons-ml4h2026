"""
Generate taxonomy validation heatmaps and compute statistics for LaTeX table.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("results/reframe/tables_media/qualitative_analysis/annotated")
OUT_DIR  = Path("results/reframe/tables_media")

FP_FILES = [
    DATA_DIR / "rq2_qualitative_validation_fp_annotated.jsonl",
    DATA_DIR / "rq2_qualitative_validation_fp_resampled_annotated.jsonl",
]
FN_FILES = [
    DATA_DIR / "rq2_qualitative_validation_fn_evened_annotated.jsonl",
]
TP_FILES = [
    DATA_DIR / "rq2_qualitative_validation_tp_agrees_evened_annotated.jsonl",
    DATA_DIR / "rq2_qualitative_validation_tp_agrees_resampled_annotated.jsonl",
]
DISAGREES_FILES = [
    DATA_DIR / "rq2_qualitative_validation_disagrees_resampled_annotated.jsonl",
]


# ── Loading helpers ───────────────────────────────────────────────────────────

def load_jsonl(paths):
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return pd.DataFrame(records)


def dedup_keep_labeled(df, label_col):
    """Deduplicate by ID, preferring rows that have the human label filled in."""
    has_label = df[label_col].notna() & (df[label_col] != "")
    df = df.copy()
    df["_has_label"] = has_label.astype(int)
    df = df.sort_values("_has_label", ascending=False).drop_duplicates("ID")
    df = df.drop(columns=["_has_label"])
    return df


def load_and_filter(paths, analysis_cat, label_col):
    df = load_jsonl(paths)
    df = df[df["analysis_category"] == analysis_cat].copy()
    df = dedup_keep_labeled(df, label_col)
    # Keep only rows with human labels
    labeled = df[df[label_col].notna() & (df[label_col] != "")].copy()
    return df, labeled


# ── Build pivot for heatmap ───────────────────────────────────────────────────

def make_heatmap_pivot(labeled_df, llm_col, human_col, llm_order=None, human_order=None):
    """Return (pivot_pct, pivot_count, llm_order_used, human_order_used)."""
    ct = pd.crosstab(labeled_df[llm_col], labeled_df[human_col])
    if llm_order:
        ct = ct.reindex([c for c in llm_order if c in ct.index], axis=0)
    if human_order:
        ct = ct.reindex([c for c in human_order if c in ct.columns], axis=1, fill_value=0)
    ct = ct.fillna(0).astype(int)
    row_sums = ct.sum(axis=1).replace(0, np.nan)
    pct = ct.div(row_sums, axis=0).multiply(100).fillna(0)
    return pct, ct


# ── Custom annotator with % suffix ───────────────────────────────────────────

def fmt_annot(pct_val, count_val):
    return f"{int(round(pct_val))}%\n({int(count_val)})"


def plot_heatmap(ax, pct, counts, title, diag_match=True):
    """Plot a single heatmap on ax."""
    # Build annotation strings combining % and count
    annot = pd.DataFrame(
        [[fmt_annot(pct.iloc[i, j], counts.iloc[i, j])
          for j in range(pct.shape[1])]
         for i in range(pct.shape[0])],
        index=pct.index, columns=pct.columns,
    )

    mask = np.zeros_like(pct.values, dtype=bool)

    sns.heatmap(
        pct,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 8},
    )

    # Highlight diagonal cells (where row label == col label) in light green
    if diag_match:
        row_labels = list(pct.index)
        col_labels = list(pct.columns)
        for i, rl in enumerate(row_labels):
            if rl in col_labels:
                j = col_labels.index(rl)
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (j, i), 1, 1,
                        boxstyle="square,pad=0",
                        linewidth=1.5,
                        edgecolor="#2ca02c",
                        facecolor="#b7e4b7",
                        zorder=0,
                    )
                )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Human Label", fontsize=9)
    ax.set_ylabel("LLM Label", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    fp_all,  fp_labeled  = load_and_filter(FP_FILES,       "FP",       "human_category")
    fn_all,  fn_labeled  = load_and_filter(FN_FILES,       "FN",       "human_category")
    tp_all,  tp_labeled  = load_and_filter(TP_FILES,       "TP",       "human_omissions_aligned")
    # Disagrees: resampled file only; human_category holds FP/FN code, human_dis_category holds optional DIS code
    dis_all, dis_labeled = load_and_filter(DISAGREES_FILES,"Disagrees","human_category")

    # Also load "Agrees" from TP files
    agrees_all = load_jsonl(TP_FILES)
    agrees_all = agrees_all[agrees_all["analysis_category"] == "Agrees"].copy()
    agrees_all = dedup_keep_labeled(agrees_all, "human_omissions_aligned")
    agrees_labeled = agrees_all[
        agrees_all["human_omissions_aligned"].notna() &
        (agrees_all["human_omissions_aligned"] != "")
    ].copy()

    print("=== Data counts ===")
    print(f"FP:       {len(fp_labeled)} labeled / {len(fp_all)} unique sampled")
    print(f"FN:       {len(fn_labeled)} labeled / {len(fn_all)} unique sampled")
    print(f"TP:       {len(tp_labeled)} labeled / {len(tp_all)} unique sampled")
    print(f"Disagrees:{len(dis_labeled)} labeled / {len(dis_all)} unique sampled")
    print(f"Agrees:   {len(agrees_labeled)} labeled / {len(agrees_all)} unique sampled")
    print()

    # ── 2. Exact agreement counts (for table) ─────────────────────────────────
    def exact_agree(df, llm_col, human_col):
        n = len(df)
        exact = (df[llm_col] == df[human_col]).sum()
        return exact, n

    fp_exact, fp_n   = exact_agree(fp_labeled,  "assigned_class", "human_category")
    fn_exact, fn_n   = exact_agree(fn_labeled,  "assigned_class", "human_category")
    tp_exact, tp_n   = exact_agree(tp_labeled,  "assigned_class", "human_omissions_aligned")
    dis_exact, dis_n = exact_agree(dis_labeled, "assigned_class", "human_category")
    ag_exact, ag_n   = exact_agree(agrees_labeled, "assigned_class", "human_omissions_aligned")

    # Disagrees DIS-code stats (human_dis_category present in new schema)
    if "human_dis_category" in dis_labeled.columns:
        dis_with_dis = dis_labeled["human_dis_category"].notna() & (dis_labeled["human_dis_category"] != "")
        dis_dis_n = dis_with_dis.sum()
    else:
        dis_dis_n = 0

    print("=== Exact agreement ===")
    print(f"FP:       {fp_exact}/{fp_n} ({fp_exact/fp_n*100:.0f}%)")
    print(f"FN:       {fn_exact}/{fn_n} ({fn_exact/fn_n*100:.0f}%)")
    print(f"TP:       {tp_exact}/{tp_n} ({tp_exact/tp_n*100:.0f}%)")
    print(f"Disagrees:{dis_exact}/{dis_n} ({dis_exact/dis_n*100:.0f}%) FP/FN | {dis_dis_n}/{dis_n} ({dis_dis_n/dis_n*100:.0f}%) also assigned DIS code")
    print(f"Agrees:   {ag_exact}/{ag_n} ({ag_exact/ag_n*100:.0f}%)")
    print()

    # ── 3. Build pivot tables ─────────────────────────────────────────────────
    fp_pct, fp_ct = make_heatmap_pivot(
        fp_labeled, "assigned_class", "human_category",
        llm_order=["FP01","FP02","FP03","FP04","FP05"],
        human_order=["FP01","FP02","FP03","FP04"],
    )

    fn_pct, fn_ct = make_heatmap_pivot(
        fn_labeled, "assigned_class", "human_category",
        llm_order=["FN01","FN02","FN03","FN04","FN05"],
    )

    tp_pct, tp_ct = make_heatmap_pivot(
        tp_labeled, "assigned_class", "human_omissions_aligned",
        llm_order=["Yes","Partially","No"],
        human_order=["Yes","Partially","No"],
    )

    # Disagrees: rows = LLM FP/FN codes, cols = human FP/FN codes
    all_fp_fn = ["FP01","FP02","FP03","FP04","FP05","FN01","FN02","FN03","FN04","FN05","FN06"]
    dis_llm_order = [c for c in all_fp_fn if c in dis_labeled["assigned_class"].unique()]
    dis_human_order = [c for c in all_fp_fn if c in dis_labeled["human_category"].dropna().unique()]
    dis_pct, dis_ct = make_heatmap_pivot(
        dis_labeled, "assigned_class", "human_category",
        llm_order=dis_llm_order,
        human_order=dis_human_order,
    )

    print("=== Pivot shapes ===")
    print(f"FP  pivot: {fp_pct.shape}")
    print(f"FN  pivot: {fn_pct.shape}")
    print(f"TP  pivot: {tp_pct.shape}")
    print(f"Dis pivot: {dis_pct.shape}")
    print()
    print("FP index:", list(fp_pct.index))
    print("FP cols: ", list(fp_pct.columns))
    print("FN index:", list(fn_pct.index))
    print("FN cols: ", list(fn_pct.columns))
    print("TP index:", list(tp_pct.index))
    print("TP cols: ", list(tp_pct.columns))
    print("Dis index:", list(dis_pct.index))
    print("Dis cols: ", list(dis_pct.columns))

    # ── 4. Plot (2×2: FP, FN, TP, Disagrees) ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_heatmap(axes[0, 0], fp_pct, fp_ct, "FP: Over-penalization")
    plot_heatmap(axes[0, 1], fn_pct, fn_ct, "FN: Missed omission")
    plot_heatmap(axes[1, 0], tp_pct, tp_ct, "TP: Omission Reasoning Alignment")
    plot_heatmap(axes[1, 1], dis_pct, dis_ct, "Disagrees: Cross-dataset")

    fig.suptitle(
        "Per-code Agreement: LLM vs Human Labels",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    out_pdf = OUT_DIR / "rq2_taxonomy_validation_heatmap.pdf"
    out_png = OUT_DIR / "rq2_taxonomy_validation_heatmap.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # ── 5. Print table row values ─────────────────────────────────────────────
    print("\n=== Table row values (for LaTeX) ===")
    print(f"TP / agreement-incomplete: {tp_n} labeled / {len(tp_all)} unique | "
          f"exact {tp_exact}/{tp_n} ({tp_exact/tp_n*100:.0f}%)")
    print(f"FP (in-dataset):           {fp_n} labeled / {len(fp_all)} unique | "
          f"exact {fp_exact}/{fp_n} ({fp_exact/fp_n*100:.0f}%)")
    print(f"FN (in-dataset):           {fn_n} labeled / {len(fn_all)} unique | "
          f"exact {fn_exact}/{fn_n} ({fn_exact/fn_n*100:.0f}%)")
    print(f"Disagrees (cross-dataset): {dis_n} labeled / {len(dis_all)} unique | "
          f"exact FP/FN {dis_exact}/{dis_n} ({dis_exact/dis_n*100:.0f}%) | "
          f"also assigned DIS code {dis_dis_n}/{dis_n} ({dis_dis_n/dis_n*100:.0f}%)")
    print(f"Agrees / agreement-complete: {ag_n} labeled / {len(agrees_all)} unique | "
          f"exact {ag_exact}/{ag_n} ({ag_exact/ag_n*100:.0f}%)")


if __name__ == "__main__":
    main()
