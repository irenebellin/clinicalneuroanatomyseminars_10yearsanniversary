#!/usr/bin/env python
"""
cns_r10c_supplement.py  --  Supplementary tables and figures for the sensitivity analyses

Reads the outputs of cns_r10_sensitivity.py (v2 run), the parameter grid from the
first run, and cns_r10b_crosscheck.py, and writes publication-ready tables and figures.
No pipeline imports and no recomputation: it only reads CSV files.

  Tables  (CSV each, one Excel workbook if openpyxl is installed, and each table also
           rendered as an image: PNG 300 dpi + PDF, named TableS1_... etc.)
    Table S1  scenarios vs baseline (fixed / scaled minimum cluster size)
    Table S2  parameter grid summarised by parameter value
    Table S3  dimensionality used for clustering: between-seed stability (and by neighbours)
    Table S4  main-analysis clusters matched to the 10D solution, min. cluster size 10
    Table S5  same, min. cluster size 15
  (numbered in order of first citation in the supplement)
  Figures (PNG 300 dpi + PDF)
    Fig S1    agreement with baseline per scenario, against the seed-to-seed floor
    Fig S2    parameter grid heatmaps and stability by clustering dimensionality
    Fig S3    overlap between main-analysis clusters and the 10D clusters
    Fig S4    2D map coloured by both solutions (copied from the cross-check output)

Usage (from the project folder):
    python3 cns_r10c_supplement.py --v2 sensitivity_out_v2 --grid sensitivity_out \
        --cc crosscheck_out --out supplement_out
"""
import argparse
import os
import re
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Descriptive labels for the main-analysis clusters (from their top terms).
# Replace with the final topic names used in the paper.
THEMES = {
    0: "Genetics and disease", 1: "Open science", 2: "Neurotransmission",
    3: "Glia and cell biology", 4: "MRI acquisition and diffusion",
    5: "Language and aphasia", 6: "Anatomy and tractography",
    7: "Stimulation and neurosurgery", 8: "Memory and consciousness",
    9: "Lesion, disconnection and stroke", 10: "Networks and connectivity",
}
COL_FIXED, COL_SCALED, COL_FLOOR = "#4A6FA5", "#B07D62", "#BBBBBB"
COLLAPSE = 4


# --------------------------------------------------------------------------
def find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit("Cannot find %s in %s" % (name, dirs))


def nice(s):
    m = re.search(r"ge(\d+)_words", s)
    if s.startswith("S1"):
        return "CNSeminars only"
    if s.startswith("S2"):
        return "Excluding Neuroccino"
    if s.startswith("S3") and m:
        return "Excluding short-form (< {:,} words)".format(int(m.group(1)))
    if s.startswith("S5"):
        return "2020\u20132026 only"
    if s.startswith("S4"):
        c = re.search(r"cap(\d+)", s)
        return "Year-balanced, cap %s" % (c.group(1) if c else "")
    return s


def order_key(s):
    m = re.search(r"ge(\d+)_words", s)
    base = {"S1": 1, "S2": 2, "S3": 3, "S5": 4, "S4": 5}[s[:2]]
    return (base, int(m.group(1)) if m else 0)


def pct(x):
    return "%d%%" % round(100 * x)


def g(x):
    return "%g" % round(float(x), 1)


def ari(med, q25, q75):
    return "%.2f (%.2f\u2013%.2f)" % (med, q25, q75)


def save_fig(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  figure", name)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def scenario_long(v2):
    sc = pd.read_csv(find([v2], "sensitivity_scenarios_summary.csv"))
    bal = pd.read_csv(find([v2], "sensitivity_balanced_summary.csv"))
    d = pd.concat([sc, bal], ignore_index=True)
    d["key"] = d.scenario.map(order_key)
    d["label"] = d.scenario.map(nice)
    return d.sort_values(["key", "mcs_mode"]).reset_index(drop=True)


def floor_series(v2):
    return pd.read_csv(find([v2], "sensitivity_noise_floor_summary.csv"), index_col=0)["value"]


def table_s1(d, fl):
    rows = [dict(Scenario="Baseline vs neighbouring seed (floor)", Videos=int(fl["n_videos"]),
                 **{"Clusters, median": g(fl["n_clusters_med"]),
                    "Collapsed runs": pct(fl["collapse_frac"]),
                    "ARI, median (IQR)": ari(fl["ari_nonnoise_med"], fl["ari_nonnoise_q25"], fl["ari_nonnoise_q75"]),
                    "Trajectory rho, median": "%.2f" % fl["traj_rho_med"]})]
    for lab, grp in d.groupby("label", sort=False):
        f = grp[grp.mcs_mode == "fixed_mcs"].iloc[0]
        s = grp[grp.mcs_mode == "scaled_mcs"].iloc[0]
        rows.append({
            "Scenario": lab, "Videos": int(f.n_videos),
            "Clusters, median": "%s / %s" % (g(f.n_clusters_med), g(s.n_clusters_med)),
            "Collapsed runs": "%s / %s" % (pct(f.collapse_frac), pct(s.collapse_frac)),
            "ARI, median (IQR)": "%s / %s" % (ari(f.ari_nonnoise_med, f.ari_nonnoise_q25, f.ari_nonnoise_q75),
                                              ari(s.ari_nonnoise_med, s.ari_nonnoise_q25, s.ari_nonnoise_q75)),
            "Trajectory rho, median": "%.2f / %.2f" % (f.traj_rho_med, s.traj_rho_med)})
    return pd.DataFrame(rows)


def table_match(cc, mcs):
    t = pd.read_csv(find([cc], "crosscheck_cluster_matching_mcs%d.csv" % mcs))
    out = pd.DataFrame({
        "Main-analysis cluster": t.paper_cluster.map(lambda v: "" if pd.isna(v) else str(int(v))),
        "Theme (label)": t.paper_cluster.map(lambda v: "" if pd.isna(v) else THEMES.get(int(v), "")),
        "Videos": t.paper_n.map(lambda v: "" if pd.isna(v) else str(int(v))),
        "Top terms (main analysis)": t.paper_top_terms.fillna(""),
        "Matched cluster": t.new_cluster.map(lambda v: "" if pd.isna(v) else str(int(v))),
        "Matched cluster size": t.new_n.map(lambda v: "" if pd.isna(v) else str(int(v))),
        "Jaccard": t.jaccard.map(lambda v: "" if pd.isna(v) else "%.2f" % v),
        "Trajectory rho": t.trajectory_rho.map(lambda v: "" if pd.isna(v) else "%.2f" % v),
        "Result": t.status.str.capitalize(),
        "Top terms (matched cluster)": t.new_top_terms.fillna("")})
    return out


def table_s4(grid):
    base = grid[grid.tfidf_max == 3000]
    rows = []

    def add(param, value, sub):
        rows.append({"Parameter": param, "Value": value, "Settings": len(sub),
                     "Clusters, median": g(sub.n_clusters.median()),
                     "Clusters, range": "%d\u2013%d" % (sub.n_clusters.min(), sub.n_clusters.max()),
                     "ARI vs baseline, median": "%.2f" % sub.ari_nonnoise.median(),
                     "Collapsed settings": pct((sub.n_clusters <= COLLAPSE).mean())})
    for p, lab in (("nn", "UMAP neighbours"), ("mcs", "Minimum cluster size"),
                   ("ms", "Minimum samples"), ("md", "Minimum distance")):
        for v in sorted(base[p].unique()):
            add(lab, "%g" % v, base[base[p] == v])
    for v in sorted(grid.tfidf_max.unique()):
        sub = grid[(grid.tfidf_max == v) & (grid.mcs == 15) & (grid.ms == 3) & (grid.nn == 20) & (grid.md == 0.05)]
        if len(sub):
            add("TF-IDF vocabulary (single run)", "{:,}".format(int(v)), sub)
    return pd.DataFrame(rows)


def table_s5(dim):
    rows = []
    for k, sub in dim.groupby("ncomp"):
        rows.append({"UMAP dimensions": int(k), "Configurations": len(sub),
                     "Between-seed ARI, median": "%.2f" % sub.within_config_ari_mean.median(),
                     "Between-seed ARI, range": "%.2f\u2013%.2f" % (sub.within_config_ari_mean.min(),
                                                                  sub.within_config_ari_mean.max()),
                     "Configurations with a collapsed seed": pct((sub.collapse_frac > 0).mean())})
    by_nn = dim.pivot_table(index="ncomp", columns="nn", values="within_config_ari_mean", aggfunc="median").round(2)
    by_nn.index.name = "UMAP dimensions"
    by_nn.columns = ["ARI, %d neighbours" % c for c in by_nn.columns]
    t = pd.DataFrame(rows).merge(by_nn.reset_index(), on="UMAP dimensions")
    return t


# --------------------------------------------------------------------------
# Tables rendered as images (journal style: rules only, widths measured from the text)
# --------------------------------------------------------------------------
RESULT_COL = {"Recovered": "#4F7A55", "Partial": "#8F7340"}


def table_image(df, out, name, fontsize=8, max_width=10.5, text_cap=2.7, bold_first_col=False):
    fig0 = plt.figure()
    rend = fig0.canvas.get_renderer()

    def width(txt, bold=False):
        t = fig0.text(0, 0, txt, fontsize=fontsize, fontweight="bold" if bold else "normal")
        w = t.get_window_extent(rend).width / fig0.dpi
        t.remove()
        return w

    def wrap(txt, limit, bold=False):
        out_lines, cur = [], ""
        for word in str(txt).split(" "):
            trial = (cur + " " + word).strip()
            if cur and width(trial, bold) > limit:
                out_lines.append(cur)
                cur = word
            else:
                cur = trial
        out_lines.append(cur)
        return out_lines

    pad = 0.16
    cols = [str(c) for c in df.columns]
    data = [[str(v) for v in r] for r in df.values.tolist()]
    colbold = [(bold_first_col and j == 0) or c == "Result" for j, c in enumerate(cols)]
    nat, mn = [], []
    for j, c in enumerate(cols):
        cells = [c] + [r[j] for r in data]
        bold = [True] + [colbold[j]] * len(data)
        nat.append(max(width(t, b) for t, b in zip(cells, bold)) + pad)
        mn.append(max(width(w, b) for t, b in zip(cells, bold) for w in t.split(" ")) + pad)
    widths = [max(min(n, text_cap), m) for n, m in zip(nat, mn)]
    total = sum(widths)
    if total > max_width:
        spare = [w - m for w, m in zip(widths, mn)]
        need = total - max_width
        if sum(spare) > 0:
            f = min(1.0, need / sum(spare))
            widths = [w - sp * f for w, sp in zip(widths, spare)]
        total = sum(widths)
    lh = fontsize * 1.35 / 72.0
    vpad = 0.06
    head = [wrap(c, w - pad, True) for c, w in zip(cols, widths)]
    body = [[wrap(v, w - pad, colbold[j]) for j, (v, w) in enumerate(zip(r, widths))] for r in data]
    head_h = max(len(h) for h in head) * lh + 2 * vpad
    row_h = [max(len(c) for c in r) * lh + 2 * vpad for r in body]
    H = head_h + sum(row_h)
    plt.close(fig0)

    fig = plt.figure(figsize=(total + 0.1, H + 0.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.05, total + 0.05)
    ax.set_ylim(H + 0.05, -0.05)
    ax.axis("off")
    xs = np.concatenate([[0], np.cumsum(widths)])
    ax.hlines(0, 0, total, color="black", lw=1.3)
    for j, h in enumerate(head):
        ax.text(xs[j] + pad / 2, vpad, "\n".join(h), fontsize=fontsize, fontweight="bold", va="top",
                ha="left", linespacing=1.2)
    y = head_h
    ax.hlines(y, 0, total, color="black", lw=0.8)
    res_j = cols.index("Result") if "Result" in cols else None
    for r, rh, raw in zip(body, row_h, data):
        for j, cell in enumerate(r):
            colr, wt = "black", "normal"
            if j == res_j:
                colr, wt = RESULT_COL.get(raw[j], "#8C5A45"), "bold"
            elif bold_first_col and j == 0:
                wt = "bold"
            ax.text(xs[j] + pad / 2, y + vpad, "\n".join(cell), fontsize=fontsize, va="top", ha="left",
                    linespacing=1.2, color=colr, fontweight=wt)
        y += rh
        ax.hlines(y, 0, total, color="#CCCCCC", lw=0.4)
    ax.hlines(y, 0, total, color="black", lw=1.3)
    save_fig(fig, out, name)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_s1(d, fl, out):
    labels = list(d.drop_duplicates("label").label)
    nmap = d.groupby("label").n_videos.first().to_dict()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ax[0].axvspan(fl["ari_nonnoise_q25"], fl["ari_nonnoise_q75"], color=COL_FLOOR, alpha=.35,
                  label="Seed-to-seed IQR")
    ax[0].axvline(fl["ari_nonnoise_med"], color="#666666", lw=1, ls="--")
    for mode, off, col, lab in (("fixed_mcs", -0.17, COL_FIXED, "Fixed minimum cluster size (15)"),
                                ("scaled_mcs", 0.17, COL_SCALED, "Scaled minimum cluster size")):
        s = d[d.mcs_mode == mode].set_index("label").loc[labels]
        ax[0].errorbar(s.ari_nonnoise_med, y + off,
                       xerr=[s.ari_nonnoise_med - s.ari_nonnoise_q25, s.ari_nonnoise_q75 - s.ari_nonnoise_med],
                       fmt="o", color=col, ms=5, lw=1.4, label=lab)
        ax[1].scatter(s.traj_rho_med, y + off, color=col, s=28, label=lab)
    ax[1].axvline(fl["traj_rho_med"], color="#666666", lw=1, ls="--", label="Seed-to-seed median")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(["%s (n = %d)" % (l, nmap[l]) for l in labels], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 1)
    ax[1].set_xlim(0, 1)
    ax[0].set_xlabel("Adjusted Rand index vs same-seed baseline\n(median and IQR over seeds)", fontsize=8)
    ax[1].set_xlabel("Spearman correlation of yearly prevalence\n(matched clusters, median)", fontsize=8)
    ax[0].legend(fontsize=7, loc="lower right")
    for a in ax:
        a.tick_params(labelsize=8)
    plt.tight_layout()
    save_fig(fig, out, "figS1_scenarios")


def heat(ax, M, rows, cols, title, fmt, cmap, mark=None):
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=9)
    vmax = np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt % v, ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * vmax else "black")
    if mark:
        ax.add_patch(Rectangle((mark[1] - .5, mark[0] - .5), 1, 1, fill=False, ec="black", lw=2))
    return im


def fig_s2(grid, dim, out):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    sub = grid[(grid.tfidf_max == 3000) & (grid.ms == 3) & (grid.md == 0.05)]
    nns, mcss = [10, 15, 20, 30], [8, 10, 15, 20, 30]
    C = np.full((len(nns), len(mcss)), np.nan)
    A = np.full_like(C, np.nan)
    for i, n in enumerate(nns):
        for j, m in enumerate(mcss):
            r = sub[(sub.nn == n) & (sub.mcs == m)]
            if len(r):
                C[i, j], A[i, j] = r.n_clusters.iloc[0], r.ari_nonnoise.iloc[0]
    mark = (nns.index(20), mcss.index(15))
    heat(ax[0], C, nns, mcss, "Number of clusters (seed 42)", "%d", "Greys", mark)
    heat(ax[1], A, nns, mcss, "ARI vs main-analysis partition", "%.2f", "Blues", mark)
    for a in ax[:2]:
        a.set_xlabel("Minimum cluster size", fontsize=8)
        a.set_ylabel("UMAP neighbours", fontsize=8)
    cols = {10: "#7A9E7E", 15: "#B0956A", 20: "#4A6FA5", 30: "#B07D62"}
    rng = np.random.RandomState(0)
    for k, nc in enumerate((2, 5, 10)):
        s = dim[dim.ncomp == nc]
        for _, r in s.iterrows():
            ax[2].scatter(k + rng.uniform(-.18, .18), r.within_config_ari_mean, color=cols[int(r.nn)], s=26,
                          alpha=.85, edgecolor="black" if (r.nn == 20 and r.mcs == 15) else "none",
                          linewidth=1.4)
        ax[2].hlines(s.within_config_ari_mean.median(), k - .3, k + .3, color="black", lw=1.6)
    for n, c in cols.items():
        ax[2].scatter([], [], color=c, s=26, label="%d neighbours" % n)
    ax[2].scatter([], [], color="white", edgecolor="black", s=26, linewidth=1.4, label="Main-text setting\n(20 neighbours, size 15)")
    ax[2].set_xticks([0, 1, 2])
    ax[2].set_xticklabels(["2", "5", "10"], fontsize=8)
    ax[2].set_xlabel("UMAP dimensions used for clustering", fontsize=8)
    ax[2].set_ylabel("Between-seed ARI within configuration\n(bar = median)", fontsize=8)
    ax[2].set_ylim(0, 1)
    ax[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax[2].tick_params(labelsize=8)
    plt.tight_layout()
    save_fig(fig, out, "figS2_parameters")


def fig_s3(cc, out):
    lab = pd.read_csv(find([cc], "crosscheck_labels.csv"))
    ct = pd.crosstab(lab.paper_cluster, lab.new_cluster_mcs10)
    rows = [r for r in sorted(ct.index) if r >= 0] + ([-1] if -1 in ct.index else [])
    cols_all = [c for c in ct.columns if c >= 0]
    best = []
    for r in [r for r in rows if r >= 0]:
        c = ct.loc[r, cols_all].idxmax()
        if c not in best:
            best.append(c)
    cols = best + [c for c in cols_all if c not in best] + ([-1] if -1 in ct.columns else [])
    M = ct.loc[rows, cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["unclustered" if c == -1 else str(c) for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["Unclustered" if r == -1 else "%d  %s" % (r, THEMES.get(r, "")) for r in rows], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                ax.text(j, i, "%d" % M[i, j], ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 * M.max() else "black")
    ax.set_xlabel("Cluster in the 10-dimensional solution (30 neighbours, minimum cluster size 10)", fontsize=8)
    ax.set_ylabel("Main-analysis cluster", fontsize=8)
    plt.tight_layout()
    save_fig(fig, out, "figS3_overlap")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="sensitivity_out_v2", help="output folder of cns_r10_sensitivity.py (v2)")
    ap.add_argument("--grid", default="sensitivity_out", help="folder holding sensitivity_param_grid.csv")
    ap.add_argument("--cc", default="crosscheck_out", help="output folder of cns_r10b_crosscheck.py")
    ap.add_argument("--out", default="supplement_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = scenario_long(a.v2)
    fl = floor_series(a.v2)
    grid = pd.read_csv(find([a.grid, a.v2], "sensitivity_param_grid.csv"))
    dim = pd.read_csv(find([a.v2], "sensitivity_dim_grid.csv"))

    keep = ["Main-analysis cluster", "Theme (label)", "Videos", "Matched cluster",
            "Matched cluster size", "Jaccard", "Trajectory rho", "Result"]
    short = {"Main-analysis cluster": "Cluster", "Theme (label)": "Theme",
             "Matched cluster size": "Matched size", "Trajectory rho": "Trajectory \u03c1"}
    # numbered in order of first citation in the supplement
    tables = [("Table_S1_scenarios", table_s1(d, fl)),
              ("Table_S2_parameter_grid", table_s4(grid)),
              ("Table_S3_dimensionality", table_s5(dim)),
              ("Table_S4_cluster_matching_mcs10", table_match(a.cc, 10)),
              ("Table_S5_cluster_matching_mcs15", table_match(a.cc, 15))]
    for name, t in tables:
        t.to_csv(os.path.join(a.out, name + ".csv"), index=False)
        print("  table", name)
    try:
        with pd.ExcelWriter(os.path.join(a.out, "supplementary_tables.xlsx")) as xw:
            for name, t in tables:
                t.to_excel(xw, sheet_name=name[:31], index=False)
        print("  workbook supplementary_tables.xlsx")
    except ImportError:
        print("  (openpyxl not installed: CSV tables only)")

    # tables as images (long term lists stay in the CSV / Excel versions)
    for name, t in tables:
        img = t
        if "cluster_matching" in name:
            img = t[keep].rename(columns=short)
        if name.startswith("Table_S1"):
            img = t.rename(columns={"Trajectory rho, median": "Trajectory \u03c1, median"})
        table_image(img, a.out, name.replace("Table_S", "TableS", 1).split("_")[0],
                    bold_first_col=name.startswith("Table_S1"))

    fig_s1(d, fl, a.out)
    fig_s2(grid, dim, a.out)
    fig_s3(a.cc, a.out)

    # Fig S4: the 2D map written by cns_r10b_crosscheck.py (copied so all figures sit together)
    src = os.path.join(a.cc, "fig_crosscheck_map.png")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(a.out, "figS4_map.png"))
        print("  figure figS4_map (copied from %s)" % src)
        pdf = os.path.join(a.cc, "fig_crosscheck_map.pdf")
        if os.path.exists(pdf):
            shutil.copy(pdf, os.path.join(a.out, "figS4_map.pdf"))
    else:
        print("  WARNING: %s not found; re-run cns_r10b_crosscheck.py to create Fig S4" % src)
    print("Done. Outputs in", a.out)


if __name__ == "__main__":
    main()