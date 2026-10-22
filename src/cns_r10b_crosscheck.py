#!/usr/bin/env python
"""
cns_r10b_crosscheck.py  --  Do the paper's 11 themes survive a more stable clustering?

Clusters the videos in 10 UMAP dimensions (n_neighbors=30 by default) over several
seeds, then compares with the paper's 2D solution (baseline_labels.csv written by
cns_r10_sensitivity.py). The 2D map used for plotting is unchanged: only the
cluster labels differ.

Outputs (in --out):
  crosscheck_runs.csv                 every run vs the paper clusters
  crosscheck_summary.csv              per min_cluster_size: stability and agreement
  crosscheck_cluster_matching_mcs*.csv  paper cluster -> best-matching new cluster
                                        (overlap, top terms, yearly-trend correlation)
  crosscheck_labels.csv               per-video labels (paper and new)
  fig_crosscheck_map.png              same 2D map, coloured by paper vs new clusters

Run next to cns_r10_sensitivity.py, cns_analysis_pipeline.py, cns_r2_embedding.py.
"""
import argparse
import io
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from cns_r2_embedding import get_embeddings, run_umap
from cns_r10_sensitivity import (COLLAPSE, cluster, compare_labels, full_compare,
                                 load_docs, prevalence)


def top_terms(emb, vocab, labels, cid, n=8):
    mask = labels.values == cid
    diff = emb[mask].mean(axis=0) - emb.mean(axis=0)
    return ", ".join(str(vocab[i]).replace("_", " ") for i in np.argsort(-diff)[:n])


def medoid_seed(runs):
    seeds = sorted(runs)
    k = len(seeds)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            v = compare_labels(runs[seeds[i]], runs[seeds[j]])["ari_nonnoise"]
            M[i, j] = M[j, i] = 0.0 if np.isnan(v) else v
    mean_pair = M.sum() / max(1, k * (k - 1))
    return seeds[int(np.argmax(M.sum(axis=1)))], mean_pair


def matching_table(base, new, years, emb, vocab, min_docs=3, min_years=5):
    bc, nc = sorted(set(base[base >= 0])), sorted(set(new[new >= 0]))
    J = np.zeros((len(bc), len(nc)))
    for i, x in enumerate(bc):
        bi = set(base.index[base == x])
        for j, y in enumerate(nc):
            nj = set(new.index[new == y])
            J[i, j] = len(bi & nj) / float(len(bi | nj))
    r, c = linear_sum_assignment(-J)
    pairs = dict(zip(r, c))
    pb, _ = prevalence(base, years)
    pn, tn = prevalence(new, years)
    yrs = [y for y in pb.columns if y in pn.columns and tn[y] >= min_docs]
    rows, used = [], set()
    for i, x in enumerate(bc):
        row = dict(paper_cluster=x, paper_n=int((base == x).sum()),
                   paper_top_terms=top_terms(emb, vocab, base, x))
        if i in pairs:
            j = pairs[i]
            used.add(j)
            rho = np.nan
            if len(yrs) >= min_years:
                rho = spearmanr(pb.loc[x, yrs], pn.loc[nc[j], yrs])[0]
            row.update(new_cluster=nc[j], new_n=int((new == nc[j]).sum()),
                       jaccard=J[i, j], trajectory_rho=rho,
                       new_top_terms=top_terms(emb, vocab, new, nc[j]),
                       status="recovered" if J[i, j] >= 0.5 else
                       ("partial" if J[i, j] >= 0.3 else "not recovered"))
        else:
            row.update(status="not recovered")
        rows.append(row)
    for j, y in enumerate(nc):
        if j not in used:
            rows.append(dict(new_cluster=y, new_n=int((new == y).sum()),
                             new_top_terms=top_terms(emb, vocab, new, y),
                             status="new cluster (no paper counterpart)"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", default="sensitivity_out_v2/baseline_labels.csv")
    ap.add_argument("--out", default="crosscheck_out")
    ap.add_argument("--ncomp", type=int, default=10)
    ap.add_argument("--nn", type=int, default=30)
    ap.add_argument("--md", type=float, default=0.05)
    ap.add_argument("--ms", type=int, default=3)
    ap.add_argument("--mcs", type=int, nargs="+", default=[10, 15])
    ap.add_argument("--tfidf_max", type=int, default=3000)
    ap.add_argument("--n-seeds", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    docs = load_docs(a.root)
    N = len(docs)
    df = pd.DataFrame([{k: d[k] for k in ("id", "year", "stem", "broad")} for d in docs]).set_index("id")
    years = df["year"]
    base = pd.read_csv(a.baseline, index_col="id")["baseline_cluster"].reindex(df.index)
    if base.isnull().any():
        raise SystemExit("baseline labels do not match the loaded videos; re-run cns_r10_sensitivity.py")
    base = base.astype(int)
    with redirect_stdout(io.StringIO()):
        emb, vocab = get_embeddings(docs, a.tfidf_max)
    vocab = np.array(vocab)
    seeds = list(range(1, a.n_seeds + 1))

    out_labels = df.copy()
    out_labels["paper_cluster"] = base
    run_rows, summ_rows, medoids = [], [], {}
    for mcs in a.mcs:
        runs = {}
        for s in seeds:
            runs[s] = cluster(docs, s, a.tfidf_max, a.nn, a.md, mcs, a.ms, ncomp=a.ncomp, n_full=N)
            run_rows.append(dict(ncomp=a.ncomp, nn=a.nn, mcs=mcs, seed=s,
                                 **full_compare(base, runs[s], years)))
            print("mcs=%d seed=%d: %d clusters" % (mcs, s, len(set(runs[s]) - {-1})))
        m, mean_pair = medoid_seed(runs)
        medoids[mcs] = runs[m]
        out_labels["new_cluster_mcs%d" % mcs] = runs[m]
        sub = pd.DataFrame([r for r in run_rows if r["mcs"] == mcs])
        summ_rows.append(dict(
            ncomp=a.ncomp, nn=a.nn, mcs=mcs, n_seeds=len(seeds), medoid_seed=m,
            between_seed_ari_mean=mean_pair,
            n_clusters_min=sub.n_clusters.min(), n_clusters_max=sub.n_clusters.max(),
            collapse_frac=float((sub.n_clusters <= COLLAPSE).mean()),
            ari_vs_paper_median=sub.ari_nonnoise.median(),
            jaccard_vs_paper_median=sub.mean_jaccard_size_weighted.median(),
            recovered_j50_median=sub.frac_clusters_recovered_j50.median(),
            trajectory_rho_median=sub.trajectory_rho_median.median()))
        tab = matching_table(base, runs[m], years, emb, vocab)
        tab.to_csv(os.path.join(a.out, "crosscheck_cluster_matching_mcs%d.csv" % mcs), index=False)
        print("mcs=%d matching table written (medoid seed %d)" % (mcs, m))
    pd.DataFrame(run_rows).to_csv(os.path.join(a.out, "crosscheck_runs.csv"), index=False)
    pd.DataFrame(summ_rows).to_csv(os.path.join(a.out, "crosscheck_summary.csv"), index=False)
    out_labels.to_csv(os.path.join(a.out, "crosscheck_labels.csv"))

    # ---- same 2D map as the paper (n_neighbors 20, min_dist 0.05, seed 42) ----
    with redirect_stdout(io.StringIO()):
        coords = run_umap(emb, 20, 0.05, 2, 42)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Paper clusters (2D, n_neighbors=20)", base)]
    panels += [("%dD, n_neighbors=%d, min_cluster_size=%d" % (a.ncomp, a.nn, m), medoids[m])
               for m in a.mcs]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5))
    cmap = plt.get_cmap("tab20")
    for ax, (title, lab) in zip(np.atleast_1d(axes), panels):
        noise = lab.values == -1
        ax.scatter(coords[noise, 0], coords[noise, 1], s=8, c="lightgrey")
        for k, cid in enumerate(sorted(set(lab[lab >= 0]))):
            mk = lab.values == cid
            ax.scatter(coords[mk, 0], coords[mk, 1], s=12, color=cmap(k % 20))
            ax.text(np.median(coords[mk, 0]), np.median(coords[mk, 1]), str(cid), fontsize=9,
                    weight="bold", ha="center")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "fig_crosscheck_map.png"), dpi=300)
    print("Done. Outputs in", a.out)


if __name__ == "__main__":
    main()