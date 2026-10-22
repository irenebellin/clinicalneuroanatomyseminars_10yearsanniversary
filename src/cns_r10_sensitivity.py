#!/usr/bin/env python
"""
cns_r10_sensitivity.py (v2)  --  Sensitivity and robustness analyses (Supplementary)

Changes from v1, motivated by the v1 results:
  * UMAP is seed-dependent and the baseline partition collapses to 2-6 clusters
    for some seeds. Every scenario is therefore run over several seeds and
    compared with the baseline run of the SAME seed; the seed-to-seed noise
    floor is summarised the same way.
  * min_cluster_size is also run in a "scaled" mode (proportional to the number
    of videos in the subset), because a fixed value of 15 forces small subsets
    (CNSeminars only, balanced subsamples) into 2-3 clusters by construction.
  * An extra grid varies the dimensionality used for clustering (2, 5, 10) and
    reports within-configuration stability (agreement between seeds), which
    does not depend on the baseline.

Scenarios: S1 CNSeminars only | S2 excl. Neuroccino | S3 excl. short-form |
           S4 year-balanced subsampling (+ weighted vocabulary) | S5 2020-2026 |
           S6 parameter grids | noise floor.
Run next to cns_analysis_pipeline.py and cns_r2_embedding.py. Python 3.8.
"""
import argparse
import io
import itertools
import os
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from cns_analysis_pipeline import classify_content_type, tokenize
from cns_r2_embedding import get_embeddings, run_hdbscan, run_umap

N_COMPONENTS = 2
CNS_LABEL = "cns_seminar"
NEUROCCINO_LABEL = "neuroccino"
COLLAPSE = 4          # a run with <= this many clusters is called "collapsed"
PAPER_SEED = 42
MIN_TOKENS = 50         # same filter as cns_r2_embedding.load_documents (main analysis)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
def load_docs(root):
    docs, n_files = [], 0
    for sub in sorted(Path(root).iterdir()):
        if not sub.is_dir():
            continue
        try:
            year = int(sub.name)
        except ValueError:
            continue
        for f in sorted(sub.glob("*.txt")):
            n_files += 1
            raw = f.read_text(encoding="utf-8", errors="replace")
            toks = tokenize(raw)
            if len(toks) < MIN_TOKENS:
                print("SKIPPED (%d tokens < %d after preprocessing): %d/%s"
                      % (len(toks), MIN_TOKENS, year, f.stem))
                continue
            broad, granular = classify_content_type(f.stem)
            docs.append(dict(id="%d/%s" % (year, f.stem), year=year, stem=f.stem,
                             broad=broad, granular=granular,
                             text=" ".join(toks), n_words=len(raw.split())))
    print(".txt files found in year folders: %d" % n_files)
    return docs


def subset(docs, mask):
    return [d for d, m in zip(docs, mask) if m]


# --------------------------------------------------------------------------
# Clustering and comparison
# --------------------------------------------------------------------------
def cluster(docs, seed, tfidf_max, nn, md, mcs, ms, ncomp=N_COMPONENTS,
            n_full=None, scale=False):
    """TF-IDF -> UMAP -> HDBSCAN with the R2 functions. If scale=True,
    min_cluster_size is scaled by len(docs)/n_full (floor 5)."""
    if scale and n_full:
        mcs = max(5, int(round(mcs * len(docs) / float(n_full))))
    ms = min(ms, mcs)
    with redirect_stdout(io.StringIO()):
        emb, vocab = get_embeddings(docs, tfidf_max)
        coords = run_umap(emb, min(nn, len(docs) - 1), md, ncomp, seed)
        labels, _, _ = run_hdbscan(coords, mcs, ms)
    return pd.Series(labels, index=[d["id"] for d in docs])


def compare_labels(base, new):
    shared = base.index.intersection(new.index)
    a, b = base[shared].values, new[shared].values
    m = (a >= 0) & (b >= 0)
    return dict(
        n_shared=len(shared),
        ari_all=adjusted_rand_score(a, b),
        ami_all=adjusted_mutual_info_score(a, b),
        ari_nonnoise=adjusted_rand_score(a[m], b[m]) if m.sum() > 2 else np.nan,
        n_clusters=len(set(new.values) - {-1}),
        noise_frac=float((new.values == -1).mean()),
    )


def prevalence(labels, years):
    df = pd.DataFrame({"c": labels, "y": years.loc[labels.index]})
    tot = df.groupby("y").size()
    ct = df[df.c >= 0].groupby(["c", "y"]).size().unstack(fill_value=0)
    return ct.reindex(columns=tot.index, fill_value=0).div(tot, axis=1), tot


def match_clusters(base, new, years, min_years=5, min_docs=3, min_jaccard=0.3):
    shared = base.index.intersection(new.index)
    b, n = base[shared], new[shared]
    bc, nc = sorted(set(b[b >= 0])), sorted(set(n[n >= 0]))
    out = dict(n_baseline_clusters=len(bc), mean_jaccard_size_weighted=np.nan,
               frac_clusters_recovered_j50=np.nan, trajectory_rho_median=np.nan,
               n_trajectories=0)
    if not bc or not nc:
        return out
    J = np.zeros((len(bc), len(nc)))
    for i, x in enumerate(bc):
        bi = set(b.index[b == x])
        for j, y in enumerate(nc):
            nj = set(n.index[n == y])
            J[i, j] = len(bi & nj) / float(len(bi | nj))
    r, c = linear_sum_assignment(-J)
    jac = np.zeros(len(bc))
    jac[r] = J[r, c]
    sizes = np.array([(b == x).sum() for x in bc], dtype=float)
    out["mean_jaccard_size_weighted"] = float(np.average(jac, weights=sizes))
    out["frac_clusters_recovered_j50"] = float((jac >= 0.5).mean())
    pb, _ = prevalence(base, years)
    pn, tn = prevalence(new, years)
    yrs = [y for y in pb.columns if y in pn.columns and tn[y] >= min_docs]
    rhos = []
    if len(yrs) >= min_years:
        for i, j in zip(r, c):
            if J[i, j] >= min_jaccard:
                rhos.append(spearmanr(pb.loc[bc[i], yrs], pn.loc[nc[j], yrs])[0])
    if rhos:
        out["trajectory_rho_median"] = float(np.nanmedian(rhos))
        out["n_trajectories"] = len(rhos)
    return out


def full_compare(base, new, years):
    d = compare_labels(base, new)
    d.update(match_clusters(base, new, years))
    return d


def summ(g):
    return pd.Series(dict(
        n_runs=len(g), n_videos=g["n_videos"].median(),
        ari_nonnoise_med=g.ari_nonnoise.median(),
        ari_nonnoise_q25=g.ari_nonnoise.quantile(.25),
        ari_nonnoise_q75=g.ari_nonnoise.quantile(.75),
        jaccard_med=g.mean_jaccard_size_weighted.median(),
        recovered_j50_med=g.frac_clusters_recovered_j50.median(),
        traj_rho_med=g.trajectory_rho_median.median(),
        n_clusters_med=g.n_clusters.median(),
        collapse_frac=(g.n_clusters <= COLLAPSE).mean()))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="sensitivity_out_v2")
    ap.add_argument("--tfidf_max", type=int, default=3000)
    ap.add_argument("--nn", type=int, default=20)
    ap.add_argument("--md", type=float, default=0.05)
    ap.add_argument("--mcs", type=int, default=15)
    ap.add_argument("--ms", type=int, default=3)
    ap.add_argument("--min-words", type=int, nargs="+", default=[1500, 3000])
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--n-rep", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--skip-grid", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    BASE = dict(tfidf_max=a.tfidf_max, nn=a.nn, md=a.md, mcs=a.mcs, ms=a.ms)

    docs = load_docs(a.root)
    N = len(docs)
    df = pd.DataFrame([{k: d[k] for k in ("id", "year", "stem", "broad", "granular", "n_words")}
                       for d in docs]).set_index("id")
    years = df["year"]
    print("Videos analysed: %d" % N)
    for lab in (CNS_LABEL, NEUROCCINO_LABEL):
        if lab not in set(df["broad"]):
            raise SystemExit("Label %r not found" % lab)
    seeds = list(range(1, a.n_seeds + 1))

    # ---- baseline runs: paper seed + all seeds ----
    base_runs = {}
    for s in [PAPER_SEED] + seeds:
        base_runs[s] = cluster(docs, s, n_full=N, **BASE)
    out = df.copy()
    out["baseline_cluster"] = base_runs[PAPER_SEED]
    out.to_csv(os.path.join(a.out, "baseline_labels.csv"))
    nc = [len(set(base_runs[s]) - {-1}) for s in seeds]
    print("Baseline clusters by seed:", nc)

    # ---- noise floor: baseline vs baseline, neighbouring seeds ----
    floor = []
    for i, s in enumerate(seeds):
        t = seeds[(i + 1) % len(seeds)]
        floor.append(dict(seed_a=s, seed_b=t, n_videos=N,
                          **full_compare(base_runs[s], base_runs[t], years)))
    fl = pd.DataFrame(floor)
    fl.to_csv(os.path.join(a.out, "sensitivity_noise_floor.csv"), index=False)
    floor_summary = summ(fl)
    floor_summary.to_csv(os.path.join(a.out, "sensitivity_noise_floor_summary.csv"), header=["value"])
    print("Noise floor done")

    # ---- S1, S2, S3, S5: subset scenarios x seeds x min_cluster_size mode ----
    broad, yr, nw = df["broad"].values, df["year"].values, df["n_words"].values
    scen = OrderedDict()
    scen["S1_cns_seminars_only"] = broad == CNS_LABEL
    scen["S2_excl_neuroccino"] = broad != NEUROCCINO_LABEL
    for w in a.min_words:
        scen["S3_excl_short_form_ge%d_words" % w] = nw >= w
    scen["S5_2020_2026"] = (yr >= 2020) & (yr <= 2026)
    rows = []
    for name, mask in scen.items():
        sub = subset(docs, mask)
        for mode in ("fixed_mcs", "scaled_mcs"):
            for s in seeds:
                lab = cluster(sub, s, n_full=N, scale=(mode == "scaled_mcs"), **BASE)
                rows.append(dict(scenario=name, mcs_mode=mode, seed=s, n_videos=len(sub),
                                 **full_compare(base_runs[s], lab, years)))
        print(name, "done")
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(a.out, "sensitivity_scenarios_runs.csv"), index=False)
    sc_sum = sc.groupby(["scenario", "mcs_mode"]).apply(summ).reset_index()
    sc_sum.to_csv(os.path.join(a.out, "sensitivity_scenarios_summary.csv"), index=False)

    # ---- S4: year-balanced subsampling ----
    rng = np.random.RandomState(PAPER_SEED)
    ids_by_year = df.groupby("year").apply(lambda g: list(g.index)).to_dict()
    by_id = dict((d["id"], d) for d in docs)
    reps = []
    for rep in range(a.n_rep):
        chosen = []
        for y, ids in ids_by_year.items():
            chosen.extend(list(rng.choice(ids, min(a.cap, len(ids)), replace=False)))
        sub = [by_id[i] for i in chosen]
        s = seeds[rep % len(seeds)]
        for mode in ("fixed_mcs", "scaled_mcs"):
            lab = cluster(sub, s, n_full=N, scale=(mode == "scaled_mcs"), **BASE)
            reps.append(dict(scenario="S4_year_balanced_cap%d" % a.cap, mcs_mode=mode,
                             rep=rep, seed=s, n_videos=len(sub),
                             **full_compare(base_runs[s], lab, years)))
    bal = pd.DataFrame(reps)
    bal.to_csv(os.path.join(a.out, "sensitivity_balanced_runs.csv"), index=False)
    bal.groupby(["scenario", "mcs_mode"]).apply(summ).reset_index().to_csv(
        os.path.join(a.out, "sensitivity_balanced_summary.csv"), index=False)
    print("Balanced subsampling done")

    # ---- S4b: inverse-year-weighted pooled vocabulary ----
    with redirect_stdout(io.StringIO()):
        emb, vocab = get_embeddings(docs, BASE["tfidf_max"])
    vocab = np.array(vocab)
    w = 1.0 / df.groupby("year")["stem"].transform("size").values
    m_u, m_w = emb.mean(axis=0), emb.T.dot(w) / w.sum()
    t_u, t_w = set(vocab[np.argsort(-m_u)[:100]]), set(vocab[np.argsort(-m_w)[:100]])
    pd.DataFrame([dict(jaccard_top100=len(t_u & t_w) / float(len(t_u | t_w)),
                       spearman_all_terms=spearmanr(m_u, m_w)[0])]).to_csv(
        os.path.join(a.out, "sensitivity_vocab_weighting.csv"), index=False)

    # ---- S6a: parameter grid (paper seed) ----
    g = None
    if not a.skip_grid:
        grid = []
        for mcs, ms, nn, md in itertools.product([8, 10, 15, 20, 30], [1, 3, 5],
                                                 [10, 15, 20, 30], [0.0, 0.05, 0.1]):
            lab = cluster(docs, PAPER_SEED, n_full=N, **dict(BASE, mcs=mcs, ms=ms, nn=nn, md=md))
            grid.append(dict(mcs=mcs, ms=ms, nn=nn, md=md, tfidf_max=a.tfidf_max, n_videos=N,
                             **full_compare(base_runs[PAPER_SEED], lab, years)))
        for tm in (1500, 5000):
            lab = cluster(docs, PAPER_SEED, n_full=N, **dict(BASE, tfidf_max=tm))
            grid.append(dict(mcs=a.mcs, ms=a.ms, nn=a.nn, md=a.md, tfidf_max=tm, n_videos=N,
                             **full_compare(base_runs[PAPER_SEED], lab, years)))
        g = pd.DataFrame(grid)
        g.to_csv(os.path.join(a.out, "sensitivity_param_grid.csv"), index=False)
        print("Parameter grid done")

    # ---- S6b: clustering dimensionality x nn x mcs, 3 seeds, within-config stability ----
    dim_rows = []
    for ncomp, nn, mcs in itertools.product([2, 5, 10], [10, 15, 20, 30], [8, 10, 15, 20]):
        labs = [cluster(docs, s, n_full=N, ncomp=ncomp, **dict(BASE, nn=nn, mcs=mcs))
                for s in (1, 2, 3)]
        pair = [compare_labels(labs[i], labs[j])["ari_nonnoise"]
                for i, j in ((0, 1), (0, 2), (1, 2))]
        ncl = [len(set(l) - {-1}) for l in labs]
        dim_rows.append(dict(ncomp=ncomp, nn=nn, mcs=mcs,
                             within_config_ari_mean=float(np.nanmean(pair)),
                             n_clusters_min=min(ncl), n_clusters_max=max(ncl),
                             collapse_frac=float(np.mean([c <= COLLAPSE for c in ncl])),
                             noise_frac_mean=float(np.mean([(l == -1).mean() for l in labs]))))
    dim = pd.DataFrame(dim_rows)
    dim.to_csv(os.path.join(a.out, "sensitivity_dim_grid.csv"), index=False)
    print("Dimensionality grid done")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    s2 = sc_sum.copy()
    s2["label"] = s2.scenario.str.replace("_excl_short_form_", "_short<").str.replace("_words", "w")
    ypos = {}
    for k, (mode, off) in enumerate((("fixed_mcs", -0.18), ("scaled_mcs", 0.18))):
        d = s2[s2.mcs_mode == mode].reset_index(drop=True)
        y = np.arange(len(d)) + off
        ax[0].errorbar(d.ari_nonnoise_med, y,
                       xerr=[d.ari_nonnoise_med - d.ari_nonnoise_q25, d.ari_nonnoise_q75 - d.ari_nonnoise_med],
                       fmt="o", label=mode)
    ax[0].set_yticks(np.arange(len(s2[s2.mcs_mode == "fixed_mcs"])))
    ax[0].set_yticklabels(s2[s2.mcs_mode == "fixed_mcs"].label, fontsize=7)
    ax[0].axvspan(floor_summary["ari_nonnoise_q25"], floor_summary["ari_nonnoise_q75"],
                  alpha=.2, color="grey", label="seed-to-seed IQR")
    ax[0].set_xlabel("ARI vs same-seed baseline (median, IQR)")
    ax[0].legend(fontsize=7)
    if g is not None:
        ax[1].hist(g.ari_nonnoise.dropna(), bins=20)
        ax[1].axvline(floor_summary["ari_nonnoise_med"], color="k", ls="--")
        ax[1].set_xlabel("ARI vs baseline (grid, paper seed)")
        ax[1].set_ylabel("Parameter settings")
    for k, nc_ in enumerate((2, 5, 10)):
        v = dim[dim.ncomp == nc_].within_config_ari_mean.values
        ax[2].scatter(np.full(len(v), k) + np.random.uniform(-.1, .1, len(v)), v, alpha=.7)
    ax[2].set_xticks([0, 1, 2])
    ax[2].set_xticklabels(["2D", "5D", "10D"])
    ax[2].set_ylabel("Mean between-seed ARI within configuration")
    ax[2].set_xlabel("UMAP dimensions used for clustering")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "fig_sensitivity.png"), dpi=300)
    print("Done. Outputs in", a.out)


if __name__ == "__main__":
    main()