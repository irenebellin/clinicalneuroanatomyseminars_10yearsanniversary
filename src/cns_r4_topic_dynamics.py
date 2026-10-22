"""
CNSeminar — Result 4: Topics That Emerged, Persisted and Declined
=================================================================
Builds on cluster assignments from cns_r2_embedding.py (or re-runs
them internally) and adds trajectory classification to each topic.

Can run standalone (re-runs UMAP/HDBSCAN) or read the CSV output
from r2 to avoid recomputing embeddings.

Produces:
  r4a_topic_prevalence_area   — stacked area: topic proportions over time
  r4b_topic_trajectories      — small-multiple line plots per topic
  r4c_topic_classification    — summary chart: emerging / persisting / declined
  r4d_topic_heatmap           — topics × years heatmap
  r4_topic_dynamics.csv       — per-topic per-year prevalence + classification

Usage:
    # Re-run embeddings internally (slower, self-contained):
    python cns_r4_topic_dynamics.py --folder /path/to/transcripts

    # Use pre-computed cluster assignments from r2:
    python cns_r4_topic_dynamics.py --folder /path/to/transcripts \\
        --clusters ./r2_figures/r2_document_clusters.csv
"""

import os, sys, argparse, warnings, csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import textwrap
from cns_figstyle import boost_style, declutter
from cns_analysis_pipeline import (
    tokenize, display, classify_content_type,
    STOP, PALETTE, CLUSTER_PALETTE, GREY_DARK, GREY_MID, GREY_FILL, ACCENT,
    set_style, load_corpus
)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY CLASSIFICATION (same logic as r3, kept local for independence)
# ═══════════════════════════════════════════════════════════════════════════════

def linear_slope(vals):
    x = np.arange(len(vals)).reshape(-1, 1)
    y = np.array(vals, dtype=float)
    if y.std() == 0:
        return 0.0, 0.0
    reg = LinearRegression().fit(x, y)
    ss_res = ((y - reg.predict(x))**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(reg.coef_[0]), r2


def classify_trajectory(vals, min_active=3, slope_thresh=0.003, r2_thresh=0.20):
    active = sum(v > 0 for v in vals)
    if active < min_active:
        return 'sparse'
    slope, r2 = linear_slope(vals)
    if r2 < r2_thresh:
        peak_idx = int(np.argmax(vals))
        if peak_idx < len(vals) * 0.65 and vals[-1] < vals[peak_idx] * 0.5:
            return 'peaked'
        return 'persistent'
    if slope > slope_thresh:
        return 'emerging'
    if slope < -slope_thresh:
        return 'declining'
    return 'persistent'


TRAJ_COLORS = {
    'emerging':   '#7A9E7E',   # muted sage-green
    'persistent': '#4A6FA5',   # muted slate-blue
    'peaked':     '#B0956A',   # muted amber
    'declining':  '#B07D62',   # muted terracotta
    'sparse':     '#BBBBBB',   # grey
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD CLUSTER ASSIGNMENTS FROM R2 CSV
# ═══════════════════════════════════════════════════════════════════════════════

def load_clusters_from_csv(csv_path):
    """
    Read r2_document_clusters.csv produced by cns_r2_embedding.py.
    Returns list of dicts: stem, year, broad, cluster_id, cluster_label.
    """
    docs = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append({
                'stem':          row['stem'],
                'year':          int(row['year']),
                'broad':         row['broad'],
                'cluster_id':    int(row['cluster_id']),
                'cluster_label': row['cluster_label'],
            })
    print(f'  Loaded {len(docs)} document assignments from {csv_path}')
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# OR: RE-RUN EMBEDDING + CLUSTERING INTERNALLY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_clusters(folder, start_year, end_year,
                     n_neighbors=15, min_dist=0.05,
                     min_cluster_size=8, min_samples=3,
                     max_features=3000, random_state=42):
    """
    Lightweight re-run of cns_r2_embedding pipeline.
    Returns docs list with cluster_id / cluster_label populated.
    """
    try:
        import umap
        import hdbscan
    except ImportError:
        print('ERROR: umap-learn and hdbscan required. '
              'Install with: pip install umap-learn hdbscan --break-system-packages')
        sys.exit(1)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from cns_r2_embedding import (
        load_documents, get_embeddings, run_umap, run_hdbscan, label_clusters
    )

    docs       = load_documents(folder, start_year, end_year)
    embeddings, vocab = get_embeddings(docs, max_features)
    coords     = run_umap(embeddings, n_neighbors, min_dist,
                          random_state=random_state)
    labels, probs, _ = run_hdbscan(coords, min_cluster_size, min_samples)
    cluster_info = label_clusters(docs, labels, vocab)

    for i, doc in enumerate(docs):
        cid = int(labels[i])
        doc['cluster_id']    = cid
        doc['cluster_label'] = cluster_info.get(cid, {}).get('label', 'Noise')

    return docs, cluster_info


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE TOPIC PREVALENCE BY YEAR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_prevalence(docs):
    """
    Returns:
        all_years       sorted list of years
        cluster_ids     sorted list of cluster ids (excluding -1 noise)
        cluster_labels  dict cluster_id -> label
        prev_matrix     np.array (n_clusters × n_years): proportion of year's
                        docs assigned to each cluster
        counts_matrix   raw counts
    """
    all_years  = sorted(set(d['year'] for d in docs))
    cluster_ids = sorted(set(d['cluster_id'] for d in docs if d['cluster_id'] >= 0))
    cluster_labels = {}
    for d in docs:
        if d['cluster_id'] >= 0:
            cluster_labels[d['cluster_id']] = d['cluster_label']

    counts = np.zeros((len(cluster_ids), len(all_years)), dtype=float)
    totals = np.zeros(len(all_years), dtype=float)

    for d in docs:
        yr_idx = all_years.index(d['year'])
        totals[yr_idx] += 1
        if d['cluster_id'] >= 0:
            cl_idx = cluster_ids.index(d['cluster_id'])
            counts[cl_idx, yr_idx] += 1

    prev = np.divide(counts, np.maximum(totals, 1))

    return all_years, cluster_ids, cluster_labels, prev, counts


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R4A — Stacked area: topic proportions over time
# ═══════════════════════════════════════════════════════════════════════════════

def fig_topic_area(all_years, cluster_ids, cluster_labels,
                   prev_matrix, out, fmt, dpi=300):
    n_topics = len(cluster_ids)
    colors   = [CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(n_topics)]
    labels   = [f"T{cid+1}: {cluster_labels.get(cid,'')}"
                for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.stackplot(all_years, prev_matrix,
                 labels=labels, colors=colors, alpha=0.88)
    ax.set_xlim(all_years[0], all_years[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Year')
    ax.set_ylabel('Proportion of annual corpus')
    ax.set_title('Topic composition over time (UMAP/HDBSCAN clusters)',
                 fontsize=15, fontweight='bold', pad=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=min(3, n_topics), fontsize=11.5)
    plt.tight_layout()
    save(fig, 'r4a_topic_prevalence_area', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R4B — Small-multiple trajectories per topic
# ═══════════════════════════════════════════════════════════════════════════════

def fig_topic_trajectories(all_years, cluster_ids, cluster_labels,
                            prev_matrix, out, fmt, dpi=300):
    n   = len(cluster_ids)
    nc  = min(4, n)
    nr  = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc,
                             figsize=(nc * 4.4, nr * 3.5), sharey=False)
    flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, cid in enumerate(cluster_ids):
        ax    = flat[i]
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        vals  = list(prev_matrix[i])

        ax.fill_between(all_years, vals, alpha=0.15, color=color)
        ax.plot(all_years, vals, color=color, lw=2.8, zorder=3)
        ax.scatter(all_years, vals, color=color, s=50, zorder=4, alpha=0.75)

        lbl = cluster_labels.get(cid, f'Topic {cid+1}')
        ax.set_title(textwrap.fill(f'T{cid+1}: {lbl}', 30), fontsize=12, fontweight='bold',
                     pad=3, color=GREY_DARK)

        step = max(1, len(all_years) // 5)
        ax.set_xticks(all_years[::step])
        ax.set_xticklabels([str(y)[2:] for y in all_years[::step]], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        if i % nc == 0:
            ax.set_ylabel('Proportion', fontsize=11)

    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    fig.suptitle('Topic prevalence trajectories: emergence, persistence and decline',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r4b_topic_trajectories', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R4C — Topic classification summary
# ═══════════════════════════════════════════════════════════════════════════════

def fig_topic_classification(all_years, cluster_ids, cluster_labels,
                              prev_matrix, out, fmt, dpi=300):
    traj_classes = []
    slopes_all   = []
    for i, cid in enumerate(cluster_ids):
        vals = list(prev_matrix[i])
        tc   = classify_trajectory(vals)
        sl, _ = linear_slope(vals)
        traj_classes.append(tc)
        slopes_all.append(sl)

    # sort by slope for display
    order = np.argsort(slopes_all)[::-1]
    sorted_ids    = [cluster_ids[i]    for i in order]
    sorted_labels = [cluster_labels.get(cluster_ids[i], f'T{cluster_ids[i]+1}')
                     for i in order]
    sorted_trajs  = [traj_classes[i]   for i in order]
    sorted_slopes = [slopes_all[i]     for i in order]

    colors = [TRAJ_COLORS.get(t, '#BBBBBB') for t in sorted_trajs]

    fig, ax = plt.subplots(figsize=(12, max(5, len(cluster_ids) * 0.55 + 2)))
    bars = ax.barh(range(len(sorted_ids)),
                   [s * 1000 for s in sorted_slopes],   # scale for readability
                   color=colors, edgecolor='white', linewidth=0.4, height=0.68)
    ax.axvline(0, color='#444440', lw=1.2)
    ax.margins(x=0.22)
    ax.set_yticks(range(len(sorted_ids)))
    ax.set_yticklabels(
        [f"T{cid+1}: {lbl}" for cid, lbl in zip(sorted_ids, sorted_labels)],
        fontsize=11.5)
    ax.set_xlabel('Growth rate (×10⁻³ proportion / year)', fontsize=12)
    ax.set_title('Topic growth rates and trajectory classification',
                 fontsize=15, fontweight='bold', pad=8)

    # trajectory legend
    patches = [mpatches.Patch(color=c, label=t.capitalize())
               for t, c in TRAJ_COLORS.items() if t != 'sparse']
    ax.legend(handles=patches, fontsize=12, loc='lower right')

    # annotate trajectory label
    for j, (bar, traj) in enumerate(zip(bars, sorted_trajs)):
        x = bar.get_width()
        offset = 0.1 * max(abs(s * 1000) for s in sorted_slopes)
        ax.text(x + (offset if x >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                traj, va='center', fontsize=11,
                color=TRAJ_COLORS.get(traj, '#888780'), fontweight='bold',
                ha='left' if x >= 0 else 'right')

    plt.tight_layout()
    save(fig, 'r4c_topic_classification', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R4D — Topics × years heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_topic_heatmap(all_years, cluster_ids, cluster_labels,
                       prev_matrix, out, fmt, dpi=300):
    n_topics = len(cluster_ids)
    cmap     = 'Greys'
    ylabels  = [f"T{cid+1}: {cluster_labels.get(cid,'')}" for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(max(12, len(all_years) * 0.8 + 3),
                                    max(5, n_topics * 0.6 + 2)))
    im = ax.imshow(prev_matrix, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=0)
    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels([str(y) for y in all_years],
                        rotation=45, ha='right', fontsize=11.5)
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels(ylabels, fontsize=11.5)
    ax.set_xlabel('Year')
    ax.set_title('Topic prevalence by year',
                 fontsize=15, fontweight='bold', pad=8)

    for i in range(n_topics + 1):
        ax.axhline(i - 0.5, color='white', lw=0.4)
    for j in range(len(all_years) + 1):
        ax.axvline(j - 0.5, color='white', lw=0.4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label('Proportion of annual corpus', fontsize=11.5)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    save(fig, 'r4d_topic_heatmap', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def write_dynamics_csv(all_years, cluster_ids, cluster_labels,
                        prev_matrix, counts_matrix, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'r4_topic_dynamics.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cluster_id', 'cluster_label', 'trajectory',
                    'slope', 'r2', 'mean_prevalence',
                    'first_active_year', 'peak_year']
                   + [f'prev_{yr}' for yr in all_years]
                   + [f'count_{yr}' for yr in all_years])
        for i, cid in enumerate(cluster_ids):
            vals  = list(prev_matrix[i])
            cnts  = list(counts_matrix[i])
            traj  = classify_trajectory(vals)
            sl, r2 = linear_slope(vals)
            active_years = [all_years[j] for j, v in enumerate(vals) if v > 0]
            first_yr = active_years[0] if active_years else ''
            peak_yr  = all_years[int(np.argmax(vals))]
            w.writerow(
                [cid, cluster_labels.get(cid, ''), traj,
                 f'{sl:.6f}', f'{r2:.4f}', f'{np.mean(vals):.4f}',
                 first_yr, peak_yr]
                + [f'{v:.4f}' for v in vals]
                + [int(c) for c in cnts]
            )
    print('  ✓  r4_topic_dynamics.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def save(fig, name, out, fmt, dpi=300):
    os.makedirs(out, exist_ok=True)
    declutter(fig)
    if fmt in ('png', 'both'):
        fig.savefig(os.path.join(out, f'{name}.png'), dpi=dpi, bbox_inches='tight')
    if fmt in ('pdf', 'both'):
        fig.savefig(os.path.join(out, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {name}')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='CNSeminar Result 4 — Topic trajectory dynamics')
    p.add_argument('--folder',           required=True)
    p.add_argument('--output',           default='./r4_figures')
    p.add_argument('--format',           choices=['pdf', 'png', 'both'], default='both')
    p.add_argument('--dpi',              type=int, default=300)
    p.add_argument('--start',            type=int, default=None)
    p.add_argument('--end',              type=int, default=None)
    p.add_argument('--clusters',         default=None,
                   help='Path to r2_document_clusters.csv (skips re-embedding)')
    # embedding params (only used if --clusters not supplied)
    p.add_argument('--n_neighbors',      type=int, default=15)
    p.add_argument('--min_dist',         type=float, default=0.05)
    p.add_argument('--min_cluster_size', type=int, default=8)
    p.add_argument('--min_samples',      type=int, default=3)
    p.add_argument('--max_features',     type=int, default=3000)
    p.add_argument('--random_state',     type=int, default=42)
    args = p.parse_args()

    set_style(args.dpi)
    boost_style()

    if args.clusters:
        print(f'\nLoading cluster assignments from: {args.clusters}')
        docs = load_clusters_from_csv(args.clusters)
        cluster_labels = {}
        for d in docs:
            if d['cluster_id'] >= 0:
                cluster_labels[d['cluster_id']] = d['cluster_label']
    else:
        print(f'\nComputing embeddings and clusters from: {args.folder}')
        docs, cluster_info = compute_clusters(
            args.folder, args.start, args.end,
            args.n_neighbors, args.min_dist,
            args.min_cluster_size, args.min_samples,
            args.max_features, args.random_state)
        cluster_labels = {cid: info['label']
                          for cid, info in cluster_info.items() if cid >= 0}

    print(f'\nComputing prevalence over time...')
    all_years, cluster_ids, cluster_labels, prev_matrix, counts_matrix = \
        compute_prevalence(docs)

    n_topics = len(cluster_ids)
    print(f'  {n_topics} topics × {len(all_years)} years')
    for cid in cluster_ids:
        i   = cluster_ids.index(cid)
        traj = classify_trajectory(list(prev_matrix[i]))
        print(f'  T{cid+1}: {cluster_labels.get(cid,"")} → {traj}')

    print(f'\nProducing figures (saving to {args.output})...')
    fig_topic_area(all_years, cluster_ids, cluster_labels,
                   prev_matrix, args.output, args.format, args.dpi)
    fig_topic_trajectories(all_years, cluster_ids, cluster_labels,
                            prev_matrix, args.output, args.format, args.dpi)
    fig_topic_classification(all_years, cluster_ids, cluster_labels,
                              prev_matrix, args.output, args.format, args.dpi)
    fig_topic_heatmap(all_years, cluster_ids, cluster_labels,
                      prev_matrix, args.output, args.format, args.dpi)

    write_dynamics_csv(all_years, cluster_ids, cluster_labels,
                       prev_matrix, counts_matrix, args.output)

    print(f'\nAll outputs saved to {args.output}')


if __name__ == '__main__':
    main()