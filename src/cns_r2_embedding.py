"""
CNSeminar — Result 2: The Intellectual Landscape (UMAP + HDBSCAN)
=================================================================
Embeds every transcript as a TF-IDF vector, reduces with UMAP,
clusters with HDBSCAN, and labels topics from cluster-level TF-IDF.

Produces:
  r2_umap_landscape.pdf/png  — UMAP scatter coloured by cluster,
                                with cluster labels, year encoded
                                as point alpha/size
  r2_umap_by_year.pdf/png    — same UMAP, coloured by year (continuous)
  r2_umap_by_type.pdf/png    — same UMAP, coloured by content type
  r2_cluster_summary.csv     — cluster ID, label, size, top terms,
                                year distribution

NOTE on embeddings:
  This script uses TF-IDF vectors → UMAP, which works well and has
  no extra dependencies. If sentence-transformers becomes available,
  swap get_embeddings() for sentence-level encodings (commented below)
  for richer semantic geometry.

Usage:
    python cns_r2_embedding.py --folder /path/to/transcripts --output ./r2_figures
    python cns_r2_embedding.py --folder /path/to/transcripts --n_neighbors 15 --min_dist 0.05
"""

import os, sys, argparse, warnings, csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import umap
import hdbscan

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import textwrap
from cns_figstyle import boost_style, declutter
from cns_analysis_pipeline import (
    tokenize, preprocess, display, classify_content_type,
    CONTENT_CATEGORIES, CATEGORY_PALETTE, CLUSTER_PALETTE,
    STOP, PALETTE, GREY_DARK, GREY_MID, GREY_FILL, ACCENT, set_style
)


# ═══════════════════════════════════════════════════════════════════════════════
# CORPUS LOADING — one document per transcript file
# ═══════════════════════════════════════════════════════════════════════════════

def load_documents(root_folder, start_year=None, end_year=None):
    """
    Returns a list of dicts, one per transcript:
        text, tokens, year, stem, broad, granular
    """
    root = Path(root_folder)
    docs = []
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        try:
            year = int(subfolder.name)
        except ValueError:
            continue
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue
        for fname in sorted(subfolder.glob('*.txt')):
            try:
                text = fname.read_text(encoding='utf-8', errors='replace')
                toks = tokenize(text)
                if len(toks) < 50:          # skip very short transcripts
                    continue
                broad, granular = classify_content_type(fname.stem)
                docs.append({
                    'text':     ' '.join(toks),   # preprocessed, for TF-IDF
                    'tokens':   toks,
                    'year':     year,
                    'stem':     fname.stem,
                    'broad':    broad,
                    'granular': granular,
                })
            except Exception as e:
                print(f'  Warning: {fname.name}: {e}')

    print(f'  Loaded {len(docs)} transcripts')
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

def get_embeddings(docs, max_features=3000):
    """
    TF-IDF embeddings, L2-normalised.
    To upgrade to sentence-transformers:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # join raw text (not tokenised) for each doc
        raw_texts = [d['raw_text'] for d in docs]   # store raw in load_documents
        return model.encode(raw_texts, show_progress_bar=True)
    """
    texts = [d['text'] for d in docs]
    vec   = TfidfVectorizer(max_features=max_features, min_df=2,
                            stop_words=list(STOP))
    mat   = vec.fit_transform(texts)
    emb   = normalize(mat, norm='l2').toarray()
    return emb, vec.get_feature_names_out()


# ═══════════════════════════════════════════════════════════════════════════════
# UMAP REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_umap(embeddings, n_neighbors=15, min_dist=0.05,
             n_components=2, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='cosine',
        random_state=random_state,
    )
    print('  Running UMAP...')
    coords = reducer.fit_transform(embeddings)
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# HDBSCAN CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def run_hdbscan(coords, min_cluster_size=8, min_samples=3):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    print('  Running HDBSCAN...')
    labels = clusterer.fit_predict(coords)
    probs  = clusterer.probabilities_
    return labels, probs, clusterer


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC LABELLING
# ═══════════════════════════════════════════════════════════════════════════════

def label_clusters(docs, labels, vocab, n_top=8):
    """
    For each cluster, compute within-cluster TF-IDF to find
    the most distinctive terms, then produce a short label.
    Returns: dict cluster_id -> {'label': str, 'terms': [str], 'size': int}
    """
    unique_labels = sorted(set(labels))
    cluster_info  = {}

    for cid in unique_labels:
        if cid == -1:
            cluster_info[cid] = {
                'label': 'Noise / unclustered',
                'terms': [],
                'size':  int((labels == -1).sum()),
            }
            continue

        # pool tokens for this cluster
        idx      = np.where(labels == cid)[0]
        all_toks = [t for i in idx for t in docs[i]['tokens']]
        freq     = Counter(all_toks)
        top      = [display(w) for w, _ in freq.most_common(n_top)
                    if w not in STOP][:n_top]

        # short label: first 3 top terms
        label = ' / '.join(top[:3]) if top else f'Cluster {cid}'
        cluster_info[cid] = {
            'label': label,
            'terms': top,
            'size':  len(idx),
        }

    return cluster_info


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R2A — UMAP coloured by cluster
# ═══════════════════════════════════════════════════════════════════════════════

# True  -> cluster name written inside the plot next to the T-number
# False -> only the T-number in the plot (full names stay in the legend): cleaner
SHOW_NAMES_IN_PLOT = False


def fig_umap_landscape(coords, labels, probs, cluster_info,
                        docs, out, fmt, dpi=300):
    unique_labels = sorted(set(labels))
    n_clusters    = len([l for l in unique_labels if l >= 0])

    # colour map for clusters — soft muted palette distinct from category colours
    color_map = {cid: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                 for i, cid in enumerate(l for l in unique_labels if l >= 0)}
    color_map[-1] = '#CCCCCC'   # noise = light grey

    fig, ax = plt.subplots(figsize=(13, 10))

    # noise first (background)
    noise_idx = labels == -1
    ax.scatter(coords[noise_idx, 0], coords[noise_idx, 1],
               c='#CCCCCC', s=30, alpha=0.30, linewidths=0, rasterized=True,
               label='Unclustered')

    # clusters
    for cid in [l for l in unique_labels if l >= 0]:
        idx   = labels == cid
        col   = color_map[cid]
        alpha = np.clip(probs[idx], 0.3, 1.0)
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=[col] * idx.sum(), s=70,
                   alpha=0.75, linewidths=0, rasterized=True)

        # label at cluster centroid
        cx, cy = coords[idx, 0].mean(), coords[idx, 1].mean()
        info   = cluster_info[cid]
        ax.text(cx, cy, (f"T{cid+1}\n{textwrap.fill(str(info['label']), 16)}" if SHOW_NAMES_IN_PLOT else f"T{cid+1}"),
                fontsize=11 if SHOW_NAMES_IN_PLOT else 16, fontweight='bold', ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec=col, alpha=0.82, lw=0.8))

    # legend patches
    patches = [mpatches.Patch(color=color_map[cid],
                               label=f"T{cid+1}: {cluster_info[cid]['label']} "
                                     f"(n={cluster_info[cid]['size']})")
               for cid in unique_labels if cid >= 0]
    patches.append(mpatches.Patch(color='#CCCCCC',
                                   label=f"Unclustered (n={noise_idx.sum()})"))
    ax.legend(handles=patches, fontsize=11, loc='lower left',
              bbox_to_anchor=(1.01, 0), borderpad=0.5,
              title='Topic clusters', title_fontsize=12)

    ax.set_title(
        'The intellectual landscape of CNSeminars: UMAP of all talks (HDBSCAN topics)',
        fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP dimension 1', fontsize=12)
    ax.set_ylabel('UMAP dimension 2', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()

    save(fig, 'r2_umap_landscape', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R2B — UMAP coloured by year
# ═══════════════════════════════════════════════════════════════════════════════

def fig_umap_by_year(coords, labels, docs, out, fmt, dpi=300):
    years      = np.array([d['year'] for d in docs])
    year_min   = years.min()
    year_max   = years.max()
    norm       = Normalize(vmin=year_min, vmax=year_max)
    cmap       = plt.get_cmap('Greys')

    fig, ax = plt.subplots(figsize=(12, 9))

    # noise grey
    noise_idx = labels == -1
    ax.scatter(coords[noise_idx, 0], coords[noise_idx, 1],
               c='#CCCCCC', s=30, alpha=0.25, linewidths=0, rasterized=True)

    sc = ax.scatter(coords[~noise_idx, 0], coords[~noise_idx, 1],
                    c=years[~noise_idx], cmap=cmap, norm=norm,
                    s=60, alpha=0.8, linewidths=0, rasterized=True)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Recording year', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title('Intellectual landscape coloured by year', fontsize=15,
                 fontweight='bold', pad=10)
    ax.set_xlabel('UMAP dimension 1', fontsize=12)
    ax.set_ylabel('UMAP dimension 2', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, 'r2_umap_by_year', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R2C — UMAP coloured by content type
# ═══════════════════════════════════════════════════════════════════════════════

def fig_umap_by_type(coords, labels, docs, out, fmt, dpi=300):
    cats   = list(CONTENT_CATEGORIES.keys())
    c_map  = {c: CATEGORY_PALETTE[c] for c in cats}
    colors = [c_map[d['broad']] for d in docs]

    fig, ax = plt.subplots(figsize=(12, 9))

    for cat in cats:
        idx = np.array([d['broad'] == cat for d in docs])
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=CATEGORY_PALETTE[cat], s=60, alpha=0.78,
                   linewidths=0, label=CONTENT_CATEGORIES[cat],
                   rasterized=True)

    ax.legend(fontsize=12, loc='lower left',
              bbox_to_anchor=(1.01, 0), borderpad=0.5,
              title='Content type', title_fontsize=12, markerscale=1.4)
    ax.set_title('Intellectual landscape coloured by content type', fontsize=15,
                 fontweight='bold', pad=10)
    ax.set_xlabel('UMAP dimension 1', fontsize=12)
    ax.set_ylabel('UMAP dimension 2', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, 'r2_umap_by_type', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R2D — Cluster composition: year distribution per topic
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cluster_year_profile(coords, labels, docs, cluster_info,
                              out, fmt, dpi=300):
    unique_labels = sorted(l for l in set(labels) if l >= 0)
    if not unique_labels:
        return
    years_all = sorted(set(d['year'] for d in docs))
    n_cls     = len(unique_labels)

    # matrix: clusters × years
    mat = np.zeros((n_cls, len(years_all)))
    for i, cid in enumerate(unique_labels):
        idx   = np.where(labels == cid)[0]
        yr_counts = Counter(docs[j]['year'] for j in idx)
        for k, yr in enumerate(years_all):
            mat[i, k] = yr_counts.get(yr, 0)
    # normalise each cluster to proportions
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, row_sums, where=row_sums > 0)

    cmap_clusters = None  # unused — colours drawn from CLUSTER_PALETTE

    fig, ax = plt.subplots(figsize=(14, max(4.5, n_cls * 0.8 + 2.5)))
    im = ax.imshow(mat_norm, aspect='auto',
                   cmap='Greys', interpolation='nearest', vmin=0)
    ax.set_xticks(range(len(years_all)))
    ax.set_xticklabels([str(y) for y in years_all],
                        rotation=45, ha='right', fontsize=11.5)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels([f"T{cid+1}: {cluster_info[cid]['label']}"
                         for cid in unique_labels], fontsize=11.5)
    ax.set_xlabel('Year')
    ax.set_title('Topic prevalence by year (proportion of cluster)',
                 fontsize=15, fontweight='bold', pad=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label('Proportion of cluster', fontsize=11.5)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    save(fig, 'r2_cluster_year_profile', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def write_cluster_csv(docs, labels, cluster_info, coords, out):
    os.makedirs(out, exist_ok=True)

    # per-document
    path = os.path.join(out, 'r2_document_clusters.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stem', 'year', 'broad', 'cluster_id', 'cluster_label',
                    'umap_x', 'umap_y'])
        for i, doc in enumerate(docs):
            cid = int(labels[i])
            lbl = cluster_info.get(cid, {}).get('label', 'Noise')
            w.writerow([doc['stem'], doc['year'], doc['broad'],
                        cid, lbl,
                        f'{coords[i,0]:.4f}', f'{coords[i,1]:.4f}'])
    print('  ✓  r2_document_clusters.csv')

    # cluster summary
    path2 = os.path.join(out, 'r2_cluster_summary.csv')
    with open(path2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cluster_id', 'label', 'size', 'top_terms'])
        for cid, info in sorted(cluster_info.items()):
            w.writerow([cid, info['label'], info['size'],
                        ' | '.join(info['terms'])])
    print('  ✓  r2_cluster_summary.csv')


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
        description='CNSeminar Result 2 — UMAP embedding and topic clustering')
    p.add_argument('--folder',           required=True)
    p.add_argument('--output',           default='./r2_figures')
    p.add_argument('--format',           choices=['pdf', 'png', 'both'], default='both')
    p.add_argument('--dpi',              type=int, default=300)
    p.add_argument('--start',            type=int, default=None)
    p.add_argument('--end',              type=int, default=None)
    p.add_argument('--max_features',     type=int, default=3000,
                   help='TF-IDF vocabulary size (default: 3000)')
    p.add_argument('--n_neighbors',      type=int, default=15,
                   help='UMAP n_neighbors (default: 15)')
    p.add_argument('--min_dist',         type=float, default=0.05,
                   help='UMAP min_dist (default: 0.05)')
    p.add_argument('--min_cluster_size', type=int, default=8,
                   help='HDBSCAN min_cluster_size (default: 8)')
    p.add_argument('--min_samples',      type=int, default=3,
                   help='HDBSCAN min_samples (default: 3)')
    p.add_argument('--random_state',     type=int, default=42)
    args = p.parse_args()

    set_style(args.dpi)
    boost_style()
    print(f'\nLoading transcripts from: {args.folder}')
    docs = load_documents(args.folder, args.start, args.end)

    if len(docs) < 10:
        print('Too few documents. Check folder structure.')
        sys.exit(1)

    print(f'\n[1/5] Computing TF-IDF embeddings '
          f'(max_features={args.max_features})...')
    embeddings, vocab = get_embeddings(docs, args.max_features)

    print(f'[2/5] UMAP reduction '
          f'(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...')
    coords = run_umap(embeddings,
                      n_neighbors=args.n_neighbors,
                      min_dist=args.min_dist,
                      random_state=args.random_state)

    print(f'[3/5] HDBSCAN clustering '
          f'(min_cluster_size={args.min_cluster_size}, '
          f'min_samples={args.min_samples})...')
    labels, probs, clusterer = run_hdbscan(
        coords,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f'  {n_clusters} clusters found, {n_noise} unclustered documents')

    print('[4/5] Labelling clusters...')
    cluster_info = label_clusters(docs, labels, vocab)
    for cid, info in sorted(cluster_info.items()):
        if cid >= 0:
            print(f'  T{cid+1} (n={info["size"]}): {info["label"]}')
            print(f'       terms: {", ".join(info["terms"])}')

    print(f'\n[5/5] Producing figures (saving to {args.output})...')
    fig_umap_landscape(coords, labels, probs, cluster_info,
                       docs, args.output, args.format, args.dpi)
    fig_umap_by_year(coords, labels, docs,
                     args.output, args.format, args.dpi)
    fig_umap_by_type(coords, labels, docs,
                     args.output, args.format, args.dpi)
    fig_cluster_year_profile(coords, labels, docs, cluster_info,
                              args.output, args.format, args.dpi)

    write_cluster_csv(docs, labels, cluster_info, coords, args.output)

    print(f'\nAll outputs saved to {args.output}')
    print(f'Tip: re-run with different --n_neighbors, --min_dist,')
    print(f'     --min_cluster_size to explore robustness.')


if __name__ == '__main__':
    main()