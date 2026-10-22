"""
CNSeminar - R2 sensitivity sweep
================================
Computes the TF-IDF + UMAP embedding ONCE, then sweeps the HDBSCAN settings
(min_cluster_size x min_samples) and prints, for each combination, the number
of clusters, the number of unclustered documents and the smallest / largest
cluster. Use it to choose and to report the clustering parameters.

Usage:
    python cns_r2_sweep.py --folder /path/to/transcripts \
        --output r2_sweep
"""
import os, sys, argparse, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def sweep(coords, sizes, samples):
    from cns_r2_embedding import run_hdbscan
    rows = []
    for mcs in sizes:
        for ms in samples:
            if ms > mcs:
                continue
            labels, _, _ = run_hdbscan(coords, min_cluster_size=mcs, min_samples=ms)
            ids = [l for l in set(labels) if l != -1]
            sizes_ = sorted((labels == l).sum() for l in ids)
            rows.append({
                'min_cluster_size': mcs,
                'min_samples': ms,
                'n_clusters': len(ids),
                'n_unclustered': int((labels == -1).sum()),
                'pct_unclustered': float(round(100 * (labels == -1).mean(), 1)),
                'smallest': int(sizes_[0]) if sizes_ else 0,
                'largest': int(sizes_[-1]) if sizes_ else 0,
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--output', default='./r2_sweep')
    p.add_argument('--max_features', type=int, default=3000)
    p.add_argument('--n_neighbors', type=int, default=15)
    p.add_argument('--min_dist', type=float, default=0.05)
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--sizes', type=int, nargs='+', default=[8, 10, 12, 15, 20, 25])
    p.add_argument('--samples', type=int, nargs='+', default=[3, 5, 8, 10, 15])
    args = p.parse_args()

    from cns_r2_embedding import load_documents, get_embeddings, run_umap
    docs = load_documents(args.folder)
    emb, _ = get_embeddings(docs, args.max_features)
    coords = run_umap(emb, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                      random_state=args.random_state)

    rows = sweep(coords, args.sizes, args.samples)
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, 'r2_sweep_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('\n min_cluster_size  min_samples  clusters  unclustered(%)  smallest  largest')
    for r in rows:
        print(f"  {r['min_cluster_size']:>14}  {r['min_samples']:>11}  {r['n_clusters']:>8}"
              f"  {r['n_unclustered']:>6} ({r['pct_unclustered']:>4}%)  {r['smallest']:>8}  {r['largest']:>7}")
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()