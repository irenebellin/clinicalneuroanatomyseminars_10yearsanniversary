"""
CNSeminar — Result 1: Corpus Characteristics and Channel Growth
===============================================================
Produces a multi-panel demographics figure:
  A  Videos and words per year (already in main pipeline Fig1A — reproduced
     here with richer annotations)
  B  Cumulative video count over time
  C  Content-type composition by year (stacked bar)
  D  Token density (words per video) per year
  E  Summary statistics table

Usage:
    python cns_r1_demographics.py --folder /path/to/transcripts --output ./r1_figures
    python cns_r1_demographics.py --folder /path/to/transcripts --output ./r1_figures --format pdf

Imports the shared preprocessing layer from cns_analysis_pipeline.py, which
must be on the same directory or on PYTHONPATH.
"""

import os, sys, argparse, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── shared layer from main pipeline ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import textwrap
from cns_figstyle import boost_style, declutter
from cns_analysis_pipeline import (
    tokenize, display, classify_content_type,
    CONTENT_CATEGORIES, CATEGORY_PALETTE, STOP, PALETTE, set_style
)


# ═══════════════════════════════════════════════════════════════════════════════
# CORPUS LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_corpus(root_folder, start_year=None, end_year=None):
    """
    Returns per-file metadata and per-year aggregates.
    manifest: list of dicts with year, stem, broad, granular,
              n_tokens, n_words_raw (character-level estimate).
    """
    root = Path(root_folder)
    manifest = []
    years_data   = {}   # year -> [tokens]
    file_counts  = {}   # year -> n files
    type_year_counts = defaultdict(lambda: defaultdict(int))

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

        tokens_year, n = [], 0
        for fname in sorted(subfolder.glob('*.txt')):
            try:
                text = fname.read_text(encoding='utf-8', errors='replace')
                toks = tokenize(text)
                broad, granular = classify_content_type(fname.stem)
                # raw word count (split on whitespace, no lemmatisation)
                n_words_raw = len(text.split())
                manifest.append({
                    'year': year,
                    'stem': fname.stem,
                    'broad': broad,
                    'granular': granular,
                    'n_tokens': len(toks),
                    'n_words_raw': n_words_raw,
                })
                tokens_year.extend(toks)
                type_year_counts[broad][year] += 1
                n += 1
            except Exception as e:
                print(f'  Warning: {fname.name}: {e}')

        if tokens_year:
            years_data[year]  = tokens_year
            file_counts[year] = n
            print(f'  {year}: {n} talks, {len(tokens_year):,} tokens')

    return years_data, file_counts, type_year_counts, manifest


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
# FIGURE R1 — Demographics
# ═══════════════════════════════════════════════════════════════════════════════

def fig_r1_demographics(years, file_counts, years_data,
                        type_year_counts, manifest, out, fmt, dpi=300):

    all_years   = sorted(years)
    cats        = list(CONTENT_CATEGORIES.keys())
    cat_labels  = [CONTENT_CATEGORIES[c] for c in cats]
    cat_colors  = [CATEGORY_PALETTE[c]   for c in cats]

    # ── derived series ────────────────────────────────────────────────────────
    n_videos   = [file_counts.get(yr, 0) for yr in all_years]
    n_tokens   = [len(years_data.get(yr, [])) / 1000 for yr in all_years]  # thousands
    n_words_raw_by_year = defaultdict(int)
    for row in manifest:
        n_words_raw_by_year[row['year']] += row['n_words_raw']
    n_words_raw = [n_words_raw_by_year.get(yr, 0) / 1000 for yr in all_years]

    cumulative  = np.cumsum(n_videos)

    density = []  # mean tokens per video
    for yr in all_years:
        vids = [r for r in manifest if r['year'] == yr]
        density.append(np.mean([r['n_tokens'] for r in vids]) if vids else 0)

    # content-type matrix
    mat = np.array([[type_year_counts[c].get(yr, 0) for yr in all_years]
                    for c in cats], dtype=float)

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])   # A: videos + words per year
    ax_b = fig.add_subplot(gs[0, 1])   # B: cumulative
    ax_c = fig.add_subplot(gs[0, 2])   # C: token density
    ax_d = fig.add_subplot(gs[1, :2])  # D: stacked bar content types (wide)
    ax_e = fig.add_subplot(gs[1, 2])   # E: summary stats table

    bar_w = 0.68

    # ── A: videos per year + words line ───────────────────────────────────────
    ax_a.bar(all_years, n_videos, color='#D0D9E8', edgecolor='white',
             linewidth=0.4, width=bar_w, zorder=2, label='Videos')
    ax_a2 = ax_a.twinx()
    ax_a2.plot(all_years, n_words_raw, color=PALETTE[0], lw=2,
               marker='o', ms=7, zorder=3, label='Words (raw)')
    ax_a2.spines['top'].set_visible(False)
    ax_a2.tick_params(axis='y', labelcolor=PALETTE[0], labelsize=11.5)
    ax_a2.set_ylabel('Total words (thousands)', color=PALETTE[0], fontsize=12)
    ax_a.set_ylabel('Number of videos')
    ax_a.set_title('A   Videos and words per year', loc='left', pad=6)
    ax_a.set_xticks(all_years[::max(1, len(all_years)//6)])
    ax_a.set_xticklabels([str(y) for y in all_years[::max(1, len(all_years)//6)]],
                          rotation=45, ha='right', fontsize=11.5)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # annotate YouTube launch
    if 2016 in all_years:
        ax_a.axvline(2016, color='#D85A30', lw=1.2, ls='--', alpha=0.7)
        ax_a.text(2016.1, max(n_videos) * 0.92, 'YouTube\nlaunch',
                  fontsize=11, color='#D85A30')

    # ── B: cumulative video count ─────────────────────────────────────────────
    ax_b.fill_between(all_years, cumulative, alpha=0.15, color=PALETTE[0])
    ax_b.plot(all_years, cumulative, color=PALETTE[0], lw=2.2,
              marker='o', ms=7)
    ax_b.set_ylabel('Cumulative videos')
    ax_b.set_title('B   Cumulative channel growth', loc='left', pad=6)
    ax_b.set_xticks(all_years[::max(1, len(all_years)//6)])
    ax_b.set_xticklabels([str(y) for y in all_years[::max(1, len(all_years)//6)]],
                          rotation=45, ha='right', fontsize=11.5)
    # milestones
    for milestone in []:
        if milestone <= cumulative[-1]:
            ax_b.axhline(milestone, color='grey', lw=0.5, ls=':', alpha=0.5)
            ax_b.text(all_years[0] + 0.2, milestone + 3, str(milestone),
                      fontsize=10.5, color='grey')

    # ── C: token density (words per video) ───────────────────────────────────
    ax_c.bar(all_years, density, color=PALETTE[2], edgecolor='white',
             linewidth=0.4, width=bar_w, alpha=0.85)
    mean_density = np.mean([d for d in density if d > 0])
    ax_c.axhline(mean_density, color='#D85A30', lw=1.2, ls='--', alpha=0.8)
    ax_c.text(all_years[-1] + 0.1, mean_density + 5,
              f'mean\n{mean_density:.0f}', fontsize=11, color='#D85A30', va='bottom')
    ax_c.set_ylabel('Mean tokens per video')
    ax_c.set_title('C   Token density per video', loc='left', pad=6)
    ax_c.set_xticks(all_years[::max(1, len(all_years)//6)])
    ax_c.set_xticklabels([str(y) for y in all_years[::max(1, len(all_years)//6)]],
                          rotation=45, ha='right', fontsize=11.5)

    # ── D: stacked bar — content types ───────────────────────────────────────
    bottoms = np.zeros(len(all_years))
    for i, (cat, col, lbl) in enumerate(zip(cats, cat_colors, cat_labels)):
        vals = mat[i]
        ax_d.bar(all_years, vals, bottom=bottoms, color=col, width=bar_w,
                 label=lbl, edgecolor='white', linewidth=0.35, zorder=2)
        bottoms += vals

    # 2018-2019 gap band
    ax_d.axvspan(2017.5, 2019.5, color='#f0efe8', zorder=1, linewidth=0)
    ax_d.text(2018.5, bottoms.max() * 0.45, 'reduced\nactivity',
              ha='center', va='center', fontsize=11,
              color='#888780', style='italic')
    ax_d.set_xlabel('Year')
    ax_d.set_ylabel('Number of videos')
    ax_d.set_title('D   Content-type composition by year', loc='left', pad=6)
    ax_d.set_xticks(all_years)
    ax_d.set_xticklabels([str(y) for y in all_years],
                          rotation=45, ha='right', fontsize=11)
    ax_d.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_d.set_ylim(0, bottoms.max() * 1.4)
    ax_d.legend(loc='upper left', fontsize=11.5, ncol=2,
                bbox_to_anchor=(0.01, 0.99), borderpad=0.4)

    # ── E: summary statistics table ───────────────────────────────────────────
    ax_e.axis('off')
    ax_e.set_title('E   Corpus summary', loc='left', pad=6)

    total_videos = sum(n_videos)
    total_words  = sum(n_words_raw_by_year.values())
    total_tokens = sum(len(years_data.get(yr, [])) for yr in all_years)
    n_years      = len(all_years)
    yr_range     = f'{all_years[0]}–{all_years[-1]}'

    # per-category totals
    cat_totals = {c: sum(type_year_counts[c].values()) for c in cats}

    rows = [
        ['Metric', 'Value'],
        ['Years covered', yr_range],
        ['Total years with content', str(n_years)],
        ['Total videos', f'{total_videos:,}'],
        ['Total words (raw)', f'{total_words:,}'],
        ['Total tokens (preprocessed)', f'{total_tokens:,}'],
        ['Mean videos / year', f'{np.mean(n_videos):.1f}'],
        ['Mean words / video', f'{total_words / max(total_videos, 1):,.0f}'],
        ['─' * 22, '─' * 10],
    ] + [[f'  {CONTENT_CATEGORIES[c]}', str(cat_totals[c])] for c in cats]

    col_x = [0.0, 0.64]
    y_start, row_h = 0.97, min(0.072, 0.95 / len(rows))
    for ri, row in enumerate(rows):
        y = y_start - ri * row_h
        bold = ri == 0
        for ci, cell in enumerate(row):
            ax_e.text(col_x[ci], y, cell,
                      transform=ax_e.transAxes,
                      fontsize=11.5 if not bold else 12,
                      fontweight='bold' if bold else 'normal',
                      va='top', color='#2C2C2A' if ci == 0 else '#444440')

    fig.suptitle(
        'Sixteen years of open neuroscience communication: corpus characteristics and channel growth',
        fontsize=16, fontweight='bold', y=1.01)

    save(fig, 'r1_demographics', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def write_demographics_csv(years, file_counts, years_data,
                           type_year_counts, manifest, out):
    import csv
    os.makedirs(out, exist_ok=True)
    cats = list(CONTENT_CATEGORIES.keys())

    # per-year summary
    n_words_raw_by_year = defaultdict(int)
    for row in manifest:
        n_words_raw_by_year[row['year']] += row['n_words_raw']

    path = os.path.join(out, 'r1_yearly_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['year', 'n_videos', 'n_tokens', 'n_words_raw',
                    'cumulative_videos', 'mean_tokens_per_video']
                   + [CONTENT_CATEGORIES[c] for c in cats])
        cum = 0
        for yr in sorted(years):
            vids = [r for r in manifest if r['year'] == yr]
            n    = file_counts.get(yr, 0)
            cum += n
            tok  = len(years_data.get(yr, []))
            raw  = n_words_raw_by_year.get(yr, 0)
            mean_tok = np.mean([r['n_tokens'] for r in vids]) if vids else 0
            cat_counts = [type_year_counts[c].get(yr, 0) for c in cats]
            w.writerow([yr, n, tok, raw, cum, f'{mean_tok:.1f}'] + cat_counts)
    print(f'  ✓  r1_yearly_summary.csv')

    # full manifest
    path2 = os.path.join(out, 'r1_file_manifest.csv')
    with open(path2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['year','stem','broad','granular',
                                          'n_tokens','n_words_raw'])
        w.writeheader()
        w.writerows(manifest)
    print(f'  ✓  r1_file_manifest.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='CNSeminar Result 1 — Corpus demographics and channel growth')
    p.add_argument('--folder',  required=True,
                   help='Root folder with year subfolders of .txt transcripts')
    p.add_argument('--output',  default='./r1_figures')
    p.add_argument('--format',  choices=['pdf', 'png', 'both'], default='both')
    p.add_argument('--dpi',     type=int, default=300)
    p.add_argument('--start',   type=int, default=None)
    p.add_argument('--end',     type=int, default=None)
    args = p.parse_args()

    set_style(args.dpi)
    boost_style()
    print(f'\nLoading transcripts from: {args.folder}')
    years_data, file_counts, type_year_counts, manifest = \
        load_corpus(args.folder, args.start, args.end)

    if not years_data:
        print('No data found. Check folder structure (expects year/subfolders/*.txt).')
        sys.exit(1)

    years = sorted(years_data.keys())
    total = sum(len(t) for t in years_data.values())
    print(f'\n{len(years)} years: {years[0]}–{years[-1]}  |  {total:,} tokens total')
    print(f'Saving to: {args.output}\n')

    print('[1/2] Demographics figure...')
    fig_r1_demographics(years, file_counts, years_data,
                        type_year_counts, manifest,
                        args.output, args.format, args.dpi)

    print('[2/2] CSV summaries...')
    write_demographics_csv(years, file_counts, years_data,
                           type_year_counts, manifest, args.output)

    print(f'\nAll outputs saved to {args.output}')


if __name__ == '__main__':
    main()