"""
CNSeminar — Result 3: The Changing Vocabulary of Neuroscience
=============================================================
Analyses how the lexicon of the corpus has shifted over time.

Produces:
  r3a_tfidf_heatmap        — top TF-IDF terms per year (high-variance terms)
  r3b_vocabulary_growth    — cumulative unique vocabulary over time
  r3c_term_trajectories    — multi-panel trajectory plots for tracked terms
  r3d_emerging_terms       — terms with strongest sustained growth
  r3e_declining_terms      — terms with strongest sustained decline
  r3f_era_differential     — heatmap: early vs late era term salience shift
  r3_vocabulary_stats.csv  — year-level lexical diversity metrics
  r3_term_trajectories.csv — per-term per-year frequency (per 1k tokens)
  r3_emerging.csv          — ranked emerging terms with growth stats
  r3_declining.csv         — ranked declining terms with decline stats

Usage:
    python cns_r3_vocabulary.py --folder /path/to/transcripts --output ./r3_figures
    python cns_r3_vocabulary.py --folder /path/to/transcripts \\
        --track myelin machine_learning open_science fmri deep_learning
"""

import os, sys, argparse, warnings, csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import textwrap
from cns_figstyle import boost_style, declutter
from cns_analysis_pipeline import (
    tokenize, display, STOP, PALETTE, KEEP_SHORT,
    GREY_DARK, GREY_MID, GREY_FILL, GREY_LIGHT, ACCENT, set_style,
    load_corpus, freq_per_1k, smooth
)

DEFAULT_TRACK = [
    'myelin', 'language', 'diversity', 'axon', 'machine_learning',
    'consciousness', 'white_matter', 'virtual_reality', 'decoding',
    'cortex', 'plasticity', 'network', 'tractography', 'working_memory',
    'deep_learning', 'open_science', 'connectivity', 'embodiment',
    'multiple_sclerosis', 'resting_state', 'aphasia', 'bayesian',
    'preregistration', 'reproducibility', 'prediction',
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
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


def build_freq_matrix(years, years_data, vocab_size=2000, min_df=2):
    """
    Returns freq_matrix (years × vocab), vocab list.
    Frequencies are per-1k-tokens, normalised within year.
    """
    all_years = sorted(years)
    # build year-level TF-IDF
    corpora = [' '.join(years_data[yr]) for yr in all_years]
    vec = TfidfVectorizer(max_features=vocab_size, min_df=min_df,
                          stop_words=list(STOP))
    mat = vec.fit_transform(corpora).toarray()
    vocab = vec.get_feature_names_out()
    return mat, vocab, all_years


def term_trajectory(years, years_data, word):
    """Return per-1k frequency for `word` across years."""
    return [freq_per_1k(years_data[yr], word) for yr in years]


def linear_slope(vals):
    """Fit OLS slope across indices. Returns slope, r²."""
    x = np.arange(len(vals)).reshape(-1, 1)
    y = np.array(vals)
    if y.std() == 0:
        return 0.0, 0.0
    reg = LinearRegression().fit(x, y)
    ss_res = ((y - reg.predict(x))**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(reg.coef_[0]), r2


def classify_trajectory(vals, min_years=3, slope_thresh=0.01, r2_thresh=0.25):
    """
    Classify a frequency trajectory into one of:
      'emerging', 'declining', 'stable', 'peaked', 'resurgent', 'sparse'
    """
    if sum(v > 0 for v in vals) < min_years:
        return 'sparse'
    slope, r2 = linear_slope(vals)
    if r2 < r2_thresh:
        # check for peak then decline
        peak_idx = int(np.argmax(vals))
        if peak_idx < len(vals) * 0.65 and vals[-1] < vals[peak_idx] * 0.4:
            return 'peaked'
        return 'stable'
    if slope > slope_thresh:
        return 'emerging'
    if slope < -slope_thresh:
        return 'declining'
    return 'stable'


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R3A — TF-IDF heatmap (high-variance terms)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_tfidf_heatmap(years, years_data, out, fmt, dpi=300,
                      n_terms=45, vocab_size=500):
    all_years = sorted(years)
    corpora   = [' '.join(years_data[yr]) for yr in all_years]
    vec = TfidfVectorizer(max_features=vocab_size, min_df=1,
                          stop_words=list(STOP))
    mat  = vec.fit_transform(corpora).toarray()
    voc  = vec.get_feature_names_out()

    # select highest-variance terms across years
    var     = np.var(mat, axis=0)
    top_idx = np.argsort(var)[-n_terms:][::-1]
    h_terms = [display(voc[i]) for i in top_idx]
    h_data  = mat[:, top_idx].T          # terms × years

    n_yrs = len(all_years)
    fig_w = min(17, max(12, 5 + n_yrs * 0.6))
    fig, ax = plt.subplots(figsize=(fig_w, n_terms * 0.34 + 2))
    im   = ax.imshow(h_data, aspect='auto', cmap='Greys', interpolation='nearest')

    ax.set_xticks(range(n_yrs))
    ax.set_xticklabels([str(y) for y in all_years],
                        rotation=45, ha='right',
                        fontsize=15)
    ax.set_yticks(range(len(h_terms)))
    ax.set_yticklabels(h_terms, fontsize=15)
    ax.set_xlabel('Year')
    ax.set_title('Term salience over time (TF-IDF, highest-variance terms)',
                 fontsize=19.5, fontweight='bold', pad=8)

    # grid lines
    for i in range(len(h_terms) + 1):
        ax.axhline(i - 0.5, color='white', lw=0.3)
    for j in range(n_yrs + 1):
        ax.axvline(j - 0.5, color='white', lw=0.3)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label('TF-IDF weight', fontsize=15.5)
    cbar.ax.tick_params(labelsize=14.5)
    plt.tight_layout()
    save(fig, 'r3a_tfidf_heatmap', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R3B — Vocabulary growth
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vocabulary_growth(years, years_data, out, fmt, dpi=300):
    all_years  = sorted(years)
    cumulative_vocab = set()
    annual_vocab     = []
    cumulative_sizes = []

    for yr in all_years:
        yr_vocab = set(years_data[yr])
        annual_vocab.append(len(yr_vocab))
        cumulative_vocab |= yr_vocab
        cumulative_sizes.append(len(cumulative_vocab))

    # type-token ratio per year
    ttr = [len(set(years_data[yr])) / max(len(years_data[yr]), 1)
           for yr in all_years]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))

    # cumulative vocabulary
    axes[0].fill_between(all_years, cumulative_sizes, alpha=0.15, color=GREY_MID)
    axes[0].plot(all_years, cumulative_sizes, color=GREY_DARK, lw=2.2,
                 marker='o', ms=7)
    axes[0].set_ylabel('Cumulative unique types')
    axes[0].set_title('A   Cumulative vocabulary growth', loc='left', pad=6)
    axes[0].set_xticks(all_years[::max(1, len(all_years)//6)])
    axes[0].set_xticklabels(
        [str(y) for y in all_years[::max(1, len(all_years)//6)]],
        rotation=45, ha='right', fontsize=15)

    # annual unique types
    axes[1].bar(all_years, annual_vocab, color=GREY_FILL,
                edgecolor='white', linewidth=0.4, width=0.7)
    axes[1].set_ylabel('Unique word types per year')
    axes[1].set_title('B   Annual vocabulary richness', loc='left', pad=6)
    axes[1].set_xticks(all_years[::max(1, len(all_years)//6)])
    axes[1].set_xticklabels(
        [str(y) for y in all_years[::max(1, len(all_years)//6)]],
        rotation=45, ha='right', fontsize=15)
    axes[1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # type-token ratio
    axes[2].plot(all_years, ttr, color=GREY_DARK, lw=2, marker='s', ms=7)
    axes[2].fill_between(all_years, ttr, alpha=0.12, color=GREY_MID)
    axes[2].set_ylabel('Type–token ratio')
    axes[2].set_title('C   Lexical diversity (TTR)', loc='left', pad=6)
    axes[2].set_xticks(all_years[::max(1, len(all_years)//6)])
    axes[2].set_xticklabels(
        [str(y) for y in all_years[::max(1, len(all_years)//6)]],
        rotation=45, ha='right', fontsize=15)

    fig.suptitle('Vocabulary growth and lexical diversity, 2010–2026',
                 fontsize=19.5, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r3b_vocabulary_growth', out, fmt, dpi)

    return annual_vocab, cumulative_sizes, ttr


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R3C — Tracked keyword trajectories
# ═══════════════════════════════════════════════════════════════════════════════

def fig_term_trajectories(years, years_data, track_words, out, fmt, dpi=300):
    all_years = sorted(years)
    n    = len(track_words)
    ncol = min(5, n)
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 3.6, nrow * 3.5), sharey=False)
    flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, word in enumerate(track_words):
        ax    = flat[i]
        vals  = term_trajectory(all_years, years_data, word)
        smo   = smooth(vals, w=1)

        ax.fill_between(all_years, smo, alpha=0.12, color=GREY_MID)
        ax.plot(all_years, smo, color=GREY_DARK, lw=2.6, zorder=3)
        ax.scatter(all_years, vals, color=GREY_DARK, s=45, zorder=4, alpha=0.65)

        ax.set_title(display(word), fontsize=16, fontweight='bold', pad=3)
        step = max(1, len(all_years) // 5)
        ax.set_xticks(all_years[::step])
        ax.set_xticklabels([str(y)[2:] for y in all_years[::step]], fontsize=14.5)
        ax.tick_params(axis='y', labelsize=14.5)
        if i % ncol == 0:
            ax.set_ylabel('Freq / 1k words', fontsize=14.5)

    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    fig.suptitle(f'Keyword frequency trajectories, {all_years[0]}–{all_years[-1]} '
                 f'(per 1,000 tokens)',
                 fontsize=19.5, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r3c_term_trajectories', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R3D/E — Emerging and declining terms (data-driven)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_emerging_declining(years, years_data, out, fmt, dpi=300,
                           vocab_size=1500, top_n=20, min_years_active=3):
    all_years = sorted(years)
    corpora   = [' '.join(years_data[yr]) for yr in all_years]
    vec = TfidfVectorizer(max_features=vocab_size, min_df=2,
                          stop_words=list(STOP))
    mat   = vec.fit_transform(corpora).toarray()   # years × vocab
    vocab = vec.get_feature_names_out()

    slopes, r2s, means, actives = [], [], [], []
    for j in range(len(vocab)):
        vals = mat[:, j]
        active = (vals > 0).sum()
        s, r2  = linear_slope(vals)
        slopes.append(s)
        r2s.append(r2)
        means.append(vals.mean())
        actives.append(active)

    slopes  = np.array(slopes)
    r2s     = np.array(r2s)
    actives = np.array(actives)

    # filter: must be active in enough years and have reasonable fit
    mask_emerging = (actives >= min_years_active) & (r2s >= 0.20) & (slopes > 0)
    mask_declining = (actives >= min_years_active) & (r2s >= 0.10) & (slopes < 0)

    emerging_idx  = np.where(mask_emerging)[0]
    declining_idx = np.where(mask_declining)[0]

    emerging_sorted  = emerging_idx[np.argsort(slopes[emerging_idx])[::-1]][:top_n]
    declining_sorted = declining_idx[np.argsort(slopes[declining_idx])][:top_n]

    def panel(ax, indices, title, base_color, direction=1):
        if len(indices) == 0:
            ax.text(0.5, 0.5, 'No terms found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=17, color='grey')
            ax.set_title(title, loc='left', pad=6)
            return
        labels_plot = [display(vocab[i]) for i in indices]
        slope_vals  = slopes[indices] * direction  # positive for both bars
        colors = [base_color] * len(indices)

        # colour by r²
        norm_r2 = r2s[indices] / max(r2s[indices].max(), 1e-9)
        r, g, b = matplotlib.colors.to_rgb(base_color)
        bar_colors = [
            (r * (0.4 + 0.6 * nr), g * (0.4 + 0.6 * nr), b * (0.4 + 0.6 * nr))
            for nr in norm_r2
        ]

        bars = ax.barh(range(len(indices)), slope_vals,
                       color=bar_colors, edgecolor='white',
                       linewidth=0.4, height=0.7)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(labels_plot, fontsize=15)
        ax.set_xlabel('Linear slope (TF-IDF / year)', fontsize=15.5)
        ax.set_title(title, loc='left', pad=6)
        ax.set_xlim(0, slope_vals.max() * 1.3)
        for bar, idx in zip(bars, indices):
            ax.text(bar.get_width() + slope_vals.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'r²={r2s[idx]:.2f}',
                    va='center', fontsize=14.5, color='#555555')

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, top_n * 0.5 + 2)))
    panel(axes[0], emerging_sorted,
          f'A   Top {top_n} emerging terms',
          '#7A9E7E', direction=1)   # muted sage-green
    panel(axes[1], declining_sorted,
          f'B   Top {top_n} declining terms',
          '#B07D62', direction=-1)  # muted terracotta

    fig.suptitle('Emerging and declining vocabulary in CNSeminars (2010–2026)',
                 fontsize=19.5, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r3d_emerging_declining', out, fmt, dpi)

    return vocab, slopes, r2s, actives, mat, all_years


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R3F — Era differential heatmap (early vs late)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_era_differential(years, years_data, out, fmt, dpi=300,
                          vocab_size=500, n_terms=40):
    all_years = sorted(years)
    mid       = all_years[len(all_years) // 2]
    early     = [yr for yr in all_years if yr <= mid]
    late      = [yr for yr in all_years if yr > mid]

    if not early or not late:
        print('  Skipping era differential (insufficient year range)')
        return

    early_tok = [w for yr in early for w in years_data.get(yr, [])]
    late_tok  = [w for yr in late  for w in years_data.get(yr, [])]

    early_freq = Counter(early_tok)
    late_freq  = Counter(late_tok)
    total_e    = max(sum(early_freq.values()), 1)
    total_l    = max(sum(late_freq.values()), 1)

    # relative frequency shift: log2(late_rate / early_rate)
    all_words = set(early_freq) | set(late_freq)
    # restrict to reasonable vocabulary
    min_count = 5
    all_words = {w for w in all_words
                 if early_freq.get(w, 0) + late_freq.get(w, 0) >= min_count
                 and w not in STOP and (len(w) > 3 or w in KEEP_SHORT)}

    shifts = {}
    for w in all_words:
        re_ = (early_freq.get(w, 0) + 0.5) / total_e
        rl  = (late_freq.get(w,  0) + 0.5) / total_l
        shifts[w] = np.log2(rl / re_)

    sorted_shifts = sorted(shifts.items(), key=lambda x: x[1])
    bottom_n = sorted_shifts[:n_terms // 2]    # most declined
    top_n    = sorted_shifts[-(n_terms // 2):]  # most risen
    selected = bottom_n + top_n
    selected_terms  = [display(w) for w, _ in selected]
    selected_shifts = [v for _, v in selected]

    colors = ['#4A6FA5' if v > 0 else '#B07D62' for v in selected_shifts]

    fig, ax = plt.subplots(figsize=(10, max(8, len(selected) * 0.4 + 2)))
    bars = ax.barh(range(len(selected)), selected_shifts,
                   color=colors, edgecolor='white', linewidth=0.3, height=0.75)
    ax.axvline(0, color=GREY_DARK, lw=0.8)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected_terms, fontsize=15)
    ax.set_xlabel(f'Log₂ frequency shift (late {late[0]}–{late[-1]} vs early '
                  f'{early[0]}–{early[-1]})', fontsize=15.5)
    ax.set_title('Vocabulary shift between early and late corpus periods',
                 fontsize=19.5, fontweight='bold', pad=8)

    # legend
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color='#4A6FA5', label='More prominent in late period'),
        mpatches.Patch(color='#B07D62', label='More prominent in early period'),
    ], fontsize=15.5, loc='lower right')

    plt.tight_layout()
    save(fig, 'r3f_era_differential', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

def write_vocabulary_csvs(years, years_data, annual_vocab, cumulative_sizes,
                           ttr, out):
    os.makedirs(out, exist_ok=True)
    all_years = sorted(years)
    path = os.path.join(out, 'r3_vocabulary_stats.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['year', 'n_tokens', 'unique_types',
                    'cumulative_types', 'type_token_ratio'])
        for i, yr in enumerate(all_years):
            w.writerow([yr, len(years_data[yr]),
                        annual_vocab[i], cumulative_sizes[i],
                        f'{ttr[i]:.4f}'])
    print('  ✓  r3_vocabulary_stats.csv')


def write_trajectories_csv(years, years_data, track_words, out):
    os.makedirs(out, exist_ok=True)
    all_years = sorted(years)
    path = os.path.join(out, 'r3_term_trajectories.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['term'] + [str(yr) for yr in all_years] + ['trajectory'])
        for word in track_words:
            vals = term_trajectory(all_years, years_data, word)
            traj = classify_trajectory(vals)
            w.writerow([display(word)] + [f'{v:.4f}' for v in vals] + [traj])
    print('  ✓  r3_term_trajectories.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='CNSeminar Result 3 — Changing vocabulary of neuroscience')
    p.add_argument('--folder',   required=True)
    p.add_argument('--output',   default='./r3_figures')
    p.add_argument('--format',   choices=['pdf', 'png', 'both'], default='both')
    p.add_argument('--dpi',      type=int, default=300)
    p.add_argument('--start',    type=int, default=None)
    p.add_argument('--end',      type=int, default=None)
    p.add_argument('--track',    nargs='+', default=DEFAULT_TRACK,
                   help='Terms to plot trajectories for (underscore compounds ok)')
    p.add_argument('--vocab',    type=int, default=1500,
                   help='TF-IDF vocabulary size for trend analyses (default: 1500)')
    p.add_argument('--top_n',    type=int, default=20,
                   help='Number of emerging/declining terms to show (default: 20)')
    args = p.parse_args()

    set_style(args.dpi)
    boost_style()
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15,
                         'legend.fontsize': 15, 'figure.titlesize': 19})
    print(f'\nLoading transcripts from: {args.folder}')
    years_data, file_counts = load_corpus(args.folder, args.start, args.end)

    if not years_data:
        print('No data found.')
        sys.exit(1)

    years = sorted(years_data.keys())
    total = sum(len(t) for t in years_data.values())
    print(f'{len(years)} years: {years[0]}–{years[-1]}  |  {total:,} tokens')
    print(f'Saving to: {args.output}\n')

    print('[1/6] TF-IDF heatmap...')
    fig_tfidf_heatmap(years, years_data, args.output, args.format, args.dpi)

    print('[2/6] Vocabulary growth...')
    annual_vocab, cumulative_sizes, ttr = \
        fig_vocabulary_growth(years, years_data, args.output, args.format, args.dpi)

    print('[3/6] Keyword trajectories...')
    fig_term_trajectories(years, years_data, args.track,
                          args.output, args.format, args.dpi)

    print('[4/6] Emerging / declining terms...')
    fig_emerging_declining(years, years_data, args.output, args.format, args.dpi,
                           vocab_size=args.vocab, top_n=args.top_n)

    print('[5/6] Era differential heatmap...')
    fig_era_differential(years, years_data, args.output, args.format, args.dpi)

    print('[6/6] CSV outputs...')
    write_vocabulary_csvs(years, years_data, annual_vocab,
                          cumulative_sizes, ttr, args.output)
    write_trajectories_csv(years, years_data, args.track, args.output)

    print(f'\nAll outputs saved to {args.output}')


if __name__ == '__main__':
    main()