"""
CNSeminar — Results 8 & 9: Emerging-Topic Forecasting and Temporal Validation
=============================================================================
Asks whether patterns of topic growth observable up to year t can predict
continued growth at t+1 and t+2. Uses a rolling temporal window framework
with backtesting against observed prevalence.

Result 8 — Can the future be detected in the past?
  - Defines emerging topics operationally (sustained growth rate, first-
    appearance recency, acceleration)
  - Trains simple forecasting models on years ≤ t
  - Predicts prevalence at t+1 and t+2
  - Backtests predictions against observed values
  - Compares against a naïve persistence baseline
  - Reports performance metrics (MAE, rank correlation, hit rate)

Result 9 — Emerging terms for the year after the last observed year
  - Only generated if the model beats persistence (lower MAE AND hit rate
    above --hit_threshold)
  - Descriptive composite ranking (slope, acceleration, recency)

Input: topic prevalence data from r4_topic_dynamics.csv,
       OR term-level frequency data from r3_term_trajectories.csv.
       If neither is available, re-runs the term-level analysis internally.

Produces:
  r8a_backtest_performance    — scatter: predicted vs actual at t+1, t+2
  r8b_model_vs_baseline       — MAE comparison across rolling windows
  r8c_emerging_definition     — criteria dashboard (slope, acceleration,
                                first-year recency)
  r9_<year>_predictions       — ranked candidate emerging terms
                                (only generated if model beats persistence)
  r89_backtest_results.csv    — full backtest record
  r89_<year>_forecast.csv     — next-year emergence ranking

Usage:
    python cns_r89_forecasting.py --folder /path/to/transcripts
    python cns_r89_forecasting.py --folder /path/to/transcripts \\
        --trajectories ./r3_figures/r3_term_trajectories.csv
    python cns_r89_forecasting.py --folder /path/to/transcripts \\
        --topics ./r4_figures/r4_topic_dynamics.csv
"""

import os, sys, argparse, warnings, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import textwrap
from cns_figstyle import boost_style, declutter
from cns_analysis_pipeline import (
    STOP, PALETTE, GREY_DARK, GREY_MID, GREY_FILL, GREY_LIGHT, ACCENT,
    set_style, load_corpus, freq_per_1k, display
)
from sklearn.feature_extraction.text import TfidfVectorizer


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_trajectories_csv(path):
    """
    Read r3_term_trajectories.csv.
    Returns: terms (list), years (list of int), matrix (terms × years).
    """
    terms, years, matrix = [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            years_row = []
            vals_row  = []
            for k, v in row.items():
                if k.isdigit():
                    years_row.append(int(k))
                    try:
                        vals_row.append(float(v))
                    except ValueError:
                        vals_row.append(0.0)
            if not years:
                years = years_row
            terms.append(row['term'])
            matrix.append(vals_row)
    return terms, years, np.array(matrix)


def load_topic_csv(path):
    """
    Read r4_topic_dynamics.csv.
    Returns: topics (list of labels), years (list of int), matrix (topics × years).
    """
    topics, years, matrix = [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prev_cols = {k: v for k, v in row.items() if k.startswith('prev_')}
            if not years:
                years = [int(k.replace('prev_', '')) for k in prev_cols]
            topics.append(f"{row['cluster_id']}: {row['cluster_label']}")
            matrix.append([float(prev_cols[f'prev_{yr}']) for yr in years])
    return topics, years, np.array(matrix)


def compute_trajectories_from_corpus(years_data, vocab_size=1000, min_df=2):
    """
    Fallback: compute TF-IDF per-year term frequencies directly.
    Returns: terms, years, matrix (terms × years).
    """
    all_years = sorted(years_data.keys())
    corpora   = [' '.join(years_data[yr]) for yr in all_years]
    vec = TfidfVectorizer(max_features=vocab_size, min_df=min_df,
                          stop_words=list(STOP))
    mat = vec.fit_transform(corpora).toarray()  # years × vocab
    return list(vec.get_feature_names_out()), all_years, mat.T  # → vocab × years


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(series, window_years):
    """
    Given a time series (list of floats) over window_years indices,
    extract features used for prediction.

    Features:
        slope_full      OLS slope over full window
        slope_recent    OLS slope over last 3 years
        acceleration    slope_recent - slope_early
        mean_recent     mean of last 3 values
        last_value      most recent value
        recency_boost   proportion of activity in last 3 years
        nonzero_prop    proportion of years with non-zero value
    """
    n = len(series)
    if n < 3:
        return None

    x_full   = np.arange(n).reshape(-1, 1)
    y        = np.array(series, dtype=float)

    def slope(x_, y_):
        if len(y_) < 2 or y_.std() == 0:
            return 0.0
        return LinearRegression().fit(x_, y_).coef_[0]

    s_full   = slope(x_full, y)
    n_recent = min(3, n)
    s_recent = slope(np.arange(n_recent).reshape(-1, 1), y[-n_recent:])
    n_early  = max(1, n - n_recent)
    s_early  = slope(np.arange(n_early).reshape(-1, 1), y[:n_early])
    accel    = s_recent - s_early

    total    = y.sum()
    recent_share = y[-n_recent:].sum() / max(total, 1e-9)
    nonzero  = (y > 0).mean()

    return np.array([
        s_full, s_recent, accel,
        y[-3:].mean() if n >= 3 else y.mean(),
        y[-1],
        recent_share,
        nonzero,
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING WINDOW FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_backtest(terms, years, matrix,
                     min_train_years=4,
                     horizons=(1, 2)):
    """
    Leakage-free rolling-origin backtest.

    For each forecast origin t (the model may only see series[:t], i.e. the
    years up to and including years[t-1]) and each horizon h:
      - TRAIN on pairs (features from series[:s], target series[s-1+h])
        with s-1+h <= t-1, so every training target lies strictly before
        the origin;
      - PREDICT series[t-1+h], i.e. h years after the last observed year;
      - BASELINE = persistence: series[t-1].

    NB: years are indexed by their position in `years`; the 2018-2019 gap in
    the archive is therefore treated as one step.

    Returns list of dicts: term, train_end_year, test_year, horizon,
                           predicted, naive, actual, features.
    """
    n_years = len(years)
    results = []

    for h in horizons:
        for t in range(min_train_years, n_years - h + 1):
            X, Y = [], []
            for s in range(3, t - h + 1):
                for ti in range(len(terms)):
                    series = list(matrix[ti])
                    f = extract_features(series[:s], years[:s])
                    if f is None:
                        continue
                    X.append(f)
                    Y.append(series[s - 1 + h])
            if len(X) < 5:
                continue
            mdl = Ridge(alpha=1.0).fit(np.array(X), np.array(Y))

            for ti, term in enumerate(terms):
                series = list(matrix[ti])
                feats = extract_features(series[:t], years[:t])
                if feats is None:
                    continue
                results.append({
                    'term':           term,
                    'train_end_year': years[t - 1],
                    'test_year':      years[t - 1 + h],
                    'horizon':        h,
                    'predicted':      float(mdl.predict(feats.reshape(1, -1))[0]),
                    'naive':          series[t - 1],
                    'actual':         series[t - 1 + h],
                    'ma3':            float(np.mean(series[max(0, t - 3):t])),
                    'features':       feats,
                })

    return results


def compute_metrics(results, horizon=1):
    """MAE, Spearman rho, and directional hit rate vs the persistence baseline.

    The hit rate is computed only on cases where the observed value actually
    differs from the baseline (ties, e.g. a term absent in both years, carry
    no directional information). A two-sided binomial test against 0.5 is
    reported alongside."""
    from scipy.stats import binomtest
    sub = [r for r in results if r['horizon'] == horizon]
    if not sub:
        return {}

    preds   = np.array([r['predicted'] for r in sub])
    actuals = np.array([r['actual']    for r in sub])
    naives  = np.array([r['naive']     for r in sub])

    ma3s       = np.array([r['ma3'] for r in sub])
    mae_model  = np.mean(np.abs(preds   - actuals))
    mae_naive  = np.mean(np.abs(naives  - actuals))
    mae_ma3    = np.mean(np.abs(ma3s    - actuals))

    # within-origin rank skill: does the predicted CHANGE rank terms by their
    # observed change? (insensitive to the model's overall tendency to predict rises)
    rhos, better = [], 0
    by_origin = {}
    for r in sub:
        by_origin.setdefault(r['train_end_year'], []).append(r)
    for yr, g in by_origin.items():
        dp = [x['predicted'] - x['naive'] for x in g]
        da = [x['actual']    - x['naive'] for x in g]
        rr = spearmanr(dp, da)[0]
        if rr == rr:
            rhos.append(rr)
        better += int(np.mean([abs(x['predicted'] - x['actual']) for x in g]) <
                      np.mean([abs(x['naive'] - x['actual']) for x in g]))

    rho_model, _ = spearmanr(preds,  actuals)
    rho_naive, _ = spearmanr(naives, actuals)

    moving   = actuals != naives
    n_moving = int(moving.sum())
    hits     = int((np.sign(actuals[moving] - naives[moving]) ==
                    np.sign(preds[moving]   - naives[moving])).sum())
    hit_rate = hits / n_moving if n_moving else float('nan')
    hit_p    = binomtest(hits, n_moving, 0.5).pvalue if n_moving else float('nan')

    return {
        'horizon':    horizon,
        'n':          len(sub),
        'mae_model':  mae_model,
        'mae_naive':  mae_naive,
        'mae_improvement': (mae_naive - mae_model) / max(mae_naive, 1e-9),
        'rho_model':  rho_model,
        'rho_naive':  rho_naive,
        'hit_rate':   hit_rate,
        'naive_hit':  0.5,
        'n_moving':   n_moving,
        'hit_pvalue': hit_p,
        'mae_ma3':    mae_ma3,
        'mae_improvement_vs_ma3': (mae_ma3 - mae_model) / max(mae_ma3, 1e-9),
        'rank_skill': float(np.mean(rhos)) if rhos else float('nan'),
        'windows_better': f'{better}/{len(by_origin)}',
    }


def emergence_score(series):
    """The composite score used for the Result 9 ranking (same weights as
    forecast_next_year)."""
    ft = extract_features(series, None)
    if ft is None:
        return None
    slope_recent, accel, last_val, recent_share = ft[1], ft[2], ft[4], ft[5]
    return (2.0 * slope_recent + 1.0 * accel +
            0.5 * recent_share + 0.3 * (last_val > 0))


def score_backtest(terms, years, matrix, min_train_years=4,
                   horizons=(1, 2), n_perm=2000, seed=0):
    """
    Validate the Result 9 ranking score itself.

    At every forecast origin t the score is computed from series[:t] only;
    it is then correlated (Spearman) across terms with the realised change
    series[t-1+h] - series[t-1]. A useful 'emerging-term' score should show a
    POSITIVE mean correlation; a negative one means that recent risers tend to
    fall back (regression to the mean).
    Significance: permutation test (realised changes shuffled across terms
    within each origin), one-sided in the positive direction; p_negative is
    the one-sided test in the negative direction.
    Returns a list of dicts, one per horizon.
    """
    rng = np.random.default_rng(seed)
    n_years = len(years)
    out = []
    for h in horizons:
        per_origin = []
        for t in range(min_train_years, n_years - h + 1):
            sc, ch = [], []
            for ti in range(len(terms)):
                s_ = list(matrix[ti])
                v = emergence_score(s_[:t])
                if v is None:
                    continue
                sc.append(v)
                ch.append(s_[t - 1 + h] - s_[t - 1])
            if len(sc) > 5 and np.std(sc) > 0 and np.std(ch) > 0:
                per_origin.append((years[t - 1], np.array(sc), np.array(ch)))
        if not per_origin:
            continue
        rhos = np.array([spearmanr(sc, ch)[0] for _, sc, ch in per_origin])
        obs = float(np.nanmean(rhos))
        perm = np.array([
            np.nanmean([spearmanr(sc, rng.permutation(ch))[0]
                        for _, sc, ch in per_origin])
            for _ in range(n_perm)])
        out.append({
            'horizon':       h,
            'n_origins':     len(rhos),
            'mean_rho':      obs,
            'n_positive':    int((rhos > 0).sum()),
            'p_positive':    float((np.sum(perm >= obs) + 1) / (n_perm + 1)),
            'p_negative':    float((np.sum(perm <= obs) + 1) / (n_perm + 1)),
            'by_origin':     [(int(y), float(r)) for (y, _, _), r in zip(per_origin, rhos)],
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# NEXT-YEAR EMERGENCE RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_next_year(terms, years, matrix, top_n=20, min_active=3):
    """
    Descriptive emergence ranking from ALL available data (the last observed
    year is years[-1], so the ranking concerns years[-1] + 1).
    The score is a hand-weighted composite of recent slope, acceleration and
    recency; it is NOT the output of the ridge model.
    Returns sorted list of (term, score, features) dicts.
    """
    n_years = len(years)
    t       = n_years  # train on everything

    predictions = []
    for ti, term in enumerate(terms):
        series = list(matrix[ti])
        active = sum(v > 0 for v in series)
        if active < min_active:
            continue
        feats = extract_features(series, years)
        if feats is None:
            continue

        # score = combination of recent slope and acceleration
        slope_recent  = feats[1]
        accel         = feats[2]
        last_val      = feats[4]
        recent_share  = feats[5]

        # composite emergence score
        score = (2.0 * slope_recent + 1.0 * accel +
                 0.5 * recent_share + 0.3 * (last_val > 0))

        predictions.append({
            'term':          term,
            'score':         score,
            'slope_recent':  slope_recent,
            'acceleration':  accel,
            'last_value':    last_val,
            'recent_share':  recent_share,
            'active_years':  active,
        })

    predictions.sort(key=lambda x: x['score'], reverse=True)
    return predictions[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
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


# Axes of the backtest scatter are cropped at this percentile of all values so the
# dense core is readable (a few extreme points fall outside). None = show everything.
CROP_PERCENTILE = 99.5


def fig_backtest_scatter(results, metrics, out, fmt, dpi=300):
    horizons = sorted(set(r['horizon'] for r in results))
    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(7.5 * len(horizons), 7.5))
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        sub    = [r for r in results if r['horizon'] == h]
        preds  = [r['predicted'] for r in sub]
        acts   = [r['actual']    for r in sub]
        naives = [r['naive']     for r in sub]

        ax.scatter(acts, naives, color=GREY_LIGHT, s=12, alpha=0.35,
                   linewidths=0, label='Naïve (persistence)', zorder=2,
                   rasterized=True)
        ax.scatter(acts, preds,  color=GREY_DARK, s=14, alpha=0.45,
                   linewidths=0, label='Model (Ridge)',        zorder=3,
                   rasterized=True)

        allv = np.array(acts + preds + naives, dtype=float)
        hi = max(allv) if CROP_PERCENTILE is None else np.percentile(allv, CROP_PERCENTILE) * 1.15
        lims = [min(allv) * 0.95, hi * 1.05]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Observed value at t+' + str(h))
        ax.set_ylabel('Predicted value')

        m = metrics.get(h, {})
        info = (f"MAE model: {m.get('mae_model', 0):.4f}\n"
                f"MAE naïve: {m.get('mae_naive', 0):.4f}\n"
                f"Δ MAE: {m.get('mae_improvement', 0)*100:.1f}%\n"
                f"ρ model: {m.get('rho_model', 0):.3f}\n"
                f"Hit rate: {m.get('hit_rate', 0):.2f} "
                f"(chance: {m.get('naive_hit', 0.5):.2f})")
        ax.text(0.03, 0.97, info, transform=ax.transAxes,
                fontsize=11.5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC',
                          alpha=0.9))
        ax.set_title(f'Horizon t+{h}: predicted vs observed',
                     fontsize=13, fontweight='bold', pad=6)
        leg = ax.legend(fontsize=11.5, loc='center right', markerscale=3)
        for hnd in getattr(leg, 'legend_handles', getattr(leg, 'legendHandles', [])):
            hnd.set_alpha(1)

    fig.suptitle('Backtesting: forecasting model vs naïve baseline',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r8a_backtest_scatter', out, fmt, dpi)


def fig_mae_by_window(results, out, fmt, dpi=300):
    """MAE of model and naïve baseline per training window end-year."""
    horizons  = sorted(set(r['horizon'] for r in results))
    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(7.5 * len(horizons), 6))
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        sub      = [r for r in results if r['horizon'] == h]
        windows  = sorted(set(r['train_end_year'] for r in sub))
        mae_m, mae_n = [], []
        for w in windows:
            sw = [r for r in sub if r['train_end_year'] == w]
            mae_m.append(np.mean([abs(r['predicted'] - r['actual']) for r in sw]))
            mae_n.append(np.mean([abs(r['naive']     - r['actual']) for r in sw]))

        ax.plot(windows, mae_n, color=GREY_LIGHT, lw=3, marker='s',
                ms=8, label='Naïve baseline', ls='--')
        ax.plot(windows, mae_m, color=GREY_DARK, lw=3, marker='o',
                ms=8, label='Ridge model')
        ax.fill_between(windows,
                        [m - n for m, n in zip(mae_m, mae_n)],
                        [0] * len(windows),
                        where=[m < n for m, n in zip(mae_m, mae_n)],
                        alpha=0.12, color=GREY_MID,
                        label='Model better')
        ax.set_xlabel('Training window end year')
        ax.set_ylabel('Mean absolute error')
        ax.set_title(f'MAE by window (horizon t+{h})',
                     fontsize=13, fontweight='bold', pad=6)
        ax.legend(fontsize=11.5)

    fig.suptitle('Forecasting error by training window: model vs naïve baseline',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r8b_mae_by_window', out, fmt, dpi)


def fig_next_year_predictions(predictions, next_year, out, fmt, dpi=300):
    if not predictions:
        print(f'  Skipping r9_{next_year}_predictions (no candidates found)')
        return

    labels = [p['term'] for p in predictions]
    scores = [p['score'] for p in predictions]
    # colour by acceleration
    accels = np.array([p['acceleration'] for p in predictions])
    norm_a = (accels - accels.min()) / max(accels.max() - accels.min(), 1e-9)

    cmap   = matplotlib.colormaps['Greys']
    colors = [cmap(0.3 + 0.6 * a) for a in norm_a]

    fig, ax = plt.subplots(figsize=(11.5, max(6.5, len(labels) * 0.5 + 2)))
    bars = ax.barh(range(len(labels)), scores,
                   color=colors, edgecolor='white',
                   linewidth=0.4, height=0.72)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Emergence score (composite: slope + acceleration + recency)',
                  fontsize=12)
    ax.set_title(f'Terms with the strongest recent growth, for {next_year} (descriptive ranking)\n'
                 '(not a validated forecast of emergence: see Result 8)',
                 fontsize=13, fontweight='bold', pad=8)

    # colourbar for acceleration
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=matplotlib.colors.Normalize(vmin=accels.min(), vmax=accels.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label('Acceleration (recent slope − early slope)', fontsize=11.5)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    save(fig, f'r9_{next_year}_predictions', out, fmt, dpi)


def fig_growth_vs_reversion(predictions, score_res, next_year, out, fmt, dpi=300,
                            n_terms=15):
    """Left: terms with the strongest recent growth (descriptive ranking).
    Right: validation - correlation between that score and the term's actual
    change one / two years later, for every forecast origin."""
    if not predictions or not score_res:
        return
    BLUE, BLUE_L = '#4A6FA5', '#7A9BBF'
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.4)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ── Panel A ──────────────────────────────────────────────────────────────
    top = predictions[:n_terms][::-1]
    labels = [display(p['term']) for p in top]
    scores = [p['score'] for p in top]
    ypos = np.arange(len(top))
    ax1.barh(ypos, scores, color=BLUE_L, height=0.68, zorder=2)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(0, max(scores) * 1.08)
    ax1.set_xlabel('Growth score (recent slope + acceleration + recency)')
    ax1.xaxis.grid(True, color=GREY_FILL, lw=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title(f'A   Terms with the strongest recent growth\n'
                  f'      (as of {next_year - 1}; descriptive ranking)',
                  loc='left', fontsize=14)
    ax1.text(0.0, -0.15, 'Descriptive only: this ranking is not a forecast (see B).',
             transform=ax1.transAxes, fontsize=12, color=GREY_MID, style='italic')

    # ── Panel B ──────────────────────────────────────────────────────────────
    cols = {1: GREY_DARK, 2: GREY_LIGHT}
    width = 0.38
    origins = [y for y, _ in score_res[0]['by_origin']]
    allv = []
    for k, r in enumerate(score_res):
        ys = [y for y, _ in r['by_origin']]
        rs = [v for _, v in r['by_origin']]
        xs = np.array([origins.index(y) for y in ys]) + (k - 0.5) * width
        ax2.bar(xs, rs, width=width, color=cols.get(r['horizon'], GREY_MID),
                zorder=3, label=f't+{r["horizon"]}   mean $\\rho$ = {r["mean_rho"]:+.2f}   '
                                f'({r["n_origins"] - r["n_positive"]}/{r["n_origins"]} '
                                f'origins negative)')
        allv += rs
    ax2.axhline(0, color='black', lw=0.9, zorder=4)
    ax2.set_ylim(min(allv) * 1.18, 0.14 if max(allv) < 0.1 else max(allv) * 1.5)
    ax2.set_xticks(np.arange(len(origins)))
    ax2.set_xticklabels(origins, rotation=0, fontsize=12)
    ax2.yaxis.grid(True, color=GREY_FILL, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    # mark the 2018-2019 gap in the archive
    _lo, _hi = ax2.get_ylim()
    ax2.set_ylim(_lo, max(_hi, 0.32 * abs(_lo)))   # headroom for legend + gap note
    for i in range(len(origins) - 1):
        if origins[i + 1] - origins[i] > 1:
            ax2.axvline(i + 0.5, color=GREY_MID, lw=0.8, ls=':', zorder=1)
            ax2.text(i + 0.5, 0.0,
                     f'no transcripts\n{origins[i] + 1}\u2013{origins[i + 1] - 1}',
                     ha='center', va='bottom', fontsize=11, color=GREY_MID,
                     bbox=dict(fc='white', ec='none', pad=1.5))
    ax2.set_xlabel('Forecast origin (last year observed)')
    ax2.set_ylabel(r'Spearman $\rho$: growth score vs. later change')
    ax2.set_title('B   Did recent growth predict further growth?\n'
                  '      Negative = the fastest risers fell back',
                  loc='left', fontsize=14)
    ax2.legend(loc='upper left', ncol=1, fontsize=12, handlelength=1.2,
               bbox_to_anchor=(0.0, 1.0))
    n_all = sum(r['n_origins'] for r in score_res)
    n_neg = sum(r['n_origins'] - r['n_positive'] for r in score_res)
    ax2.text(0.99, 0.03, f'{n_neg} of {n_all} correlations are below zero',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=12.5,
             color=GREY_DARK, fontweight='bold')
    plt.tight_layout()
    save(fig, f'r9_{next_year}_growth_vs_reversion', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def write_backtest_csv(results, metrics, out):
    os.makedirs(out, exist_ok=True)

    path = os.path.join(out, 'r89_backtest_results.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['term', 'train_end_year', 'test_year', 'horizon',
                    'predicted', 'naive', 'actual',
                    'error_model', 'error_naive'])
        for r in results:
            w.writerow([
                r['term'], r['train_end_year'], r['test_year'], r['horizon'],
                f"{r['predicted']:.6f}", f"{r['naive']:.6f}", f"{r['actual']:.6f}",
                f"{abs(r['predicted'] - r['actual']):.6f}",
                f"{abs(r['naive']     - r['actual']):.6f}",
            ])
    print('  ✓  r89_backtest_results.csv')

    path2 = os.path.join(out, 'r89_metrics_summary.csv')
    with open(path2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'horizon', 'n', 'mae_model', 'mae_naive',
            'mae_improvement', 'rho_model', 'rho_naive',
            'hit_rate', 'naive_hit', 'n_moving', 'hit_pvalue',
            'mae_ma3', 'mae_improvement_vs_ma3', 'rank_skill', 'windows_better'])
        w.writeheader()
        for m in metrics.values():
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in m.items()})
    print('  ✓  r89_metrics_summary.csv')


def write_score_csv(score_res, out):
    path = os.path.join(out, 'r89_score_validation.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['horizon', 'n_origins', 'mean_rho',
                                          'n_positive', 'p_positive', 'p_negative'])
        w.writeheader()
        for r in score_res:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in r.items() if k != 'by_origin'})
    print('  ✓  r89_score_validation.csv')
    path2 = os.path.join(out, 'r89_score_validation_by_origin.csv')
    with open(path2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['horizon', 'origin_year', 'spearman_rho'])
        for r in score_res:
            for y, rho in r['by_origin']:
                w.writerow([r['horizon'], y, f'{rho:.4f}'])
    print('  ✓  r89_score_validation_by_origin.csv')


def write_trajectories_csv(terms, years, matrix, out):
    path = os.path.join(out, 'r89_trajectories_used.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['term'] + list(years))
        for ti, term in enumerate(terms):
            w.writerow([term] + [f'{v:.6f}' for v in matrix[ti]])
    print('  ✓  r89_trajectories_used.csv')


def write_forecast_csv(predictions, next_year, out, suffix=''):
    if not predictions:
        return
    path = os.path.join(out, f'r89_{next_year}_forecast{suffix}.csv')
    with open(path, 'w', newline='') as f:
        fieldnames = ['term', 'score', 'slope_recent', 'acceleration',
                      'last_value', 'recent_share', 'active_years']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in predictions:
            w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v)
                        for k, v in p.items()})
    print(f'  ✓  r89_{next_year}_forecast{suffix}.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='CNSeminar Results 8–9: Forecasting and temporal validation')
    p.add_argument('--folder',        required=True)
    p.add_argument('--output',        default='./r89_figures')
    p.add_argument('--format',        choices=['pdf', 'png', 'both'], default='both')
    p.add_argument('--dpi',           type=int, default=300)
    p.add_argument('--start',         type=int, default=None)
    p.add_argument('--end',           type=int, default=None)
    p.add_argument('--trajectories',  default=None,
                   help='Path to r3_term_trajectories.csv (preferred input)')
    p.add_argument('--topics',        default=None,
                   help='Path to r4_topic_dynamics.csv (alternative input)')
    p.add_argument('--vocab',         type=int, default=800,
                   help='Vocab size if computing trajectories from corpus (default: 800)')
    p.add_argument('--min_train',     type=int, default=4,
                   help='Minimum training years before first prediction (default: 4)')
    p.add_argument('--top_n',         type=int, default=20,
                   help='Top N terms to show in the next-year ranking (default: 20)')
    p.add_argument('--no_score_gate', action='store_true',
                   help='Generate Result 9 even if the ranking score is not '
                        'validated (label it purely descriptive)')
    p.add_argument('--hit_threshold', type=float, default=0.55,
                   help='Min directional hit rate (with MAE below baseline) to generate Result 9 (default: 0.55)')
    args = p.parse_args()

    set_style(args.dpi)
    boost_style()
    print(f'\nLoading transcripts from: {args.folder}')
    years_data, _ = load_corpus(args.folder, args.start, args.end)

    if not years_data:
        print('No data found.')
        sys.exit(1)

    # ── load or compute trajectory data ──────────────────────────────────────
    if args.trajectories:
        print(f'\nLoading term trajectories from: {args.trajectories}')
        terms, years, matrix = load_trajectories_csv(args.trajectories)
        label = 'terms'
    elif args.topics:
        print(f'\nLoading topic trajectories from: {args.topics}')
        terms, years, matrix = load_topic_csv(args.topics)
        label = 'topics'
    else:
        print(f'\nComputing term trajectories from corpus '
              f'(vocab={args.vocab})...')
        terms, years, matrix = compute_trajectories_from_corpus(
            years_data, vocab_size=args.vocab)
        label = 'terms'

    print(f'  {len(terms)} {label} × {len(years)} years')

    # ── backtesting ───────────────────────────────────────────────────────────
    print(f'\n[1/3] Rolling backtest (min_train={args.min_train} years)...')
    results = rolling_backtest(terms, years, matrix,
                               min_train_years=args.min_train,
                               horizons=(1, 2))
    print(f'  {len(results):,} backtest predictions generated')

    metrics = {h: compute_metrics(results, h) for h in (1, 2)}
    for h, m in metrics.items():
        if m:
            print(f'\n  Horizon t+{h}:')
            print(f'    MAE model:       {m["mae_model"]:.5f}')
            print(f'    MAE naïve:       {m["mae_naive"]:.5f}')
            print(f'    MAE improvement: {m["mae_improvement"]*100:.1f}%')
            print(f'    Spearman ρ:      {m["rho_model"]:.3f}')
            print(f'    MAE 3-yr mean:   {m["mae_ma3"]:.5f} '
                  f'(model vs 3-yr mean: {m["mae_improvement_vs_ma3"]*100:+.1f}%)')
            print(f'    Rank skill:      {m["rank_skill"]:.3f} '
                  f'(mean within-origin rho, predicted vs observed change)')
            print(f'    Windows model beats persistence: {m["windows_better"]}')
            print(f'    Hit rate:        {m["hit_rate"]:.3f} '
                  f'(chance: {m["naive_hit"]:.2f}; n={m["n_moving"]}, '
                  f'p={m["hit_pvalue"]:.3g})')

    # ── figures ───────────────────────────────────────────────────────────────
    print(f'\n[2/3] Producing figures...')
    fig_backtest_scatter(results, metrics, args.output, args.format, args.dpi)
    fig_mae_by_window(results, args.output, args.format, args.dpi)

    # ── Validate the ranking score used for Result 9 ──────────────────────────
    print('\n  Validating the Result 9 ranking score (does recent growth '
          'predict further growth?)...')
    score_res = score_backtest(terms, years, matrix,
                               min_train_years=args.min_train)
    for r in score_res:
        print(f'    t+{r["horizon"]}: mean Spearman rho = {r["mean_rho"]:+.3f} '
              f'({r["n_positive"]}/{r["n_origins"]} origins positive); '
              f'p(positive) = {r["p_positive"]:.3f}, '
              f'p(negative) = {r["p_negative"]:.3f}')
    score_ok = any(r['mean_rho'] > 0 and r['p_positive'] < 0.05 for r in score_res)
    os.makedirs(args.output, exist_ok=True)
    write_score_csv(score_res, args.output)
    write_trajectories_csv(terms, years, matrix, args.output)

    # ── Result 9: next-year emerging-term ranking ─────────────────────────────
    # Gate: (i) the ridge model beats persistence on error and calls the
    # direction of change better than --hit_threshold, AND (ii) the ranking
    # score itself shows a significantly positive relation with later change.
    next_year = years[-1] + 1
    passing = [h for h, m in metrics.items()
               if m and m['mae_improvement'] > 0
               and m['hit_rate'] >= args.hit_threshold]
    if passing and (score_ok or args.no_score_gate):
        print(f'\n[3/3] Model beats persistence at horizon(s) {passing} — '
              f'generating Result 9 ({next_year} ranking)...')
        predictions = forecast_next_year(terms, years, matrix, top_n=args.top_n)
        fig_next_year_predictions(predictions, next_year,
                                  args.output, args.format, args.dpi)
        write_forecast_csv(predictions, next_year, args.output)

        print(f'\n  Top emerging {label} for {next_year}:')
        for i, pred in enumerate(predictions[:10]):
            print(f'    {i+1:2d}. {pred["term"]:40s}  score={pred["score"]:.4f}  '
                  f'accel={pred["acceleration"]:.5f}')
    else:
        print(f'\n[3/3] Result 9 NOT generated: model beats persistence = '
              f'{bool(passing)}; ranking score validated = {score_ok}.')
        print('  Report Result 8 as it stands (see r89_metrics_summary.csv and '
              'r89_score_validation.csv); a list of "emerging" terms would not '
              'be supported by the backtest.')
        predictions = forecast_next_year(terms, years, matrix, top_n=args.top_n)
        write_forecast_csv(predictions, next_year, args.output,
                           suffix='_descriptive_unvalidated')
        fig_next_year_predictions(predictions, next_year,
                                  args.output, args.format, args.dpi)
        print('  Descriptive growth ranking saved (NOT a validated forecast); '
              'see r9_*_growth_vs_reversion for the combined figure.')

    # Combined, fully transparent figure: ranking + its backtest
    fig_growth_vs_reversion(predictions, score_res, next_year,
                            args.output, args.format, args.dpi)

    # ── CSV ───────────────────────────────────────────────────────────────────
    write_backtest_csv(results, metrics, args.output)

    print(f'\nAll outputs saved to {args.output}')


if __name__ == '__main__':
    main()