"""
CNSeminar — Results 5, 6, 7: Conceptual Reorganisation, Methods & Open Science
===============================================================================
Three related thematic analyses run from a single script:

Result 5 — Changing conceptual organisation
  Tracks pre-defined concept-pair oppositions that capture known
  paradigm shifts in neuroscience:
    localisation  →  network / connectivity
    group stats   →  individual / variability
    classic stats →  Bayesian / prediction / machine learning / AI
    task-based    →  resting state
    structure     →  function / dynamics

Result 6 — Changing methods, populations and resources
  Tracks term-category clusters:
    Modalities   : fMRI, dMRI, EEG, MEG, fNIRS, PET, TMS, ECoG
    Populations  : healthy, clinical, children/development, ageing, sex/gender
    Datasets     : HCP, UK Biobank, ABIDE, ADHD200, openneuro
    Computation  : deep learning, transformer, GPU, python, nipype

Result 7 — Open science enters scientific discourse
  Tracks open-science vocabulary:
    open science, preregistration, reproducibility, replication,
    effect size, statistical power, p-value, FAIR, data sharing,
    open access, GitHub, OSF, preprint

All three produce:
  r5_conceptual_shifts.pdf/png
  r6a_modalities.pdf/png
  r6b_populations.pdf/png
  r6c_datasets_tools.pdf/png
  r7_open_science.pdf/png
  r567_thematic_trajectories.csv

Usage:
    python cns_r567_thematic.py --folder /path/to/transcripts --output ./r567_figures
"""

import os, sys, argparse, warnings, csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import textwrap
from cns_figstyle import boost_style, declutter
from cns_analysis_pipeline import (
    tokenize, display, STOP, PALETTE,
    GREY_DARK, GREY_MID, GREY_FILL, ACCENT, set_style, load_corpus,
    freq_per_1k, smooth
)


# ═══════════════════════════════════════════════════════════════════════════════
# TERM CATEGORY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Result 5: Conceptual shifts ───────────────────────────────────────────────
# Each entry: (label, [old paradigm terms], [new paradigm terms], color_old, color_new)
CONCEPT_SHIFTS = [
    (
        'Localisation → Networks',
        ['localisation', 'localisation', 'lesion', 'area', 'region', 'focal'],
        ['network', 'connectivity', 'connectome', 'circuit', 'hub', 'functional_connectivity'],
        '#B07D62', '#4A6FA5',   # muted terracotta=older, slate-blue=newer
    ),
    (
        'Group → Individual / Variability',
        ['group', 'average', 'mean', 'population'],
        ['individual', 'variability', 'heterogeneity', 'stratification', 'precision'],
        '#B07D62', '#4A6FA5',
    ),
    (
        'Classical stats → Bayesian / ML',
        ['anova', 'regression', 't_test', 'p_value', 'null_hypothesis'],
        ['bayesian', 'machine_learning', 'deep_learning', 'prediction', 'decoding', 'classifier'],
        '#B07D62', '#4A6FA5',
    ),
    (
        'Task-based → Resting state',
        ['task', 'paradigm', 'stimulus', 'block_design', 'event_related'],
        ['resting_state', 'default_mode_network', 'default_mode', 'spontaneous', 'intrinsic'],
        '#B07D62', '#4A6FA5',
    ),
    (
        'Structure → Function / Dynamics',
        ['anatomy', 'morphology', 'grey_matter', 'volume', 'thickness'],
        ['dynamics', 'temporal', 'oscillation', 'functional', 'activity', 'signal'],
        '#B07D62', '#4A6FA5',
    ),
]

# ── Result 6: Methods, populations, resources ─────────────────────────────────
METHOD_CATEGORIES = {
    'Imaging modalities': {
        'MRI / fMRI':    ['mri', 'fmri', 'bold', 'blood_oxygen_level_dependent',
                          'haemodynamic', 'hemodynamic'],
        'dMRI / tractography': ['dmri', 'diffusion', 'tractography',
                                'diffusion_tensor_imaging', 'white_matter'],
        'EEG / MEG':     ['eeg', 'electroencephalography', 'meg',
                          'magnetoencephalography', 'erp', 'oscillation'],
        'fNIRS':         ['fnirs', 'functional_near_infrared', 'nirs'],
        'PET':           ['pet', 'positron_emission_tomography', 'tracer',
                          'amyloid', 'dopamine'],
        'TMS / tDCS':    ['tms', 'transcranial_magnetic_stimulation',
                          'tdcs', 'transcranial_direct_current', 'stimulation'],
    },
    'Study populations': {
        'Healthy adults':    ['healthy', 'control', 'volunteer', 'normative'],
        'Clinical':          ['patient', 'clinical', 'disorder', 'disease',
                              'lesion', 'stroke'],
        'Development / children': ['child', 'children', 'developmental',
                                   'adolescent', 'infant', 'paediatric'],
        'Ageing':            ['ageing', 'aging', 'elderly', 'lifespan',
                              'older', 'dementia'],
        'Sex / gender':      ['sex', 'gender', 'female', 'male', 'hormonal'],
    },
    'Datasets and tools': {
        'HCP':             ['human_connectome', 'hcp', 'connectome'],
        'UK Biobank':      ['uk_biobank', 'biobank', 'ukbiobank'],
        'Open datasets':   ['openneuro', 'abide', 'adhd200', 'openfmri',
                            'datalad', 'neurovault'],
        'Python / nipype': ['python', 'nipype', 'nilearn', 'fsl', 'freesurfer',
                            'ants', 'mriqc', 'fmriprep'],
        'Deep learning / GPU': ['deep_learning', 'convolutional_neural_network',
                                'transformer', 'pytorch', 'tensorflow',
                                'neural_network', 'gpu'],
    },
}

# ── Result 7: Open science ────────────────────────────────────────────────────
OPEN_SCIENCE_TERMS = {
    'Open science':      ['open_science', 'open_data', 'open_source', 'open_access'],
    'Preregistration':   ['preregistration', 'registered_report', 'osf',
                          'preprint', 'biorxiv', 'psyarxiv'],
    'Reproducibility':   ['reproducibility', 'replication', 'replicability',
                          'robustness', 'reliability'],
    'Statistics reform': ['effect_size', 'statistical_power', 'false_discovery_rate',
                          'multiple_comparison', 'bayesian', 'confidence_interval'],
    'FAIR / sharing':    ['fair_data', 'data_sharing', 'data_availability',
                          'github', 'zenodo', 'datalad', 'version_control'],
}


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


def category_freq(years_data, year, term_list):
    """Sum per-1k frequencies across all terms in a list."""
    tokens = years_data.get(year, [])
    total  = max(len(tokens), 1)
    counts = Counter(tokens)
    return sum(counts.get(t, 0) for t in term_list) / total * 1000


def smooth(vals, w=1):
    return [np.mean(vals[max(0, i-w):min(len(vals), i+w+1)])
            for i in range(len(vals))]


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R5 — Conceptual shifts
# ═══════════════════════════════════════════════════════════════════════════════

def fig_conceptual_shifts(years, years_data, out, fmt, dpi=300):
    all_years = sorted(years)
    n_shifts  = len(CONCEPT_SHIFTS)
    fig, axes = plt.subplots(n_shifts, 1,
                             figsize=(13, n_shifts * 3.5),
                             sharex=True)
    if n_shifts == 1:
        axes = [axes]

    for ax, (label, old_terms, new_terms, col_old, col_new) in \
            zip(axes, CONCEPT_SHIFTS):

        old_vals = smooth([category_freq(years_data, yr, old_terms)
                           for yr in all_years], w=1)
        new_vals = smooth([category_freq(years_data, yr, new_terms)
                           for yr in all_years], w=1)

        ax.fill_between(all_years, old_vals, alpha=0.18, color=col_old)
        ax.fill_between(all_years, new_vals, alpha=0.18, color=col_new)
        ax.plot(all_years, old_vals, color=col_old, lw=2.2,
                label='Earlier paradigm')
        ax.plot(all_years, new_vals, color=col_new, lw=2.2,
                label='Later paradigm')

        # crossover point
        for i in range(len(all_years) - 1):
            if (old_vals[i] > new_vals[i]) != (old_vals[i+1] > new_vals[i+1]):
                cross_yr = all_years[i] + 0.5
                ax.axvline(cross_yr, color='#888780', lw=0.9, ls='--', alpha=0.7)
                ax.text(cross_yr + 0.1,
                        max(max(old_vals), max(new_vals)) * 0.88,
                        'crossover', fontsize=10.5, color='#888780')

        ax.set_ylabel('Freq / 1k words', fontsize=11.5)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=3)
        ax.legend(fontsize=11.5, loc='upper right')
        ax.tick_params(labelsize=11.5)

    axes[-1].set_xlabel('Year')
    axes[-1].set_xticks(all_years[::max(1, len(all_years)//8)])
    axes[-1].set_xticklabels(
        [str(y) for y in all_years[::max(1, len(all_years)//8)]],
        rotation=45, ha='right', fontsize=11.5)

    fig.suptitle('Changing conceptual organisation of neuroscience (2010–2026)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r5_conceptual_shifts', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R6 — Methods, populations, resources (one sub-figure per group)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_method_category(years, years_data, group_name, group_dict,
                         fig_name, out, fmt, dpi=300):
    all_years = sorted(years)
    n_series  = len(group_dict)
    from cns_analysis_pipeline import CLUSTER_PALETTE

    fig, ax = plt.subplots(figsize=(14, 6.5))

    for i, (series_name, term_list) in enumerate(group_dict.items()):
        vals = [category_freq(years_data, yr, term_list) for yr in all_years]
        smo  = smooth(vals, w=1)
        col  = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ax.fill_between(all_years, smo, alpha=0.12, color=col)
        ax.plot(all_years, smo, color=col, lw=3, label=series_name)
        ax.scatter(all_years, vals, color=col, s=45, alpha=0.65, zorder=4)

    ax.set_xlabel('Year')
    ax.set_ylabel('Freq / 1k words')
    ax.set_title(f'{group_name} (2010–2026)',
                 fontsize=15, fontweight='bold', pad=8)
    ax.set_xticks(all_years[::max(1, len(all_years)//8)])
    ax.set_xticklabels(
        [str(y) for y in all_years[::max(1, len(all_years)//8)]],
        rotation=45, ha='right', fontsize=11.5)
    ax.legend(fontsize=11.5, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderpad=0.4)
    plt.tight_layout()
    save(fig, fig_name, out, fmt, dpi)


def fig_all_methods(years, years_data, out, fmt, dpi=300):
    names_figs = [
        ('Imaging modalities',       'modalities',   'r6a_modalities'),
        ('Study populations',        'populations',  'r6b_populations'),
        ('Datasets and tools',       'datasets',     'r6c_datasets_tools'),
    ]
    group_keys = list(METHOD_CATEGORIES.keys())
    for i, (label, _, fig_name) in enumerate(names_figs):
        key   = group_keys[i]
        group = METHOD_CATEGORIES[key]
        fig_method_category(years, years_data, label, group,
                            fig_name, out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE R7 — Open science discourse
# ═══════════════════════════════════════════════════════════════════════════════

def fig_open_science(years, years_data, out, fmt, dpi=300):
    all_years = sorted(years)
    n_series  = len(OPEN_SCIENCE_TERMS)
    from cns_analysis_pipeline import CLUSTER_PALETTE

    fig, axes = plt.subplots(1, 2, figsize=(17, 7),
                             gridspec_kw={'width_ratios': [3, 1]})

    # Panel A: trajectories
    ax = axes[0]
    all_series = {}
    for i, (name, terms) in enumerate(OPEN_SCIENCE_TERMS.items()):
        vals = [category_freq(years_data, yr, terms) for yr in all_years]
        smo  = smooth(vals, w=1)
        col  = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        all_series[name] = vals
        ax.fill_between(all_years, smo, alpha=0.12, color=col)
        ax.plot(all_years, smo, color=col, lw=3, label=name)
        ax.scatter(all_years, vals, color=col, s=45, alpha=0.65, zorder=4)

    ax.set_xlabel('Year')
    ax.set_ylabel('Freq / 1k words')
    ax.set_title('A   Open science vocabulary over time', loc='left', pad=6)
    ax.set_xticks(all_years[::max(1, len(all_years)//8)])
    ax.set_xticklabels(
        [str(y) for y in all_years[::max(1, len(all_years)//8)]],
        rotation=45, ha='right', fontsize=11.5)
    ax.legend(fontsize=11.5, loc='upper left')

    # Panel B: total open-science share per era
    ax2 = axes[1]
    # pool all open-science terms
    all_os_terms = [t for terms in OPEN_SCIENCE_TERMS.values() for t in terms]
    era_mid  = all_years[len(all_years) // 2]
    era_labels = [f'{all_years[0]}–{era_mid}', f'{era_mid+1}–{all_years[-1]}']
    era_ranges = [
        [yr for yr in all_years if yr <= era_mid],
        [yr for yr in all_years if yr > era_mid],
    ]

    os_share = []
    for era_yrs in era_ranges:
        vals_all = [category_freq(years_data, yr, all_os_terms) for yr in era_yrs]
        os_share.append(np.mean(vals_all))

    bars = ax2.bar(era_labels, os_share,
                   color=['#AAAAAA', '#4A6FA5'],   # grey=early, slate-blue=late
                   edgecolor='white', linewidth=0.4, width=0.5)
    for bar, val in zip(bars, os_share):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01 * max(os_share),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=12)
    ax2.set_ylabel('Mean freq / 1k words')
    ax2.set_title('B   Mean open-science\ndiscourse by era',
                  loc='left', pad=6, fontsize=13)

    fig.suptitle('Open science enters scientific discourse (2010–2026)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'r7_open_science', out, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def write_thematic_csv(years, years_data, out):
    os.makedirs(out, exist_ok=True)
    all_years = sorted(years)
    rows = []

    # concept shifts
    for label, old_terms, new_terms, _, _ in CONCEPT_SHIFTS:
        rows.append({'group': 'conceptual_shift', 'category': label + ' [OLD]',
                     'terms': '|'.join(old_terms),
                     **{str(yr): f'{category_freq(years_data, yr, old_terms):.4f}'
                        for yr in all_years}})
        rows.append({'group': 'conceptual_shift', 'category': label + ' [NEW]',
                     'terms': '|'.join(new_terms),
                     **{str(yr): f'{category_freq(years_data, yr, new_terms):.4f}'
                        for yr in all_years}})

    # methods / populations / datasets
    for group_name, group_dict in METHOD_CATEGORIES.items():
        for series_name, term_list in group_dict.items():
            rows.append({'group': group_name, 'category': series_name,
                         'terms': '|'.join(term_list),
                         **{str(yr): f'{category_freq(years_data, yr, term_list):.4f}'
                            for yr in all_years}})

    # open science
    for name, terms in OPEN_SCIENCE_TERMS.items():
        rows.append({'group': 'open_science', 'category': name,
                     'terms': '|'.join(terms),
                     **{str(yr): f'{category_freq(years_data, yr, terms):.4f}'
                        for yr in all_years}})

    path = os.path.join(out, 'r567_thematic_trajectories.csv')
    fieldnames = ['group', 'category', 'terms'] + [str(y) for y in all_years]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print('  ✓  r567_thematic_trajectories.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='CNSeminar Results 5–7: Thematic trajectory analyses')
    p.add_argument('--folder',  required=True)
    p.add_argument('--output',  default='./r567_figures')
    p.add_argument('--format',  choices=['pdf', 'png', 'both'], default='both')
    p.add_argument('--dpi',     type=int, default=300)
    p.add_argument('--start',   type=int, default=None)
    p.add_argument('--end',     type=int, default=None)
    args = p.parse_args()

    set_style(args.dpi)
    boost_style()
    print(f'\nLoading transcripts from: {args.folder}')
    years_data, file_counts = load_corpus(args.folder, args.start, args.end)

    if not years_data:
        print('No data found.')
        sys.exit(1)

    years = sorted(years_data.keys())
    total = sum(len(t) for t in years_data.values())
    print(f'{len(years)} years: {years[0]}–{years[-1]}  |  {total:,} tokens')
    print(f'Saving to: {args.output}\n')

    print('[1/5] Result 5 — Conceptual reorganisation...')
    fig_conceptual_shifts(years, years_data, args.output, args.format, args.dpi)

    print('[2/5] Result 6 — Imaging modalities...')
    print('[3/5] Result 6 — Study populations...')
    print('[4/5] Result 6 — Datasets and tools...')
    fig_all_methods(years, years_data, args.output, args.format, args.dpi)

    print('[5/5] Result 7 — Open science discourse...')
    fig_open_science(years, years_data, args.output, args.format, args.dpi)

    print('\nCSV output...')
    write_thematic_csv(years, years_data, args.output)

    print(f'\nAll outputs saved to {args.output}')
    print('\nNote: extend CONCEPT_SHIFTS, METHOD_CATEGORIES, or OPEN_SCIENCE_TERMS')
    print('at the top of this script to add or adjust any tracked term groups.')


if __name__ == '__main__':
    main()