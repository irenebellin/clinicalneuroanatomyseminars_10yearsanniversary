#!/usr/bin/env python3
"""
CNSeminars YouTube channel metrics, 2017-2026.

Produces 300 dpi JPG figures:
  - fig_a_views.jpg ... fig_e_subs_per_1000_views.jpg  (one file per panel)
  - fig_combined_5panels.jpg                           (panels A-E stacked)

Requirements: pip install matplotlib pillow
Usage:        python make_channel_figures.py [output_dir]
"""

import itertools
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------------------------
# Data (edit here)
# ----------------------------------------------------------------------------
YEARS = list(range(2017, 2027))
VIEWS = [6609, 6074, 5693, 14353, 22379, 21768, 30437, 37430, 31510, 23711]
HOURS = [489, 486.8, 496, 1789.2, 3802.1, 1633.7, 2104.7, 2581.8, 1894.8, 1442.3]

# Average view duration as (minutes, seconds), converted to decimal minutes
AVG_MMSS = [(4, 26), (4, 48), (5, 13), (7, 28), (10, 11),
            (4, 30), (4, 8), (4, 8), (3, 36), (4, 0)]
AVG_MIN = [m + s / 60 for m, s in AVG_MMSS]

# Subscribers GAINED per year (net). Panel D is their running total.
SUBS_GAINED = [115, 120, 132, 430, 546, 559, 652, 871, 726, 501]
SUBS_CUM = list(itertools.accumulate(SUBS_GAINED))
SUBS_PER_1000_VIEWS = [s / v * 1000 for s, v in zip(SUBS_GAINED, VIEWS)]

# ----------------------------------------------------------------------------
# Style (edit here)
# ----------------------------------------------------------------------------
COLOR = "#0f766e"        # main colour (teal)
COLOR_LIGHT = "#a7d3cf"  # fill for the partial year (2026)
PARTIAL_HATCH = "///"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#000",
    "axes.linewidth": 1.1,
    "xtick.color": "#000",
    "ytick.color": "#000",
    "text.color": "#000",
    "axes.labelcolor": "#000",
})


def thousands(v, _):
    return f"{int(v):,}"


# (file stem, title, data, kind, y label, tick formatter, fixed y-limit)
PANELS = [
    ("a_views", "Views per year", VIEWS, "bar", "Views", thousands, None),
    ("b_hours", "Total hours watched", HOURS, "bar", "Hours watched", thousands, None),
    ("c_avg_watch_time", "Average watch time", AVG_MIN, "line", "Minutes", None, None),
    ("d_cumulative_subscribers", "Cumulative subscribers", SUBS_CUM, "line",
     "Cumulative subscribers", thousands, (0, 5000, 1000)),
    ("e_subs_per_1000_views", "Subscribers gained per 1,000 views", SUBS_PER_1000_VIEWS,
     "line", "Subscribers per 1,000 views", None, None),
]


def draw_panel(ax, title, data, kind, ylabel, fmt, ylim, letter=None):
    """Draw one panel. The last point (2026, partial year) is styled differently."""
    if kind == "bar":
        ax.bar(YEARS, data, color=[COLOR] * (len(YEARS) - 1) + [COLOR_LIGHT],
               width=0.7, edgecolor=COLOR, linewidth=0.9)
        ax.patches[-1].set_hatch(PARTIAL_HATCH)
    else:
        ax.plot(YEARS[:-1], data[:-1], color=COLOR, lw=2.4, marker="o", ms=6)
        ax.plot(YEARS[-2:], data[-2:], color=COLOR, lw=2.4, ls="--", marker="o",
                ms=6, mfc="white", mec=COLOR, mew=1.6, markevery=[1])

    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if ylim:
        lo, hi, step = ylim
        ax.set_ylim(lo, hi)
        ax.set_yticks(range(lo, hi + 1, step))
    if fmt:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    ax.grid(axis="y", color="#d9d9d9", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"{letter}  {title}" if letter else title,
                 loc="left", pad=8, fontsize=18, fontweight="bold")


def save_jpg(fig, path, dpi=300):
    """Save at 300 dpi as a high-quality JPEG (via a temporary PNG)."""
    tmp = path.replace(".jpg", ".png")
    fig.savefig(tmp, dpi=dpi, facecolor="white", bbox_inches="tight")
    Image.open(tmp).convert("RGB").save(path, "JPEG", quality=95,
                                        dpi=(dpi, dpi), subsampling=0)
    os.remove(tmp)
    plt.close(fig)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Individual panels
    for name, title, data, kind, ylabel, fmt, ylim in PANELS:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        draw_panel(ax, title, data, kind, ylabel, fmt, ylim)
        fig.subplots_adjust(bottom=0.22, left=0.15, right=0.97, top=0.87)
        save_jpg(fig, os.path.join(out_dir, f"fig_{name}.jpg"))

    # Combined figure (panels A-E)
    fig, axes = plt.subplots(5, 1, figsize=(7.5, 17))
    for ax, letter, panel in zip(axes, "ABCDE", PANELS):
        _, title, data, kind, ylabel, fmt, ylim = panel
        draw_panel(ax, title, data, kind, ylabel, fmt, ylim, letter)
    fig.tight_layout()
    save_jpg(fig, os.path.join(out_dir, "fig_combined_5panels.jpg"))

    print(f"Figures saved in: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figures")
