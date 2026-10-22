"""
CNSeminar — shared figure-readability settings
==============================================
Call boost_style() once, right AFTER set_style(), in every figure script.
It raises the default text sizes, tick sizes and line widths so that any
text that is not given an explicit fontsize is still easy to read.
Edit the numbers below to make everything globally bigger or smaller.
"""
import matplotlib.pyplot as plt


def boost_style():
    plt.rcParams.update({
        # text
        'font.size':          13,
        'axes.labelsize':     13,
        'axes.titlesize':     14.5,
        'xtick.labelsize':    12,
        'ytick.labelsize':    12,
        'legend.fontsize':    12,
        'legend.title_fontsize': 13,
        'figure.titlesize':   16,
        # lines / markers / ticks
        'lines.linewidth':    2.4,
        'lines.markersize':   7,
        'axes.linewidth':     1.1,
        'xtick.major.size':   5,
        'ytick.major.size':   5,
        'xtick.major.width':  1.0,
        'ytick.major.width':  1.0,
    })


# ───────────────────────────────────────────────────────────────────────────
# Axis decluttering: fewer tick numbers, thinned year labels.
# Called automatically from each script's save() just before saving.
# ───────────────────────────────────────────────────────────────────────────
import math, re
import matplotlib.ticker as mticker

MAX_NUMERIC_TICKS_Y = 5     # y axes with plain numbers
MAX_NUMERIC_TICKS_X = 5     # x axes with plain numbers
MAX_YEAR_LABELS     = 8     # year labels kept on any year axis


def declutter(fig):
    for ax in fig.axes:
        for name, axis in (('x', ax.xaxis), ('y', ax.yaxis)):
            loc = axis.get_major_locator()
            nb = MAX_NUMERIC_TICKS_Y if name == 'y' else MAX_NUMERIC_TICKS_X
            if isinstance(loc, mticker.AutoLocator):
                cur = axis.get_majorticklocs()
                is_int = len(cur) > 0 and all(abs(v - round(v)) < 1e-9 for v in cur)
                axis.set_major_locator(mticker.MaxNLocator(nbins=nb, integer=is_int))
            elif isinstance(loc, mticker.MaxNLocator):
                loc.set_params(nbins=nb)
            elif isinstance(loc, mticker.FixedLocator):
                pos = list(loc.locs)
                labs = [t.get_text() for t in axis.get_majorticklabels()]
                if (len(pos) > MAX_YEAR_LABELS and len(labs) == len(pos)
                        and all(re.fullmatch(r'\d{2,4}', l) for l in labs)):
                    k = math.ceil(len(pos) / MAX_YEAR_LABELS)
                    t0 = axis.get_majorticklabels()[0]
                    kw = dict(rotation=t0.get_rotation(), ha=t0.get_ha(),
                              fontsize=t0.get_fontsize())
                    getattr(ax, f'set_{name}ticks')(pos[::k])
                    getattr(ax, f'set_{name}ticklabels')(labs[::k], **kw)