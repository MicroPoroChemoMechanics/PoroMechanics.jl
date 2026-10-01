"""Regenerate the theory figures with Python, NumPy, and Matplotlib.

Run from any directory: python3 docs/src/assets/theory/draw_figures.py
Optional: --preview-dir /tmp/poromechanics-theory for PNG previews.
The curves are analytical illustrations, not simulation output or experimental data.
"""
from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "poro-theory-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preview-dir", type=Path)
args = parser.parse_args()
OUT = Path(__file__).resolve().parent
BLUE = "#0075ad"
INK = "#22384b"
ORANGE = "#b54c16"
plt.rcParams.update({
    "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "poro-theory",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name, title):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None, "Title": title})
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_dir / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                linewidth=2, color=color))


fig, ax = plt.subplots(figsize=(10, 4.3))
ax.set(xlim=(0, 10), ylim=(0, 4))
ax.axis("off")
ax.add_patch(Rectangle((0.35, 0.7), 3.3, 2.8, facecolor="#d7eef8", edgecolor=INK, linewidth=2))
for x, y, radius in [(0.95, 1.35, .44), (2, 1.3, .5), (3.05, 1.35, .42),
                     (1.1, 2.7, .5), (2.25, 2.65, .52), (3.15, 2.55, .36)]:
    ax.add_patch(Circle((x, y), radius, facecolor="#a8b1b9", edgecolor=INK))
ax.text(2, 3.72, "A volume containing many pores", ha="center", weight="bold")
ax.text(2, .25, "Gray: solid     Blue: pore water", ha="center")
arrow(ax, (3.95, 2.15), (5.35, 2.15), INK)
ax.text(4.65, 2.5, "Average", ha="center")
ax.add_patch(Rectangle((5.65, .7), 3.95, 2.8, facecolor="#eff6fa", edgecolor=INK, linewidth=2))
ax.text(7.62, 3.0, "Continuum description", ha="center", weight="bold")
ax.text(7.62, 2.4, r"Porosity: $n = V_p/V$", ha="center")
ax.text(7.62, 1.8, r"Pressure: $p$", ha="center")
ax.text(7.62, 1.2, r"Displacement: $\mathbf{u}$", ha="center")
save(fig, "porous-volume", "Averaging a saturated porous volume")

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set(xlim=(0, 10), ylim=(0, 3.3))
ax.axis("off")
ax.add_patch(Rectangle((2.25, 1), 5.5, 1.5, facecolor="#d7eef8", edgecolor=INK, linewidth=2))
for x in np.linspace(2.65, 7.35, 8):
    for y in [1.3, 2.2]:
        ax.add_patch(Circle((x, y), .13, facecolor="#a8b1b9", edgecolor="none"))
ax.text(1.05, 1.85, "Higher\npressure", ha="center", va="center")
ax.text(8.95, 1.85, "Lower\npressure", ha="center", va="center")
arrow(ax, (1.7, 1.75), (2.75, 1.75))
arrow(ax, (4.2, 1.75), (5.8, 1.75))
arrow(ax, (7.25, 1.75), (8.3, 1.75))
ax.text(5, 2.8, r"Darcy flux $q_x$ through total cross section $A$", ha="center")
ax.annotate("", (2.25, .62), (7.75, .62), arrowprops={"arrowstyle": "|-|", "color": INK})
ax.text(5, .18, r"Length $L$       Volume flow rate $Q = A q_x$", ha="center")
save(fig, "darcy-column", "Pressure-driven Darcy flow through a horizontal column")

fig, (ax, curves) = plt.subplots(1, 2, figsize=(10, 5.3), gridspec_kw={"width_ratios": [1, 1.35]}, layout="constrained")
ax.set(xlim=(0, 4.2), ylim=(0, 5.2))
ax.axis("off")
ax.add_patch(Rectangle((1, .9), 2, 3, facecolor="#d7eef8", edgecolor=INK, linewidth=2))
ax.plot([.85, .85], [.9, 3.9], color=INK, linewidth=4)
ax.plot([3.15, 3.15], [.9, 3.9], color=INK, linewidth=4)
ax.plot([.85, 3.15], [.8, .8], color=INK, linewidth=5)
for x in [1.25, 2, 2.75]:
    arrow(ax, (x, 4.85), (x, 4.05), ORANGE)
for x in [1.55, 2.45]:
    arrow(ax, (x, 3.55), (x, 4.15))
ax.text(2, 5.0, r"Load $P$", ha="center", weight="bold", color=ORANGE)
ax.text(2, 3.1, "Saturated\ncolumn", ha="center", va="center")
ax.text(2, 2.05, "Sides confined\n(no lateral strain)", ha="center", va="center")
ax.text(2, .35, "Base fixed and sealed", ha="center")
ax.text(.05, 4.05, r"$p=0$", ha="left")
Z = np.linspace(0, 1, 350)
a = (np.arange(160) + .5) * np.pi
for tau, color, style in [(0.01, "#0075ad", "-"), (0.1, "#b54c16", "--"),
                           (0.5, "#397a38", "-."), (1.0, "#7d4a9a", ":")]:
    p = np.sum((2/a * np.exp(-a*a*tau))[:, None] * np.sin(a[:, None]*Z), axis=0)
    curves.plot(p, Z, color=color, linestyle=style, linewidth=2.3, label=rf"$\tau={tau:g}$")
curves.set(xlim=(0, 1.03), ylim=(1, 0), xlabel=r"Excess pressure $p/p_0$",
           ylabel=r"Depth from drain $z/L$")
curves.set_title(r"Analytical profiles: $\tau=c_v t/L^2$", fontsize=13, pad=14)
curves.grid(alpha=.2)
curves.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=4,
              fontsize=10, frameon=False, handlelength=1.5, columnspacing=1.0)
save(fig, "consolidation", "Single-drain Terzaghi consolidation and analytical pressure profiles")

fig, ax = plt.subplots(figsize=(8.5, 4.7), layout="constrained")
pc = np.geomspace(1e2, 1e7, 450)
S = (1+(pc/1e5)**2)**(-.5)
krl = np.sqrt(S)*(1-np.sqrt(1-S**2))**2
ax.semilogx(pc, S, color=BLUE, linewidth=2.5, label=r"Liquid saturation $S_l$")
ax.semilogx(pc, krl, color=ORANGE, linestyle="--", linewidth=2.5, label=r"Relative permeability $k_{rl}$")
ax.set(xlabel=r"Capillary pressure $p_c$ [Pa]", ylabel="Dimensionless value", ylim=(-.025, 1.06))
ax.set_title(r"Illustration: $a=10^5$ Pa, $m=0.5$, $n=2$", pad=12)
ax.grid(alpha=.2)
ax.legend(loc="upper right")
save(fig, "retention", "Van Genuchten retention and Mualem relative permeability")
