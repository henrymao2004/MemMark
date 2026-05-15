"""Figure: per-carrier realized capacity (bits embedded) across
(model, backend) settings. Companion to the entropy histogram —
shows that the available entropy is actually converted into bits.

Color palette: Okabe-Ito colorblind-safe.

Run:
    python draw/fig_carrier_bits.py
Outputs:
    latex/figures/carrier_bits.pdf
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- typography (ACL / Times-friendly) ---------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- data --------------------------------------------------------------
carriers = ["Update\nTarget", "Link\nTarget", "Semantic\nRealization"]
configs = [
    "A-Mem · GLM-5",
    "Graphiti · GLM-5",
    "Graphiti · DeepSeek-V4-Pro",
]

# bits embedded per (carrier, config)
bits = np.array([
    [11,  6,  5],   # update_target
    [13,  6,  5],   # link_target
    [16, 30, 30],   # semantic_realization
], dtype=float)

# decision counts (for n=X annotations)
n_decisions = np.array([
    [213, 374, 345],
    [199, 414, 400],
    [425, 890, 896],
], dtype=int)

PAYLOAD_TOTAL = 40

# --- Okabe-Ito colorblind-safe palette --------------------------------
# https://jfly.uni-koeln.de/color/
colors = ["#0072B2", "#D55E00", "#009E73"]   # blue, vermillion, bluish green

# --- figure ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.4))

n_carriers = len(carriers)
n_configs = len(configs)
bar_width = 0.26
x = np.arange(n_carriers)
offsets = (np.arange(n_configs) - (n_configs - 1) / 2) * bar_width

for j, (cfg, col) in enumerate(zip(configs, colors)):
    rects = ax.bar(
        x + offsets[j],
        bits[:, j],
        width=bar_width,
        color=col,
        edgecolor="white",
        linewidth=0.6,
        label=cfg,
        zorder=3,
    )
    # n=... annotations on top
    for i, rect in enumerate(rects):
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.6,
            f"n={n_decisions[i, j]}",
            ha="center", va="bottom",
            fontsize=7, color="#444444",
        )

# total payload reference line
ax.axhline(
    PAYLOAD_TOTAL,
    color="#666666",
    linestyle=(0, (4, 3)),
    linewidth=0.9,
    zorder=2,
)
ax.text(
    n_carriers - 0.55,
    PAYLOAD_TOTAL + 0.7,
    f"40-bit payload (Σ across carriers)",
    ha="right", va="bottom",
    fontsize=8, color="#666666", style="italic",
)

# axes cosmetics
ax.set_xticks(x)
ax.set_xticklabels(carriers)
ax.set_ylabel("Bits embedded")
ax.set_ylim(0, 45)
ax.yaxis.grid(True, linestyle=":", color="#cccccc", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# legend below the title area
leg = ax.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    ncol=1,
    handlelength=1.6,
)
leg.get_frame().set_linewidth(0.6)

plt.tight_layout()

# --- output ------------------------------------------------------------
out_path = Path(__file__).resolve().parent.parent / "latex" / "figures" / "carrier_bits.pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"wrote {out_path}")
