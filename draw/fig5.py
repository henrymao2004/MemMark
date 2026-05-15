from pathlib import Path
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRACE_PATH = Path("watermark/results/amem_full/conv1.json")
OUT_PATH = Path("memmark/latex/figures/fig5.png")
LOW_ENTROPY_THRESHOLD = 0.2
MAX_ENTROPY = 2.0  # log2(4)

TAU_ORDER = ["update_target", "link_target", "semantic_realization"]
TAU_LABELS = {
    "update_target": "Update Target",
    "link_target": "Link Target",
    "semantic_realization": "Semantic Realization",
}
COLORS = {
    "update_target": "#9ccfcb",
    "link_target": "#f0b27a",
    "semantic_realization": "#c7b6e6",
}


def entropy_from_prob_dict(prob_dict):
    probs = [float(v) for v in prob_dict.values() if float(v) > 0]
    return -sum(p * math.log2(p) for p in probs)


def load_entropy_by_tau():
    obj = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    decisions = obj["details"]["watermark"]["decisions"]
    entropy_by_tau = {tau: [] for tau in TAU_ORDER}

    for decision in decisions:
        tau = decision.get("tau")
        prob_dict = decision.get("probabilities") or {}
        if tau not in entropy_by_tau or not isinstance(prob_dict, dict) or not prob_dict:
            continue
        entropy_by_tau[tau].append(entropy_from_prob_dict(prob_dict))

    return entropy_by_tau


def draw():
    entropy_by_tau = load_entropy_by_tau()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.dpi": 160,
            "axes.titleweight": "bold",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.6), sharey=True)
    bins = np.linspace(0, MAX_ENTROPY, 15)

    for ax, tau in zip(axes, TAU_ORDER):
        xs = np.array(entropy_by_tau[tau], dtype=float)
        mean = float(xs.mean())
        low_ratio = float((xs < LOW_ENTROPY_THRESHOLD).mean())

        ax.hist(
            xs,
            bins=bins,
            density=True,
            color=COLORS[tau],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
        )
        ax.axvline(mean, color="#5b4f4f", linestyle=(0, (3, 2)), linewidth=1.2)
        ax.axvline(LOW_ENTROPY_THRESHOLD, color="#8a8a8a", linestyle=":", linewidth=0.9)

        ax.set_title(TAU_LABELS[tau], fontsize=10.5)
        ax.set_xlim(0, MAX_ENTROPY)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)
        ax.text(
            0.98,
            0.95,
            f"n={len(xs)}\nmean={mean:.2f}\nH<0.2: {low_ratio:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d5d5d5", alpha=0.9),
        )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")

    axes[0].set_ylabel("Density", fontsize=10.5, fontweight="bold")
    fig.supxlabel(r"Entropy $H(\mathbf{p}_t)$", fontsize=10.5, fontweight="bold", y=0.03)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=1.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=360, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    draw()
