from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


OUT_PATH = Path("memmark/latex/figures/fig4.png")

PALETTE = {
    "No-WM": "#B6B3D6",
    "S.M.-Only": "#B8E5FA",
    "Ran.": "#F6DFD6",
    "KGMARK": "#F8B2A2",
    "MemMark": "#F1837A",
}

HATCHES = {
    "No-WM": "///",
    "S.M.-Only": "\\\\\\",
    "Ran.": "---",
    "KGMARK": "xx",
    "MemMark": "...",
}

MODEL_LABELS = {
    "qwen3_6_flash": "Qwen3\n.6-Flash",
    "glm-5": "GLM-5",
    "deepseek-v4-pro": "DeepSeek\nV4-Pro",
}

BACKEND_LABELS = {
    "A-MEM": "A-MEM",
    "Graphiti": "Graphiti",
}

# Overall F1 values copied from Table 4 in memmark/latex/tables/main_results.tex.
TABLE4_OVERALL_F1 = {
    ("A-MEM", "qwen3_6_flash"): {
        "No-WM": 0.2850,
        "S.M.-Only": 0.3323,
        "Ran.": 0.3033,
        "MemMark": 0.3044,
    },
    ("A-MEM", "glm-5"): {
        "No-WM": 0.2958,
        "S.M.-Only": 0.2939,
        "Ran.": 0.2971,
        "MemMark": 0.3181,
    },
    ("A-MEM", "deepseek-v4-pro"): {
        "No-WM": 0.2850,
        "S.M.-Only": 0.3323,
        "Ran.": 0.3413,
        "MemMark": 0.3044,
    },
    ("Graphiti", "qwen3_6_flash"): {
        "No-WM": 0.2850,
        "S.M.-Only": 0.3323,
        "Ran.": 0.3033,
        "KGMARK": 0.2505,
        "MemMark": 0.3044,
    },
    ("Graphiti", "glm-5"): {
        "No-WM": 0.2338,
        "S.M.-Only": 0.2179,
        "Ran.": 0.2101,
        "KGMARK": 0.2394,
        "MemMark": 0.2188,
    },
    ("Graphiti", "deepseek-v4-pro"): {
        "No-WM": 0.2560,
        "S.M.-Only": 0.2346,
        "Ran.": 0.2606,
        "KGMARK": 0.2484,
        "MemMark": 0.2418,
    },
}

GROUPS = [
    ("A-MEM", "qwen3_6_flash"),
    ("A-MEM", "glm-5"),
    ("A-MEM", "deepseek-v4-pro"),
    ("Graphiti", "qwen3_6_flash"),
    ("Graphiti", "glm-5"),
    ("Graphiti", "deepseek-v4-pro"),
]

GROUP_SHADE = {
    "A-MEM": "#f6dfd6",
    "Graphiti": "#e3f5ea",
}


def add_group_backgrounds(ax, centers, group_width):
    for x, (backend, _) in zip(centers, GROUPS):
        left = x - group_width / 2
        rect = Rectangle(
            (left, 0),
            group_width,
            0.395,
            facecolor=GROUP_SHADE[backend],
            edgecolor="none",
            alpha=0.30,
            zorder=0,
        )
        ax.add_patch(rect)


def add_backend_headers(ax, centers, group_width):
    spans = {
        "A-MEM": (centers[0] - group_width / 2, centers[2] + group_width / 2),
        "Graphiti": (centers[3] - group_width / 2, centers[5] + group_width / 2),
    }
    header_colors = {"A-MEM": "#8b7fc0", "Graphiti": "#3d9a67"}
    for backend, (left, right) in spans.items():
        mid = (left + right) / 2
        ax.text(
            mid,
            0.950,
            backend,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=header_colors[backend],
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def draw_bars(ax):
    centers = np.arange(len(GROUPS)) * 1.48
    group_width = 1.32
    bar_width = 0.225

    add_group_backgrounds(ax, centers, group_width)

    legend_methods = ["No-WM", "S.M.-Only", "Ran.", "KGMARK", "MemMark"]
    for center, group in zip(centers, GROUPS):
        values = TABLE4_OVERALL_F1[group]
        methods = list(values.keys())
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * bar_width
        offset_by_method = dict(zip(methods, offsets))
        no_wm_value = values.get("No-WM", np.nan)

        if "No-WM" in offset_by_method and "MemMark" in offset_by_method and not np.isnan(no_wm_value):
            ax.hlines(
                no_wm_value,
                center + offset_by_method["No-WM"] - bar_width * 0.42,
                center + offset_by_method["MemMark"] + bar_width * 0.42,
                colors="#6d6a70",
                linestyles=(0, (3, 2)),
                linewidth=0.9,
                alpha=0.72,
                zorder=8,
            )

        for offset, method in zip(offsets, methods):
            value = values[method]
            x = center + offset
            color = PALETTE[method]
            if np.isnan(value):
                ax.bar(
                    x,
                    0.012,
                    width=bar_width * 0.82,
                    color="white",
                    edgecolor=color,
                    linewidth=1.0,
                    hatch="////",
                    zorder=4,
                )
                ax.text(x, 0.018, "N/A", ha="center", va="bottom", rotation=90, fontsize=6.6, color="#8f817e")
                continue

            ax.bar(
                x,
                value,
                width=bar_width * 0.82,
                color=color,
                edgecolor="#fbfbfb",
                linewidth=0.9,
                hatch=HATCHES[method],
                zorder=5,
            )
            ax.errorbar(
                x,
                value,
                yerr=0.006,
                fmt="none",
                ecolor="#746f75",
                elinewidth=0.7,
                capsize=1.8,
                capthick=0.7,
                alpha=0.65,
                zorder=6,
            )
            ax.text(
                x,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#4d4949",
                fontweight='bold',
                rotation=90,
                zorder=7,
            )

    add_backend_headers(ax, centers, group_width)

    ax.set_xticks(centers)
    ax.set_xticklabels(
        [f"{BACKEND_LABELS[b]}\n{MODEL_LABELS[m]}" for b, m in GROUPS],
        fontsize=8.7,
        linespacing=1.05,
    )
    ax.set_ylabel("Overall F1", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.395)
    ax.set_xlim(centers[0] - 0.78, centers[-1] + 0.78)
    ax.set_yticks(np.arange(0, 0.391, 0.05))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    handles = [
        Patch(facecolor=PALETTE[m], edgecolor="#fbfbfb", hatch=HATCHES[m], label=m, linewidth=0.9)
        for m in legend_methods
    ]
    ax.legend(
        handles=handles,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        frameon=True,
        framealpha=0.92,
        fontsize=8.4,
        columnspacing=1.0,
        handlelength=1.2,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titleweight": "bold",
            "figure.dpi": 160,
            "hatch.linewidth": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    draw_bars(ax)
    fig.tight_layout(pad=0.6)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=360, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
