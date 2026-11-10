"""Plot the audited first differing token by return cycle."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABELS = {"gsm8k": "GSM8K", "dialogsum": "DialogSum"}
MARKED = {"gsm8k": "test:1122", "dialogsum": "test_254"}
PALETTE = [("#285580", "o", "#285580"), ("#08796c", "s", "white")]


def render(points, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 8.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#212a36",
        "xtick.color": "#253545", "ytick.color": "#253545",
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.26, 3.12), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.18, right=0.973, bottom=0.155, top=0.832, hspace=0.35)
    max_index = max((row["first_difference"] for task in points.values() for row in task), default=0)
    ymax = 142 if max_index <= 142 else 270
    for ax, task in zip(axes, LABELS):
        for block, (edge, marker, face) in enumerate(PALETTE, start=1):
            subset = [row for row in points[task] if row["block"] == block]
            ax.scatter([r["cycle"] for r in subset], [r["first_difference"] for r in subset],
                       marker=marker, s=21, facecolors=face, edgecolors=edge, linewidths=0.9, zorder=3)
        marked = [row for row in points[task] if row["id"] == MARKED[task]]
        ax.scatter([r["cycle"] for r in marked], [r["first_difference"] for r in marked],
                   marker="o", s=78, facecolors="none", edgecolors="#bb4834", linewidths=0.95, zorder=4)
        ax.set_title(LABELS[task], loc="left", fontsize=8, fontweight="bold", pad=3)
        ax.set(xlim=(0, 130), ylim=(0, ymax), xticks=[0, 32, 64, 96, 128],
               yticks=list(range(0, ymax, 64)))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e8ef", linewidth=0.65)
        for key in ("bottom", "left"):
            ax.spines[key].set_color("#657482")
            ax.spines[key].set_linewidth(0.6)
        ax.tick_params(length=2.4, width=0.55)
    axes[-1].set_xlabel("Return cycle", labelpad=3)
    fig.text(0.026, 0.50, "First differing token", ha="center", va="center", rotation="vertical")
    handles = [
        Line2D([0], [0], marker=marker, markersize=4.6, markeredgecolor=edge,
               markerfacecolor=face, linestyle="None", label=f"Block {block}")
        for block, (edge, marker, face) in enumerate(PALETTE, start=1)
    ]
    handles.append(Line2D([0], [0], marker="o", markersize=6.8, markeredgecolor="#bb4834",
                          markerfacecolor="none", linestyle="None", label="Same prompt pair"))
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.57, 0.976),
               ncol=3, handletextpad=0.3, columnspacing=1.0, borderaxespad=0, fontsize=7.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Creator": "Matplotlib"})
    plt.close(fig)
