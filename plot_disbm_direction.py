"""
Direction of the recovered drift on a DiSBM, as a function of the asymmetry p - q of the flow
between communities (r fixed): (a) signed cosine between the node drifts and the forward direction
of the cycle of communities, (b) estimated strength of the asymmetry ||hat c||_{hat g_BL}.
Lines are medians over the nodes of all seeds, bands their 10-90% range.

    python plot_disbm_direction.py [--seeds 5] [--out disbm_direction]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from finsler_embedding.evaluate import disbm_direction

SERIES = "#2a78d6"  # categorical slot 1 of the reference palette (validated)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def sweep(p_minus_q, r, K, N, seeds):
    stats = {"cos": [], "c_norm": []}
    for d in p_minus_q:
        p = (1 - r + d) / 2  # p + q = 1 - r
        cos, c_norm = [], []
        for seed in range(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            out = disbm_direction(p, r, K, N)
            cos.append(out["cos_node"].numpy())
            c_norm.append(out["c_norm"].numpy())
        for key, vals in (("cos", cos), ("c_norm", c_norm)):
            vals = np.concatenate(vals)
            stats[key].append(np.quantile(vals, [0.1, 0.5, 0.9]))
        print(f"p - q = {d:+.1f}: median cos {stats['cos'][-1][1]:+.3f}, "
              f"median ||c|| {stats['c_norm'][-1][1]:.3f}", flush=True)
    return {k: np.array(v) for k, v in stats.items()}


def panel(ax, x, q, ylabel):
    ax.fill_between(x, q[:, 0], q[:, 2], color=SERIES, alpha=0.18, linewidth=0)
    ax.plot(x, q[:, 1], color=SERIES, linewidth=1.5, marker="o", markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlabel(r"$p - q$", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--out", default="disbm_direction", help="Prefix: writes <out>_cos.pdf and <out>_cnorm.pdf.")
    args = parser.parse_args()

    # Regular grid, refined near p = q to show how small an asymmetry the drift picks up
    x = np.unique(np.round(np.concatenate([np.arange(-0.8, 0.81, 0.1), [-0.05, -0.02, 0.02, 0.05]]), 2)) + 0.0
    stats = sweep(x, args.r, args.K, args.N, args.seeds)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9})
    # One file per panel, so that the paper labels them with subfigures. Both files share the same
    # size and axes position, so that the panels line up once scaled to the same width.
    def new_panel():
        fig, ax = plt.subplots(figsize=(2.75, 2.1))
        fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.96)
        return fig, ax

    fig, ax = new_panel()
    panel(ax, x, stats["cos"], "signed cosine")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    # Meaning of the sign, in the empty corners (the curve is at -1 on the left, +1 on the right)
    ax.text(x[0], 0.85, "drift forward", color=MUTED, fontsize=7.5, ha="left", va="center")
    ax.text(x[-1], -0.85, "drift backward", color=MUTED, fontsize=7.5, ha="right", va="center")
    fig.savefig(f"{args.out}_cos.pdf")

    fig, ax = new_panel()
    panel(ax, x, stats["c_norm"], r"$\|\hat{c}\|_{\hat{g}_{BL}}$")
    ax.set_ylim(bottom=0)
    fig.savefig(f"{args.out}_cnorm.pdf")
    print(f"Saved {args.out}_cos.pdf and {args.out}_cnorm.pdf")


if __name__ == "__main__":
    main()
