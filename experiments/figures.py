#!/usr/bin/env python3
"""Figures for the paper. Reads results/, writes paper/figures/*.pdf."""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colourblind-safe and legible in greyscale by line style.
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#555555"
RES = pathlib.Path("results")
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../paper/figures")


def load(name):
    return {r["name"]: r for r in json.loads((RES / name).read_text())["per_line"]}


def cer(run, lines):
    e = sum(run[n]["grapheme_edits"] for n in lines)
    return 100 * e / sum(run[n]["grapheme_ref"] for n in lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = json.loads((RES / "checkpoint_sweep.json").read_text())
    stock = load("stock.json")
    names = sorted(stock)
    stock_cer = cer(stock, names)

    v3 = {int(k): v for k, v in sweep["checkpoints"].items()}
    v2 = {int(k): v for k, v in sweep["v2_checkpoints"].items()}

    # ---- Figure 1: real-page CER against training iteration, both runs.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)

    for axis, data, title, colour in (
            (ax, v2, r"(a) $\mathtt{LANG\_TYPE}$ unset", ORANGE),
            (bx, v3, r"(b) $\mathtt{LANG\_TYPE}$=$\mathtt{Indic}$", BLUE)):
        its = sorted(data)
        axis.axhline(stock_cer, color=GREY, ls=":", lw=1.2, zorder=1)
        axis.annotate("stock Tesseract", xy=(its[0], stock_cer), xytext=(0, 4),
                      textcoords="offset points", fontsize=7, color=GREY)
        axis.plot(its, [100 * data[i]["cer"] for i in its], "-o", color=colour,
                  ms=3.5, lw=1.6, label="real pages", zorder=3)
        axis.plot(its, [100 * data[i]["train_bcer"] for i in its], "--s",
                  color=colour, ms=3, lw=1.1, alpha=0.45,
                  label="synthetic held-out", zorder=2)
        axis.set_xscale("log")
        axis.set_xlabel("training iteration")
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.25, lw=0.5)
        axis.legend(fontsize=7, frameon=False, loc="upper right")

    best = min(v3, key=lambda i: v3[i]["cer"])
    bx.annotate(f"{100 * v3[best]['cer']:.1f}%", xy=(best, 100 * v3[best]["cer"]),
                xytext=(6, -13), textcoords="offset points", fontsize=7.5,
                color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.set_ylabel("grapheme CER (%)")
    ax.set_ylim(0, 45)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: where the gap sits, by crop fill.
    quart = ["Q1", "Q2", "Q3", "Q4"]
    worst = max(v3, key=lambda i: i)
    series = [("stock", [100 * sweep["baselines"]["stock"][f"cer_{q}"] for q in quart], GREY),
              ("ours, exported checkpoint", [100 * v3[worst][f"cer_{q}"] for q in quart], ORANGE),
              ("ours, released model", [100 * v3[1400][f"cer_{q}"] for q in quart], BLUE)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    width = 0.26
    for k, (label, vals, colour) in enumerate(series):
        ax.bar([i + (k - 1) * width for i in range(4)], vals, width,
               label=label, color=colour, edgecolor="white", lw=0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["sparsest", "Q2", "Q3", "densest"], fontsize=8)
    ax.set_xlabel("test lines by crop fill")
    ax.set_ylabel("grapheme CER (%)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "crop_fill.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT}/training_curves.pdf and {OUT}/crop_fill.pdf")


if __name__ == "__main__":
    main()
