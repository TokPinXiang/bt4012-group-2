"""Standalone PNG/PDF figures for the shared evaluation report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

NAMES = {"lightgbm": "LightGBM", "gnn": "GNN"}
COLORS = {"lightgbm": "#2376B7", "gnn": "#D16A28"}


def _save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _thin(x, y, maximum=5000):
    # Plot rendering only: every metric is computed from ALL original scores.
    indices = np.unique(np.linspace(0, len(x) - 1, min(maximum, len(x)), dtype=int))
    return x[indices], y[indices]


def save_plots(aligned, comparison, budget, typologies, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    models = list(aligned)
    for split in ("val", "test"):
        title_split = "Validation" if split == "val" else "Test"
        results = comparison.loc[comparison.split == split].set_index("model")
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in models:
            d = aligned[name].loc[aligned[name].split == split]
            precision, recall, _ = precision_recall_curve(d.label, d.score)
            x, y = _thin(recall, precision)
            ax.plot(x, y, color=COLORS.get(name),
                    label=f"{NAMES.get(name, name)} (AP {results.loc[name, 'auprc']:.4f})")
        prevalence = float(results.iloc[0].prevalence)
        ax.axhline(prevalence, color="grey", linestyle=":", label=f"Prevalence ({prevalence:.3%})")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
               title=f"{title_split}: precision-recall comparison")
        ax.legend(loc="lower left")
        _save(fig, directory, f"precision_recall_{split}")

        fig, ax = plt.subplots(figsize=(7, 5))
        for name in models:
            d = aligned[name].loc[aligned[name].split == split]
            fpr, tpr, _ = roc_curve(d.label, d.score)
            x, y = _thin(fpr, tpr)
            ax.plot(x, y, color=COLORS.get(name),
                    label=f"{NAMES.get(name, name)} (AUC {results.loc[name, 'roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02),
               title=f"{title_split}: ROC comparison")
        ax.legend(loc="lower right")
        _save(fig, directory, f"roc_{split}")

        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
        for ax, name in zip(axes[0], models):
            m = results.loc[name]
            matrix = np.array([[m.tn, m.fp], [m.fn, m.tp]], dtype=int)
            ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:,}", ha="center", va="center",
                        color="white" if value > matrix.max() / 2 else "black", fontsize=12)
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Laundering"],
                   yticklabels=["Normal", "Laundering"], xlabel="Predicted", ylabel="Actual",
                   title=f"{NAMES.get(name, name)} | threshold {m.threshold:.6f}")
        fig.suptitle(f"{title_split}: confusion matrices at validation-selected thresholds")
        _save(fig, directory, f"confusion_matrices_{split}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name in models:
            b = budget.loc[(budget.model == name) & (budget.split == split)].sort_values("k")
            for ax, metric in zip(axes, ("precision", "recall")):
                ax.plot(b.k, b[metric], marker="o", label=NAMES.get(name, name), color=COLORS.get(name))
                ax.set(xlabel="Review budget k", ylabel=f"{metric.capitalize()}@k", ylim=(0, 1.05))
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_xticks(b.k)
        axes[0].legend()
        fig.suptitle(f"{title_split}: precision and recall at fixed review budgets")
        _save(fig, directory, f"review_budget_{split}")

        if not typologies.empty:
            t = typologies.loc[typologies.split == split]
            names = sorted(t.typology.unique())
            positions = np.arange(len(names))
            width = 0.8 / len(models)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.65 * len(names))))
            for j, name in enumerate(models):
                values = t.loc[t.model == name].set_index("typology").reindex(names)
                bars = ax.barh(positions + (j - (len(models) - 1) / 2) * width,
                               values.recall, height=width, label=NAMES.get(name, name), color=COLORS.get(name))
                ax.bar_label(bars, labels=[f"{v:.1%}" for v in values.recall], padding=3, fontsize=8)
            counts = t.loc[t.model == models[0]].set_index("typology").positives
            labels = [f"{'Unpatterned (unlisted)' if n == 'UNKNOWN_UNLISTED' else n} (n={int(counts[n])})" for n in names]
            ax.set(yticks=positions, yticklabels=labels, xlabel="Recall at each model's validation-selected threshold",
                   xlim=(0, 1.17), title=f"{title_split}: laundering detection by typology")
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.invert_yaxis()
            ax.legend(loc="lower right")
            _save(fig, directory, f"typology_recall_{split}")
