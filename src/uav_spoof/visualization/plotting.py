"""Plotting helpers (the 'visualization' layer).

Deliberately minimal: this project uses single-claim, publication-style figures
(one figure answers one research question). There are no multi-panel dashboards.
Each helper returns a (fig, ax) so the calling experiment owns saving via
`save_figure`, keeping output paths explicit and per-experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# A single, consistent style applied across every figure in the repository.
_STYLE = {
    "figure.figsize": (6.4, 4.6),
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
}


def apply_style() -> None:
    """Apply the repository-wide matplotlib style."""
    plt.rcParams.update(_STYLE)


def new_axes(title: str, xlabel: str, ylabel: str):
    """Create a single-axes figure with labels already set."""
    apply_style()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def save_figure(fig, path: str | Path) -> Path:
    """Save a figure to `path`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
