"""Monte Carlo evaluation harness (experimental methodology layer).

This module draws the line the project insists on between:

  * DEVELOPMENT runs  -- a single fixed seed, used for debugging and for the
    one-seed visualization figure in each experiment; and
  * REPORTED results  -- aggregated statistics over many INDEPENDENT trials,
    which is the only thing a quantitative claim in the README/writeup may cite.

A "trial function" maps one integer seed to a flat dict of scalar metrics. The
harness runs it over N independent seeds (spawned reproducibly from a single base
entropy via numpy.random.SeedSequence), stores every per-trial value, and reports
mean / std / min / max plus a 95% confidence interval per metric.

CI choice is metric-appropriate:
  * continuous metrics  -> Student-t interval on the mean;
  * proportion metrics  -> Wilson score interval (correct for 0/1 outcomes such
    as detection success, where the normal approximation is poor near 0 or 1).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import norm, t

# A trial maps a seed to named scalar metrics.
TrialFn = Callable[[int], dict[str, float]]


def make_seeds(base_entropy: int, n_trials: int) -> list[int]:
    """Spawn N independent, reproducible 32-bit seeds from one base entropy.

    SeedSequence.spawn produces statistically independent child sequences, the
    recommended way to seed many parallel/serial simulations without accidental
    correlation between streams.
    """
    ss = np.random.SeedSequence(base_entropy)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n_trials)]


@dataclass
class AggregateStat:
    """Aggregated statistics for one metric across all trials."""

    mean: float
    std: float          # sample standard deviation (ddof=1)
    min: float
    max: float
    n: int
    ci95_low: float
    ci95_high: float
    ci_method: str      # "student_t" or "wilson"

    def to_dict(self) -> dict:
        return {
            "mean": self.mean, "std": self.std, "min": self.min, "max": self.max,
            "n": self.n, "ci95_low": self.ci95_low, "ci95_high": self.ci95_high,
            "ci_method": self.ci_method,
        }


def _student_t_ci(values: np.ndarray) -> tuple[float, float]:
    n = values.size
    mean = float(values.mean())
    if n < 2:
        return mean, mean
    se = float(values.std(ddof=1)) / np.sqrt(n)
    crit = float(t.ppf(0.975, df=n - 1))
    return mean - crit * se, mean + crit * se


def _wilson_ci(values: np.ndarray) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion (values are 0/1)."""
    n = values.size
    k = float(values.sum())
    if n == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.975))
    denom = 1.0 + z**2 / n
    center = (k / n + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt((k / n) * (1 - k / n) / n + z**2 / (4 * n**2))
    return center - half, center + half


def aggregate(values: Sequence[float], proportion: bool = False) -> AggregateStat:
    """Aggregate one metric's per-trial values into summary statistics."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if proportion:
        lo, hi = _wilson_ci(arr)
        method = "wilson"
    else:
        lo, hi = _student_t_ci(arr)
        method = "student_t"
    return AggregateStat(float(arr.mean()), std, float(arr.min()), float(arr.max()),
                         n, float(lo), float(hi), method)


@dataclass
class MonteCarloResult:
    seeds: list[int]
    per_trial: list[dict[str, float]]            # one dict per trial
    aggregates: dict[str, AggregateStat]         # one AggregateStat per metric

    def aggregates_to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.aggregates.items()}


def run_monte_carlo(
    trial_fn: TrialFn,
    base_entropy: int,
    n_trials: int,
    proportion_keys: Iterable[str] = (),
) -> MonteCarloResult:
    """Run trial_fn over n_trials independent seeds and aggregate the results.

    Args:
        trial_fn: Maps a seed to a flat dict of scalar metrics. Must return the
            same keys every call.
        base_entropy: Root seed; the N trial seeds are spawned from it.
        n_trials: Number of independent trials.
        proportion_keys: Metric names that are 0/1 outcomes; these get a Wilson
            interval instead of a Student-t interval.

    Returns:
        MonteCarloResult with seeds, per-trial dicts, and per-metric aggregates.
    """
    seeds = make_seeds(base_entropy, n_trials)
    per_trial = [trial_fn(seed) for seed in seeds]

    keys = list(per_trial[0].keys())
    prop = set(proportion_keys)
    aggregates = {
        key: aggregate([trial[key] for trial in per_trial], proportion=key in prop)
        for key in keys
    }
    return MonteCarloResult(seeds, per_trial, aggregates)


def save_trials_csv(directory: Path, result: MonteCarloResult) -> Path:
    """Persist per-trial raw values to trials.csv (seed + every metric)."""
    path = Path(directory) / "trials.csv"
    keys = list(result.per_trial[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", *keys])
        for seed, trial in zip(result.seeds, result.per_trial):
            writer.writerow([seed, *[trial[k] for k in keys]])
    return path
