"""Is the winner of a search a location or just the best of a bad lot?

A search scores thousands of flight pairs, at most one of which is right; the rest are
wrong by construction and scored against the same audio, so their scores are the null
distribution for this recording: what a wrong pair looks like here. The winner is judged
against that crowd rather than against the runner-up alone (a margin on its own means
nothing, since every search has a winner, including one over a box that cannot contain
the answer).

Model: above a high threshold the wrong-pair scores are taken to have an exponential
tail, with mean excess m fitted on the k best wrong pairs. The spacing between the two
largest values of such a tail is itself exponential with mean m, so the chance that a
wrong winner would lead the best other wrong pair by the observed gap is
p = exp(-gap / m). Pairs that share a flight with the winner are left out of the null,
since they carry part of its signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOCATED_P = 0.1
AMBIGUOUS_P = 0.5


@dataclass
class Verdict:
    verdict: str            # "located", "ambiguous", "no match", "insufficient"
    p: float                # chance a wrong winner would lead the next wrong pair by this gap
    gap: float              # winner score - best wrong pair score
    tail_width: float       # mean excess m of the k best wrong pairs over the threshold
    threshold: float
    k: int
    n_pairs: int
    n_null: int
    winner: dict

    def __str__(self):
        if self.verdict == "insufficient":
            return f"verdict: insufficient ({self.n_null} wrong pairs is too few to judge the winner against)"
        return (f"verdict: {self.verdict} (p = {self.p:.2g}): {self.winner['icao1']}/{self.winner['icao2']} scores {self.winner['score']:.2f}, "
                f"{self.gap:.2f} above the best other pair; the {self.k} best wrong pairs of {self.n_null} spread {self.tail_width:.2f} above {self.threshold:.2f}")


def best_per_pair(res: pd.DataFrame) -> pd.DataFrame:
    return res.sort_values("score", ascending=False).drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)


def verdict(res: pd.DataFrame, k_frac: float = 0.05, k_min: int = 15, k_max: int = 100,
            located_p: float = LOCATED_P, ambiguous_p: float = AMBIGUOUS_P) -> Verdict:
    """Judge the top pair of a search result (any table with icao1, icao2, score)."""
    b = best_per_pair(res)
    if not len(b):
        return Verdict("insufficient", float("nan"), float("nan"), float("nan"), float("nan"), 0, 0, 0, {})
    w = b.iloc[0]
    winner = {c: (w[c].item() if hasattr(w[c], "item") else w[c]) for c in b.columns if c in ("icao1", "icao2", "score", "lat", "lon")}
    null = np.sort(b[(b.icao1 != w.icao1) & (b.icao2 != w.icao2)].score.to_numpy())[::-1]
    k = int(np.clip(round(k_frac * len(null)), k_min, k_max))
    if len(null) < k + 5:
        return Verdict("insufficient", float("nan"), float("nan"), float("nan"), float("nan"), k, len(b), len(null), winner)
    u = float(null[k]); m = float(np.mean(null[:k] - u))
    gap = float(w.score - null[0])
    p = float(np.exp(-max(gap, 0.0) / m)) if m > 0 else (0.0 if gap > 0 else 1.0)
    v = "located" if p < located_p else "ambiguous" if p < ambiguous_p else "no match"
    return Verdict(v, p, gap, m, u, k, len(b), len(null), winner)
