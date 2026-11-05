"""The winner of a search judged against the wrong pairs the same search scored."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dopplertrack.verdict import verdict

RESULTS = Path(__file__).parent.parent / "docs" / "results"


def _table(null_scores, winner=None, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(null_scores)
    df = pd.DataFrame({"icao1": [f"a{i:05x}" for i in range(n)], "icao2": [f"b{i:05x}" for i in range(n)], "score": null_scores,
                       "lat": rng.uniform(25, 49, n), "lon": rng.uniform(-125, -66, n)})
    if winner is not None:
        df = pd.concat([pd.DataFrame({"icao1": ["ffffff"], "icao2": ["eeeeee"], "score": [winner], "lat": [40.0], "lon": [-73.0]}), df])
    return df


def test_exponential_null_located_versus_lost_in_the_crowd():
    rng = np.random.default_rng(1)
    null = 10.0 + rng.exponential(0.5, 5000)                  # tail width 0.5
    far = verdict(_table(null, winner=null.max() + 3.0, rng=rng))        # six tail widths clear
    assert far.verdict == "located" and far.p < 0.01 and abs(far.tail_width - 0.5) < 0.15
    near = verdict(_table(null, winner=null.max() + 0.05, rng=rng))
    assert near.verdict == "no match" and near.p > 0.8
    mid = verdict(_table(null, winner=null.max() + 0.6, rng=rng))
    assert mid.verdict == "ambiguous"


def test_pairs_sharing_a_flight_with_the_winner_leave_the_null():
    rng = np.random.default_rng(2)
    df = _table(10.0 + rng.exponential(0.5, 2000), winner=16.0, rng=rng)
    v = verdict(df)
    sharer = pd.DataFrame({"icao1": ["ffffff"], "icao2": ["b00001"], "score": [15.9], "lat": [40.0], "lon": [-73.0]})
    v2 = verdict(pd.concat([df, sharer]))
    assert v2.n_null == v.n_null and v2.gap == pytest.approx(v.gap) and v2.verdict == "located"


def test_too_few_pairs_is_insufficient():
    v = verdict(_table(np.linspace(1, 2, 12), winner=10.0))
    assert v.verdict == "insufficient" and v.n_null == 12


def test_one_point_per_pair_counts_once():
    rng = np.random.default_rng(3)
    df = _table(10.0 + rng.exponential(0.5, 1000), winner=16.0, rng=rng)
    dup = pd.concat([df, df.assign(score=df.score - 0.1)])          # a second, worse point for every pair
    assert verdict(dup).n_null == verdict(df).n_null


@pytest.mark.parametrize("name, expect", [
    ("oyster_bay_wide_extra_best_per_pair.csv", "located"),       # nationwide, two events + the faint flyover
    ("swfl_wide_extra_best_per_pair.csv", "located"),             # nationwide, two events + three detected flyovers
    ("oyster_bay_wide_best_per_pair.csv", "ambiguous"),           # nationwide, two events: right, but not provably so
    ("oyster_bay_search_best_per_pair.csv", "ambiguous"),         # regional, two events
    ("swfl_search_best_per_pair.csv", "ambiguous"),
    ("oyster_bay_moved_box_best_per_pair.csv", "no match"),       # the audio searched in a box 150 km north of the answer
    ("howell_wye_p3_1_best_per_pair.csv", "no match"),            # railcam, a true flyover paired with a non-aircraft detection
    ("howell_wye_p1_5_extra_best_per_pair.csv", "ambiguous"),     # railcam, the true pair + nine extras on a broken clock
    # known false positives (docs/experiments.md, Howell Wye): wrong winners the tail test accepts
    pytest.param("howell_wye_p1_5_best_per_pair.csv", "ambiguous", marks=pytest.mark.xfail(strict=True, reason="wrong pair located at p = 0.08")),
    pytest.param("howell_wye_p2_4_extra_best_per_pair.csv", "no match", marks=pytest.mark.xfail(strict=True, reason="wrong pair located at p = 0.006")),
])
def test_saved_results(name, expect):
    v = verdict(pd.read_csv(RESULTS / name))
    assert v.verdict == expect, str(v)


def test_the_eagle_cam_two_event_winner_is_not_located():
    # the nationwide two-event winner there is the wrong pair (200 km off); on the saved top-1000
    # table its p sits at the ambiguous / no-match boundary, so only "not located" is stable
    v = verdict(pd.read_csv(RESULTS / "swfl_wide_best_per_pair.csv"))
    assert v.verdict != "located" and v.p > 0.4
