"""Oyster Bay from raw audio (issue #1). Opt-in: needs the recording and that day's flight data.

Set DOPPLERTRACK_OYSTER_BAY to a directory holding
  rec.npz        `dopplertrack spectrogram` of youtu.be/4CxVxuMVkMc
  traces_ny.npz  `dopplertrack fetch 2026-09-08 ny.npz --bbox 40.4 41.3 -74.4 -72.6 --traces traces_ny.npz`
Both checks score the known location, as `search --point` does: the true pair must survive the
search filters and event 2 must land on its ridge."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from dopplertrack.adsb import load_traces
from dopplertrack.audio import Spectrogram
from dopplertrack.search import DayIndex, run_search
from dopplertrack.terrain import Terrain
from dopplertrack.tracks import detect_events, fit_event

DATA = os.environ.get("DOPPLERTRACK_OYSTER_BAY")
RESULTS = Path(__file__).parent.parent / "docs" / "results"
TRUTH = (40.876466, -73.524951)
PAIR = ("a1dfeb", "ace391")          # N22NY, then N93CQ

pytestmark = pytest.mark.skipif(not DATA, reason="set DOPPLERTRACK_OYSTER_BAY to run")


@pytest.fixture(scope="module")
def recording():
    d = Path(DATA)
    return Spectrogram.load(d / "rec.npz"), DayIndex.build(load_traces(d / "traces_ny.npz"))


def _at_truth(sp, idx, ev1, ev2):
    el = Terrain().elevation(np.array([TRUTH[0]]), np.array([TRUTH[1]]))
    return run_search(sp, ev1, ev2, idx, points=([TRUTH[0]], [TRUTH[1]], el), workers=1, log=lambda m: None)


def test_readme_fit_windows(recording):
    # the README's `fit` commands (objective "brightness", the CLI default) reproduce the
    # committed event fits, and the true pair passes the filters at the true location
    sp, idx = recording
    ev1 = fit_event(sp, 2090, 2200, 95, 200, harmonics=2, objective="brightness")
    ev2 = fit_event(sp, 3020, 3130, 95, 195, objective="brightness")
    for ev, name in ((ev1, "oyster_bay_ev1.json"), (ev2, "oyster_bay_ev2.json")):
        ref = json.load(open(RESULTS / name))
        assert abs(ev.t_cpa - ref["t_cpa"]) < 0.05 and abs(ev.f0 - ref["f0"]) < 0.05
        assert abs(ev.v - ref["v"]) < 0.1 and abs(ev.d - ref["d"]) < 5.0
    res = _at_truth(sp, idx, ev1, ev2)
    assert len(res) == 1 and (res.icao1[0], res.icao2[0]) == PAIR
    assert res.s2[0] > 10.0


@pytest.fixture(scope="module")
def detected(recording):
    sp, _ = recording
    return sorted(detect_events(sp, 60, 300, top=10)[:2], key=lambda e: e.t_cpa)


def test_detect_finds_both_flyovers(detected):
    # the README: detect's top two are the two hand-picked flyovers, CPA 2155.6 s and 3068.4 s
    assert abs(detected[0].t_cpa - 2155.6) < 5 and abs(detected[1].t_cpa - 3068.4) < 5


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="detect's fit of event 2 has f0 11 % high and the gap "
                   "between the events 10 s short, outside the search's ±6 s / ±5 % filters (issue #1)")
def test_detect_output_passes_the_filters_at_truth(recording, detected):
    sp, idx = recording
    res = _at_truth(sp, idx, *detected)
    assert len(res) == 1 and (res.icao1[0], res.icao2[0]) == PAIR
