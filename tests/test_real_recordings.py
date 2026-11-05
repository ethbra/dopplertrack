"""Real recordings with known answers: does each model option keep, or improve, the answer?

Opt-in, because the recordings are not in the repository. Point an environment variable at
a directory holding the recording's spectrogram and that day's traces:

  DOPPLERTRACK_OYSTER_BAY   rec.npz     `dopplertrack spectrogram` of youtu.be/4CxVxuMVkMc
                            traces_ny.npz `dopplertrack fetch-day 2026-09-08 traces_ny.npz --bbox 40.4 41.3 -74.4 -72.6`
  DOPPLERTRACK_SWFL         rec.npz     `dopplertrack spectrogram` of youtu.be/zs2o5Rhlp5w
                            traces_swfl.npz `dopplertrack fetch-day 2024-12-28 traces_swfl.npz --bbox 26.115 27.285 -82.505 -81.195`

Traces fetched with this version keep each report's track and both altitudes, which
--interp hermite and --calibrate-baro need; older files still load, but those two options
then fall back to the old behaviour.

For every option set, a 100 m grid of +-3 km around the truth is searched with the
committed event fits. The test asserts the answer is kept (true pair first, peak within
500 m), and with DOPPLERTRACK_REPORT=path writes one row per (recording, option set): peak
distance from the truth, score, margin over the best other flight pair. Those rows are what
decides whether an option becomes the default.
"""
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from dopplertrack import adsb, pipeline
from dopplertrack.geo import haversine_m
from dopplertrack.search import Grid
from dopplertrack.tracks import EventFit

RESULTS = Path(__file__).parent.parent / "docs" / "results"


def _ev(name, index=None):
    d = json.load(open(RESULTS / name))
    return EventFit(**(d[index] if index is not None else d))


RECORDINGS = {
    "oyster_bay": dict(env="DOPPLERTRACK_OYSTER_BAY", traces="traces_ny.npz", truth=(40.876466, -73.524951),
                       pair=("a1dfeb", "ace391"), events=lambda: (_ev("oyster_bay_ev1.json"), _ev("oyster_bay_ev2.json")),
                       harmonics=(2, 1)),
    # the eagle cam's blind search used detected events 4 and 12 (338 s apart): Beech Baron then PC-12
    "swfl": dict(env="DOPPLERTRACK_SWFL", traces="traces_swfl.npz", truth=(26.702359, -81.847766),
                 pair=("a2baea", "a336e4"), events=lambda: (_ev("swfl_detected.json", 4), _ev("swfl_detected.json", 12)),
                 harmonics=(1, 1)),
}
OPTIONS = {
    "baseline": dict(geoid=False, interp="linear", baro=False),     # how docs/results were made
    "geoid": dict(geoid=True, interp="linear", baro=False),
    "hermite": dict(geoid=True, interp="hermite", baro=False),
    "hermite+baro": dict(geoid=True, interp="hermite", baro=True),
}

CASES = [(r, o) for r in RECORDINGS for o in OPTIONS]


@pytest.fixture(scope="module")
def loaded():
    cache = {}

    def get(name):
        if name not in cache:
            cfg = RECORDINGS[name]
            d = Path(os.environ[cfg["env"]])
            from dopplertrack.audio import Spectrogram
            cache[name] = (Spectrogram.load(d / "rec.npz"), adsb.load_traces(d / cfg["traces"]))
        return cache[name]
    return get


@pytest.mark.parametrize("rec,opt", CASES, ids=[f"{r}-{o}" for r, o in CASES])
def test_options_keep_the_answer(rec, opt, loaded):
    cfg, o = RECORDINGS[rec], OPTIONS[opt]
    if not os.environ.get(cfg["env"]):
        pytest.skip(f"set {cfg['env']} to run")
    from dopplertrack.geoid import Geoid
    from dopplertrack.terrain import Terrain
    sp, traces = loaded(rec)
    traces = copy.deepcopy(traces)
    geoid = Geoid().undulation if o["geoid"] else None
    if o["baro"]:
        adsb.calibrate_baro(traces)
    elif geoid:
        adsb.baro_to_ellipsoid(traces, geoid)
    lat, lon = cfg["truth"]
    dlat, dlon = 3000 / 111320, 3000 / (111320 * np.cos(np.radians(lat)))
    grid = Grid.make(lat - dlat, lat + dlat, lon - dlon, lon + dlon, 100.0, Terrain().elevation, geoid)
    ev1, ev2 = cfg["events"]()
    res, _ = pipeline.run_search(sp, ev1, ev2, traces, grid, harmonics=cfg["harmonics"], interp=o["interp"], progress=False)
    assert len(res), "no candidate survived the filters"
    best = res.drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)
    top = best.iloc[0]
    true = best[(best.icao1 == cfg["pair"][0]) & (best.icao2 == cfg["pair"][1])]
    other = best[(best.icao1 != cfg["pair"][0]) | (best.icao2 != cfg["pair"][1])]
    dist = float(haversine_m(lat, lon, true.lat.iloc[0], true.lon.iloc[0])) if len(true) else float("nan")
    margin = (float(true.score.iloc[0]) / float(other.score.iloc[0]) - 1) * 100 if len(true) and len(other) else float("nan")
    row = {"recording": rec, "options": opt, "true_pair_rank": int(true.index[0]) + 1 if len(true) else -1,
           "peak_dist_m": round(dist, 1), "score": round(float(true.score.iloc[0]), 3) if len(true) else None,
           "margin_pct": round(margin, 2), "rec_start_unix": round(float(true.rec_start_unix.iloc[0]), 2) if len(true) else None}
    out = os.environ.get("DOPPLERTRACK_REPORT")
    if out:
        new = not Path(out).exists()
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    print(row)
    assert (top.icao1, top.icao2) == cfg["pair"], row
    assert dist < 500.0, row
