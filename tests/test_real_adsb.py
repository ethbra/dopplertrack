"""Checks on real data, offline: the readsb trace of N29EF (Cirrus SR20, 2025-09-08, New Jersey)
in tests/data and a crop of NGA's EGM96 geoid (public domain). No audio is needed: these
validate the flight-data side of the model against the aircraft's own measurements."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from dopplertrack import adsb
from dopplertrack.geo import lla_to_enu
from dopplertrack.geoid import Geoid
from dopplertrack.search import observer_enu

DATA = Path(__file__).parent / "data"
TRACE = DATA / "trace_full_a2f38b.json"


@pytest.fixture(scope="module")
def geoid():
    return Geoid(DATA / "egm96_15_crop_ne_us.npz")


@pytest.fixture(scope="module")
def trace():
    return adsb.parse_trace_file(TRACE)


def test_geoid_crop_lookup(geoid):
    n = geoid.undulation([40.876466, 40.1], [-73.524951, -74.06])
    assert -34 < n[0] < -30 and -35 < n[1] < -32            # Long Island and New Jersey, EGM96
    with pytest.raises(ValueError):
        geoid.undulation([10.0], [-73.0])


def test_adsb_geometric_altitude_is_ellipsoidal_height(trace, geoid):
    """Independent physics: convert the reported pressure altitude to height above sea level
    with the reported QNH (ISA), and compare with the GNSS altitude. The difference is the
    geoid undulation, so alt_geom is height above the ellipsoid and the terrain (MSL) has to
    be moved by N before the two are compared."""
    d = json.loads(gzip.open(TRACE).read())
    qnh, diff, lat, lon = None, [], [], []
    for r in d["trace"]:
        if isinstance(r[8], dict) and r[8].get("nav_qnh"):
            qnh = r[8]["nav_qnh"]
        if qnh and isinstance(r[3], (int, float)) and len(r) > 10 and isinstance(r[10], (int, float)):
            p = 1013.25 * (1 - 2.25577e-5 * r[3] * 0.3048) ** 5.25588       # ISA pressure at that pressure altitude
            h_msl = (1 - (p / qnh) ** 0.190263) * 44330.77                   # altitude on QNH
            diff.append(r[10] * 0.3048 - h_msl); lat.append(r[1]); lon.append(r[2])
    assert len(diff) > 500
    n = float(np.median(geoid.undulation(lat, lon)))
    assert np.median(diff) == pytest.approx(n, abs=3.0)          # measured -32.2 m, EGM96 -33.6 m
    assert abs(np.median(diff)) > 25                              # and it is far from zero


def test_observers_sit_on_the_ellipsoid(geoid):
    la, lo = np.array([40.86, 40.88, 40.89]), np.array([-73.54, -73.52, -73.51])
    el = np.full(3, 20.0)
    e0 = observer_enu(la, lo, el, 40.87, -73.52)
    e1 = observer_enu(la, lo, el, 40.87, -73.52, geoid.undulation)
    assert np.allclose(e1[:, 2] - e0[:, 2], geoid.undulation(la, lo), atol=0.01)
    assert np.allclose(e1[:, :2], e0[:, :2], atol=0.01)


def test_baro_to_ellipsoid_moves_only_baro_only_reports(trace, geoid):
    x = copy.deepcopy(trace)
    m = x.geom & ~x.on_ground
    k = np.flatnonzero(m)[::3]
    x.geom = x.geom.copy(); x.geom[k] = False; x.alt = x.alt.copy(); x.alt[k] = x.alt_baro[k]
    before = x.alt.copy()
    moved = np.flatnonzero(~x.geom & np.isfinite(x.alt_baro) & ~x.on_ground)   # ours plus the trace's own baro-only reports
    assert set(k) <= set(moved) and len(moved) < len(k) + 20
    assert adsb.baro_to_ellipsoid([x], geoid.undulation) == len(moved)
    assert np.allclose(x.alt[moved] - x.alt_baro[moved], geoid.undulation(x.lat[moved], x.lon[moved]))
    untouched = np.setdiff1d(np.arange(len(x)), moved)
    assert np.array_equal(x.alt[untouched], before[untouched], equal_nan=True)


def test_hermite_predicts_held_out_reports_better_than_linear(trace):
    """Drop every other report, predict the dropped positions from the rest. Real reports
    carry noise, so the gain is far smaller than on the ideal synthetic turn, but it is there
    in straight flight and larger in turns (measured: median 6.7 -> 4.7 m overall)."""
    tr = trace
    lat0, lon0 = float(np.median(tr.lat)), float(np.median(tr.lon))
    err = {"linear": [], "hermite": []}
    for sl in tr.segments():
        idx = np.arange(sl.start, sl.stop)
        if len(idx) < 10:
            continue
        keep, held = idx[::2], idx[1::2]
        held = held[(tr.t[held] > tr.t[keep[0]]) & (tr.t[held] < tr.t[keep[-1]])]
        truth = lla_to_enu(tr.lat[held], tr.lon[held], tr.alt[held], lat0, lon0)[:, :2]
        lin = [np.interp(tr.t[held], tr.t[keep], v[keep]) for v in (tr.lat, tr.lon, tr.alt)]
        her = adsb.hermite_track(tr.t[keep], tr.lat[keep], tr.lon[keep], tr.alt[keep], tr.gs[keep], tr.track[keep], tr.t[held])
        for k, (la, lo, al) in (("linear", lin), ("hermite", her)):
            err[k].append(np.linalg.norm(lla_to_enu(la, lo, al, lat0, lon0)[:, :2] - truth, axis=1))
    lin, her = (np.concatenate(err[k]) for k in ("linear", "hermite"))
    assert len(lin) > 400
    assert np.median(her) < 0.8 * np.median(lin)
    assert np.percentile(her, 90) < np.percentile(lin, 90)


def test_baro_calibration_on_held_out_real_altitudes(trace):
    """Hide the GNSS altitude of one 10-minute block at a time, calibrate from the rest, and
    compare. Same aircraft, other times: optimistic next to calibrating from other aircraft,
    which needs a full day of traces (see tests/test_real_day.py)."""
    tr = trace
    both = tr.geom & np.isfinite(tr.alt_baro) & ~tr.on_ground
    blk = np.floor((tr.t - tr.t[0]) / 600).astype(int)
    raw, cal = [], []
    for b in np.unique(blk[both]):
        x = copy.deepcopy(tr)
        m = both & (blk == b)
        truth = x.alt[m].copy()
        x.geom = x.geom.copy(); x.geom[m] = False
        x.alt = x.alt.copy(); x.alt[m] = x.alt_baro[m]
        adsb.calibrate_baro([x], min_samples=10)
        raw.append(x.alt_baro[m] - truth); cal.append(x.alt[m] - truth)
    raw, cal = np.concatenate(raw), np.concatenate(cal)
    assert abs(np.median(raw)) > 60                     # pressure altitude is 76 m low that day
    assert abs(np.median(cal)) < 5 and np.percentile(np.abs(cal), 90) < 20
