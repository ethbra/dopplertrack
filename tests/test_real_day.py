"""A real day of ADS-B over a region: interpolation and baro calibration checked against
the aircraft's own reports, across hundreds of aircraft. No audio needed.

Opt-in: DOPPLERTRACK_DAY_TRACES = a traces file written by this version's `fetch-day` (it
keeps each report's track and both altitudes; older files lack them and are skipped), e.g.
  dopplertrack fetch-day 2026-09-08 traces_ny.npz --bbox 40.4 41.3 -74.4 -72.6
"""
import copy
import os

import numpy as np
import pytest

from dopplertrack import adsb
from dopplertrack.geo import lla_to_enu

PATH = os.environ.get("DOPPLERTRACK_DAY_TRACES")
pytestmark = pytest.mark.skipif(not PATH, reason="set DOPPLERTRACK_DAY_TRACES to run")


@pytest.fixture(scope="module")
def traces():
    tr = adsb.load_traces(PATH)
    if not any(t.track is not None for t in tr):
        pytest.skip("this traces file predates track/baro fields; fetch it again with this version")
    return tr


def test_hermite_beats_linear_on_held_out_reports(traces):
    lin_all, her_all = [], []
    for tr in traces:
        if tr.track is None:
            continue
        lat0, lon0 = float(np.median(tr.lat)), float(np.median(tr.lon))
        for sl in tr.segments():
            idx = np.arange(sl.start, sl.stop)
            if len(idx) < 10:
                continue
            keep, held = idx[::2], idx[1::2]
            held = held[(tr.t[held] > tr.t[keep[0]]) & (tr.t[held] < tr.t[keep[-1]])]
            if not len(held):
                continue
            truth = lla_to_enu(tr.lat[held], tr.lon[held], tr.alt[held], lat0, lon0)[:, :2]
            lin = [np.interp(tr.t[held], tr.t[keep], v[keep]) for v in (tr.lat, tr.lon, tr.alt)]
            her = adsb.hermite_track(tr.t[keep], tr.lat[keep], tr.lon[keep], tr.alt[keep], tr.gs[keep], tr.track[keep], tr.t[held])
            lin_all.append(np.linalg.norm(lla_to_enu(*lin, lat0, lon0)[:, :2] - truth, axis=1))
            her_all.append(np.linalg.norm(lla_to_enu(*her, lat0, lon0)[:, :2] - truth, axis=1))
    lin, her = np.concatenate(lin_all), np.concatenate(her_all)
    print(f"{len(lin)} held-out reports: linear median {np.median(lin):.1f} m p90 {np.percentile(lin, 90):.1f}; "
          f"hermite median {np.median(her):.1f} m p90 {np.percentile(her, 90):.1f}")
    assert np.median(her) < np.median(lin) and np.percentile(her, 90) < np.percentile(lin, 90)


def test_baro_calibration_from_other_aircraft(traces):
    """Hide the GNSS altitude of every fifth aircraft that reports both, calibrate from the
    others, compare with the hidden truth."""
    both = [t for t in traces if t.geom is not None and (t.geom & np.isfinite(t.alt_baro) & ~t.on_ground).sum() > 20]
    assert len(both) >= 20, "too few aircraft report both altitudes"
    work = copy.deepcopy(both)
    hidden = work[::5]
    truth = []
    for t in hidden:
        m = t.geom & np.isfinite(t.alt_baro) & ~t.on_ground
        truth.append((m, t.alt[m].copy()))
        t.geom = t.geom.copy(); t.geom[m] = False
        t.alt = t.alt.copy(); t.alt[m] = t.alt_baro[m]
    adsb.calibrate_baro(work)
    raw = np.concatenate([t.alt_baro[m] - a for t, (m, a) in zip(hidden, truth)])
    cal = np.concatenate([t.alt[m] - a for t, (m, a) in zip(hidden, truth)])
    print(f"{len(hidden)} hidden aircraft, {len(cal)} reports: raw baro error median {np.median(raw):.1f} m, "
          f"calibrated median {np.median(cal):.1f} m, |error| p90 {np.percentile(np.abs(cal), 90):.1f} m")
    # measured on 2026-09-08 over New York (174 hidden aircraft): |error| median 68.6 -> 11.4 m,
    # 90th percentile 282 -> 63 m; the tail is a few aircraft whose two altitudes disagree by
    # hundreds of metres, which no regional fit can repair
    assert np.percentile(np.abs(cal), 50) < 0.3 * np.percentile(np.abs(raw), 50)
    assert np.percentile(np.abs(cal), 90) < 0.5 * np.percentile(np.abs(raw), 90)
