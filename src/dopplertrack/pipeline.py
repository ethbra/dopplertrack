"""High-level steps the CLI and notebooks share: spectrogram, day index, refine."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .adsb import AircraftTypes, baro_to_ellipsoid, calibrate_baro, load_many, load_region, save_traces
from .audio import compute_spectrogram, load_audio
from .doppler import speed_of_sound
from .geo import lla_to_enu
from .search import F0_SCALES, DayIndex, Grid, observer_enu, pick_flight
from .tracks import EventFit

WORLD = (-90.0, 90.0, -180.0, 180.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_spectrogram(audio_path, out_path, hop_s=0.25, fmin=40, fmax=450, nperseg=8192, whiten_window_s=600):
    x, sr = load_audio(audio_path)
    sp = compute_spectrogram(x, sr, nperseg=nperseg, hop_s=hop_s, fmin=fmin, fmax=fmax, whiten_window_s=whiten_window_s)
    sp.save(out_path)
    return sp


def fetch(date: str, out_path, bbox=None, archive=None, prop_only=True, pod="planes-readsb-prod-0",
          origin=None, traces_path=None, bucket_s=10.0, workers=None, from_traces=None, interp="linear",
          calibrate=False, geoid=None) -> DayIndex:
    """One UTC day of adsb.lol history -> the day index a search runs on.

    The traces are streamed from the GitHub release (or read from an extracted archive
    directory, or from_traces: trace files saved earlier), kept if they touch bbox (default:
    everywhere), stripped of known jets, and indexed: 1 Hz ENU segments in one frame at
    `origin` (default: the traces' mean position) plus a sample per flight per bucket_s.
    traces_path also saves the raw traces.

    The index's heights are ellipsoidal (ADS-B geometric altitude). Reports with only a
    barometric altitude are moved onto that datum: calibrate=True fits the offset from the
    aircraft that report both (adsb.calibrate_baro); otherwise, with a geoid, the pressure
    altitude is taken as height above sea level (adsb.baro_to_ellipsoid). interp="hermite"
    curves the paths between reports with each report's speed and track."""
    bbox = WORLD if bbox is None else tuple(bbox)
    t0 = time.time()
    if from_traces:
        from .adsb import in_bbox
        traces = [t for t in load_many(from_traces) if in_bbox(t, bbox)]
        log(f"read {len(traces)} traces touching {bbox} from {from_traces}")
    elif archive:
        log(f"scanning {archive} for traces in {bbox}")
        traces = load_region(archive, bbox, workers=workers)
    else:
        from .fetch import fetch_day
        log(f"streaming adsb.lol {date} ({pod}) for traces in {bbox}")
        def prog(n, nbytes):
            log(f"  {n} trace files scanned, {nbytes / 1e9:.2f} GB read, {nbytes / 1e6 / max(time.time() - t0, 1):.1f} MB/s")
        traces = fetch_day(date, bbox, pod=pod, progress=prog)
    log(f"{len(traces)} traces touch the region ({time.time() - t0:.0f} s)")
    if prop_only:
        types = AircraftTypes()
        traces = [t for t in traces if types.is_prop(t.typ)]
        log(f"{len(traces)} after dropping known jets")
    if calibrate:
        log(f"baro calibration: {calibrate_baro(traces)}")
    elif geoid is not None:
        n = baro_to_ellipsoid(traces, geoid)
        if n:
            log(f"{n} baro-only reports moved to the ellipsoid by the geoid (calibrate=True fits it from the day's aircraft instead)")
    if traces_path:
        save_traces(traces, traces_path)
    t0 = time.time()
    o = origin or (None, None)
    idx = DayIndex.build(traces, o[0], o[1], bucket_s=bucket_s, interp=interp)
    idx.save(out_path)
    log(f"wrote {out_path}: {len(idx)} segments, {len(idx.t)} 1 Hz points, {len(idx.sb)} samples, origin {idx.lat0:.4f}, {idx.lon0:.4f} ({time.time() - t0:.0f} s)")
    return idx


def best_per_pair(res: pd.DataFrame) -> pd.DataFrame:
    """One row per flight pair (its best point), best pair first."""
    return res.sort_values("score", ascending=False).drop_duplicates(["seg1", "seg2"]).reset_index(drop=True)


def rec_start(t_heard1: float, ev1: EventFit, shift: float) -> float:
    """UTC (unix s) of the recording's first sample implied by a candidate: the modelled
    curve is placed at audio time = unix + (ev1.t_cpa - t_heard1) + shift."""
    return float(t_heard1) - float(ev1.t_cpa) - float(shift)


def peak_spread(fine: pd.DataFrame, fracs=(0.01, 0.05)) -> dict:
    """How flat the fine-grid peak is: for each fraction f, the number of points scoring at
    least (1 - f) times the peak and the largest distance (m) of any of them from the peak.
    Not a calibrated confidence region, but a direct read of how well the events pin the
    point: a lateral low pass gives a tight plateau, two overhead passes a wide one."""
    from .geo import haversine_m
    b = fine.iloc[0]
    out = {}
    for f in fracs:
        m = fine[fine.score >= (1 - f) * b.score]
        out[f] = (int(len(m)), float(haversine_m(b.lat, b.lon, m.lat.to_numpy(), m.lon.to_numpy()).max()))
    return out


def refine(sp, ev1: EventFit, ev2: EventFit, idx: DayIndex, res_row, spacing_m=50.0, half_m=400.0, harmonics=(1, 1),
           elevation=None, c=None, extra: list[EventFit] | None = None, extra_harmonics: list[int] | None = None,
           score_mode="brightness", max_range_m=12000.0, log=log, wind=None, geoid=None) -> pd.DataFrame:
    """Fine grid around one result row (a search result's row: seg1, seg2, lat, lon, t_heard1,
    shift) for its flight pair.

    extra: further measured events beyond the two the search used. For each, the flight is
    chosen once at the row's point (pick_flight), then its modelled curve is scored at every
    fine point with the same time shift as event 1, so every event is locked to the same
    clock. A low, lateral pass pins the location far better than a high overhead one, which
    is what the extra events are for. Columns s3, s4, ... are added and summed into score;
    the chosen flights are in the DataFrame's .attrs["extra_icao"].

    c: a sound speed or a doppler.Atmosphere; wind: (east, north) m/s; geoid: puts the
    observers on the index's ellipsoidal datum (search.observer_enu)."""
    from .doppler import cpa_from_trajectory
    from .score import EventWindow, locked_score, modelled_curve
    c = speed_of_sound() if c is None else c
    extra = list(extra or []); extra_harmonics = list(extra_harmonics or [1] * len(extra))
    lat, lon = float(res_row.lat), float(res_row.lon)
    dlat = half_m / 111320.0; dlon = half_m / (111320.0 * np.cos(np.radians(lat)))
    g = Grid.make(lat - dlat, lat + dlat, lon - dlon, lon + dlon, spacing_m, elevation)
    seg1, seg2 = idx.segs[int(res_row.seg1)], idx.segs[int(res_row.seg2)]
    w1 = EventWindow.from_fit(sp, ev1, harmonics[0], mode=score_mode)
    w2 = EventWindow.from_fit(sp, ev2, harmonics[1], mode=score_mode)
    el0 = max(float(elevation(np.array([lat]), np.array([lon]))[0]), 0.0) if elevation else 0.0
    obs0 = observer_enu(np.array([lat]), np.array([lon]), np.array([el0]), idx.lat0, idx.lon0, geoid)[0]
    t0_map0 = ev1.t_cpa - float(res_row.t_heard1)
    xsegs, xwins = [], []
    for ev, h in zip(extra, extra_harmonics):
        w = EventWindow.from_fit(sp, ev, h, mode=score_mode)
        k, sc = pick_flight(idx, w, obs0, t0_map0, float(res_row["shift"]), c, max_range_m, wind=wind)
        if k < 0:
            raise ValueError(f"no flight near the candidate at the time of the extra event at {ev.t_cpa:.0f} s")
        log(f"extra event at {ev.t_cpa:.0f} s: {idx.segs[k].icao} (score {sc:.2f} at the candidate)")
        xsegs.append(idx.segs[k]); xwins.append(w)
    obs_all = observer_enu(g.lat, g.lon, g.elev, idx.lat0, idx.lon0, geoid)
    rows = []
    for i in range(len(g)):
        obs = obs_all[i]
        t_heard1 = cpa_from_trajectory(seg1.t, seg1.enu, obs, c=c, wind=wind)[1]
        tot, s1, s2, sh = locked_score(w1, w2, seg1, seg2, obs, float(t_heard1), c=c, wind=wind)
        t0_map = ev1.t_cpa - float(t_heard1)
        xs = []
        for seg, w in zip(xsegs, xwins):
            to, r = modelled_curve(seg, obs, c, wind)
            xs.append(float(w.sweep_scores(to + t0_map, r, [sh], F0_SCALES).max()))
        rows.append((g.lat[i], g.lon[i], tot + sum(xs), s1, s2, *xs, sh, rec_start(t_heard1, ev1, sh)))
    cols = ["lat", "lon", "score", "s1", "s2"] + [f"s{k + 3}" for k in range(len(extra))] + ["shift", "rec_start_unix"]
    out = pd.DataFrame(rows, columns=cols).sort_values("score", ascending=False).reset_index(drop=True)
    out.attrs["extra_icao"] = [s.icao for s in xsegs]
    return out
