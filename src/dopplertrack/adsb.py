"""Read adsb.lol / readsb `globe_history` daily archives and classify aircraft."""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

from .geo import lla_to_enu

FT = 0.3048
KT = 0.514444
TRACE_PARSE_ERRORS = (OSError, ValueError, IndexError, KeyError, TypeError, zlib.error, EOFError)   # KeyError: a trace row that is a dict
_DOC8643_URL = "https://opensky-network.org/datasets/metadata/doc8643AircraftTypes.csv"


def _cache_dir() -> Path:
    p = Path(os.environ.get("DOPPLERTRACK_CACHE", Path.home() / ".cache" / "dopplertrack"))
    p.mkdir(parents=True, exist_ok=True)
    return p


class AircraftTypes:
    """ICAO Doc 8643 designator -> engine class ('Piston', 'Turboprop', 'Jet', ...)."""

    def __init__(self, path: str | Path | None = None):
        p = Path(path) if path else _cache_dir() / "doc8643AircraftTypes.csv"
        if not p.exists():
            p.write_bytes(requests.get(_DOC8643_URL, timeout=120).content)
        self.types: dict[str, dict] = {}
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.types.setdefault(row["Designator"], row)

    def engine(self, designator: str | None) -> str | None:
        r = self.types.get(designator or "")
        return r["EngineType"] if r else None

    def is_prop(self, designator: str | None, unknown_ok: bool = True) -> bool:
        e = self.engine(designator)
        if e is None:
            return unknown_ok
        return any(k in e for k in ("Piston", "Turboprop", "Electric"))


@dataclass
class Trace:
    icao: str
    reg: str | None
    typ: str | None
    desc: str | None
    t: np.ndarray        # unix seconds
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray      # metres (geometric, i.e. GNSS height, when available, else barometric)
    gs: np.ndarray       # m/s (nan if missing)
    on_ground: np.ndarray
    track: np.ndarray | None = field(default=None, repr=False)      # true track over ground, deg (nan if missing)
    vrate: np.ndarray | None = field(default=None, repr=False)      # vertical rate, m/s (nan if missing)
    geom: np.ndarray | None = field(default=None, repr=False)       # True where alt is geometric
    alt_baro: np.ndarray | None = field(default=None, repr=False)   # pressure altitude, m (nan if missing)

    def __len__(self):
        return len(self.t)

    def _opt(self, name):
        v = getattr(self, name)
        return np.full(len(self.t), np.nan) if v is None else v

    def segments(self, max_gap_s: float = 90.0) -> list[slice]:
        """Index slices of contiguous airborne data."""
        out = []
        if len(self.t) == 0:
            return out
        brk = np.where((np.diff(self.t) > max_gap_s) | (self.on_ground[1:] != self.on_ground[:-1]))[0] + 1
        starts = np.r_[0, brk]; ends = np.r_[brk, len(self.t)]
        for s, e in zip(starts, ends):
            if e - s >= 2 and not self.on_ground[s]:
                out.append(slice(s, e))
        return out

    def resample(self, sl: slice, dt: float = 1.0, method: str = "linear"):
        """Uniformly resampled (t, lat, lon, alt) over one segment.

        method="linear" joins the reports with straight lines, so between two reports the
        velocity is constant and it jumps at every report; in a turn sampled every 5-15 s
        the modelled range rate is a staircase. method="hermite" uses the ground speed and
        true track each report carries as the velocity at that report (cubic Hermite in
        lat/lon) and a monotone cubic in altitude, so the path between reports curves the way
        the aircraft did and the velocity is continuous. Reports without speed/track fall
        back to finite-difference slopes."""
        if method not in INTERP_METHODS:
            raise ValueError(f"unknown interpolation {method!r}; use one of {INTERP_METHODS}")
        t = self.t[sl]
        tq = np.arange(t[0], t[-1], dt)
        if method == "linear":
            return (tq, np.interp(tq, t, self.lat[sl]), np.interp(tq, t, self.lon[sl]), np.interp(tq, t, self.alt[sl]))
        return (tq, *hermite_track(t, self.lat[sl], self.lon[sl], self.alt[sl], self.gs[sl], self._opt("track")[sl], tq))

    def enu(self, sl: slice, lat0: float, lon0: float, dt: float = 1.0, method: str = "linear"):
        tq, la, lo, al = self.resample(sl, dt, method)
        return tq, lla_to_enu(la, lo, al, lat0, lon0)


INTERP_METHODS = ("linear", "hermite")
_R_EARTH = 6371000.0


def hermite_track(t, lat, lon, alt, gs, track, tq):
    """(lat, lon, alt) at times tq from reports at times t, using the reported ground speed
    (m/s) and true track (deg) as the horizontal velocity at each report. Duplicate report
    times are dropped (the first wins)."""
    from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
    t = np.asarray(t, float)
    keep = np.r_[True, np.diff(t) > 0]
    t, lat, lon, alt, gs, track = (np.asarray(a, float)[keep] for a in (t, lat, lon, alt, gs, track))
    if len(t) < 3:
        return np.interp(tq, t, lat), np.interp(tq, t, lon), np.interp(tq, t, alt)
    # finite-difference fallback slopes (deg/s), used where a report lacks speed or track
    dlat_fd = np.gradient(lat, t); dlon_fd = np.gradient(lon, t)
    ok = np.isfinite(gs) & np.isfinite(track)
    tr = np.radians(np.where(ok, track, 0.0))
    ve = np.where(ok, gs, 0.0) * np.sin(tr); vn = np.where(ok, gs, 0.0) * np.cos(tr)
    k = 180.0 / np.pi / _R_EARTH
    dlat = np.where(ok, vn * k, dlat_fd)
    dlon = np.where(ok, ve * k / np.cos(np.radians(lat)), dlon_fd)
    la = CubicHermiteSpline(t, lat, dlat)(tq)
    lo = CubicHermiteSpline(t, lon, dlon)(tq)
    fa = np.isfinite(alt)
    al = PchipInterpolator(t[fa], alt[fa])(tq) if fa.sum() >= 2 else np.interp(tq, t, alt)
    return la, lo, al


def parse_trace_file(path: str | Path) -> Trace | None:
    return parse_trace_bytes(Path(path).read_bytes(), Path(path).stem.split("_")[-1])


def parse_trace_bytes(raw: bytes, fallback_icao: str = "") -> Trace | None:
    """Parse one readsb trace_full JSON document (gzipped or not)."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    d = json.loads(raw)
    if not isinstance(d, dict):
        return None
    base = float(d.get("timestamp", 0.0))
    rows = d.get("trace") or []
    t, lat, lon, alt, gs, grd, trk, vr, isgeom, baro = [], [], [], [], [], [], [], [], [], []
    num = (int, float)
    for r in rows:
        if r[1] is None or r[2] is None:
            continue
        a = r[3]
        on_ground = a == "ground"
        geom = r[10] if len(r) > 10 else None
        if isinstance(geom, num):
            am = geom * FT
        elif isinstance(a, num):
            am = a * FT
        elif on_ground:
            am = np.nan
        else:
            continue
        t.append(base + r[0]); lat.append(r[1]); lon.append(r[2]); alt.append(am)
        gs.append(r[4] * KT if isinstance(r[4], num) else np.nan); grd.append(on_ground)
        trk.append(r[5] if len(r) > 5 and isinstance(r[5], num) else np.nan)
        vr.append(r[7] * FT / 60.0 if len(r) > 7 and isinstance(r[7], num) else np.nan)
        isgeom.append(isinstance(geom, num))
        baro.append(a * FT if isinstance(a, num) else np.nan)
    if not t:
        return None
    return Trace(d.get("icao", fallback_icao), d.get("r"), d.get("t"), d.get("desc"),
                 np.array(t), np.array(lat), np.array(lon), np.array(alt), np.array(gs), np.array(grd),
                 np.array(trk, float), np.array(vr, float), np.array(isgeom, bool), np.array(baro, float))


def _scan_one(args):
    path, bbox = args
    try:
        tr = parse_trace_file(path)
    except TRACE_PARSE_ERRORS:
        return None
    if tr is None:
        return None
    lat0, lat1, lon0, lon1 = bbox
    m = (tr.lat >= lat0) & (tr.lat <= lat1) & (tr.lon >= lon0) & (tr.lon <= lon1)
    if not m.any():
        return None
    return tr


def load_region(archive_dir: str | Path, bbox: tuple[float, float, float, float],
                workers: int | None = None) -> list[Trace]:
    """All traces with at least one point inside bbox=(lat_min, lat_max, lon_min, lon_max)."""
    files = sorted(Path(archive_dir).glob("traces/*/trace_full_*.json"))
    out = []
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for tr in ex.map(_scan_one, ((str(f), bbox) for f in files), chunksize=256):
            if tr is not None:
                out.append(tr)
    return out


def in_bbox(tr: Trace, bbox) -> bool:
    lat0, lat1, lon0, lon1 = bbox
    return bool(((tr.lat >= lat0) & (tr.lat <= lat1) & (tr.lon >= lon0) & (tr.lon <= lon1)).any())


def save_traces(traces: list[Trace], path: str | Path):
    """Rows per trace: t, lat, lon, alt, gs, on_ground, track, vrate, geom, alt_baro (files
    written before the last four existed have six rows and still load)."""
    def rows(t):
        geom = np.ones(len(t.t)) if t.geom is None else t.geom.astype(float)
        baro = np.where(geom > 0, np.nan, t.alt) if t.alt_baro is None else t.alt_baro
        return np.stack([t.t, t.lat, t.lon, t.alt, t.gs, t.on_ground.astype(float),
                         t._opt("track"), t._opt("vrate"), geom, baro])
    np.savez_compressed(path, n=len(traces),
                        meta=np.array([json.dumps([t.icao, t.reg, t.typ, t.desc]) for t in traces]),
                        **{f"a{i}": rows(t) for i, t in enumerate(traces)})


def load_traces(path: str | Path) -> list[Trace]:
    z = np.load(path, allow_pickle=False)
    out = []
    for i in range(int(z["n"])):
        icao, reg, typ, desc = json.loads(str(z["meta"][i]))
        a = z[f"a{i}"]
        extra = (a[6], a[7], a[8] > 0, a[9]) if a.shape[0] >= 10 else ()
        out.append(Trace(icao, reg, typ, desc, a[0], a[1], a[2], a[3], a[4], a[5].astype(bool), *extra))
    return out


def load_many(paths) -> list[Trace]:
    """Traces from several files (e.g. consecutive UTC days), concatenated."""
    out = []
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        out.extend(load_traces(p))
    return out


def calibrate_baro(traces: list[Trace], bin_s: float = 1800.0, min_samples: int = 30,
                   max_alt_m: float = 6000.0) -> dict:
    """Put reports that only carry barometric altitude on the geometric datum, in place.

    Barometric (pressure) altitude assumes a standard atmosphere, so on a high-pressure day
    every baro-only aircraft is placed tens of metres too low (and too high on a low-pressure
    day): 76 m on the 2025-09-08 test trace, QNH 1026 hPa. Aircraft that report both
    altitudes measure the offset (geometric - baro) directly. It is fitted, per time bin of
    bin_s over the whole region, as offset = a + b * baro (the slope absorbs a non-standard
    temperature), with one robust refit after dropping 3-sigma outliers; bins with fewer
    than min_samples reports use the all-day fit. Returns a summary dict."""
    tt, bb, oo = [], [], []
    for tr in traces:
        if tr.geom is None or tr.alt_baro is None:
            continue
        m = tr.geom & np.isfinite(tr.alt_baro) & ~tr.on_ground & (tr.alt_baro < max_alt_m)
        tt.append(tr.t[m]); bb.append(tr.alt_baro[m]); oo.append(tr.alt[m] - tr.alt_baro[m])
    if not tt or sum(len(x) for x in tt) < min_samples:
        return {"samples": 0, "corrected": 0}
    tt, bb, oo = np.concatenate(tt), np.concatenate(bb), np.concatenate(oo)

    def fit(m):
        A = np.stack([np.ones(m.sum()), bb[m]], 1)
        coef = np.linalg.lstsq(A, oo[m], rcond=None)[0]
        res = oo[m] - A @ coef
        keep = np.abs(res) <= 3 * max(np.std(res), 1.0)
        idx = np.flatnonzero(m)[keep]
        if len(idx) >= 2:
            coef = np.linalg.lstsq(np.stack([np.ones(len(idx)), bb[idx]], 1), oo[idx], rcond=None)[0]
        return coef

    allday = fit(np.ones(len(tt), bool))
    bins = {}
    for k in np.unique(np.floor(tt / bin_s).astype(np.int64)):
        m = np.floor(tt / bin_s).astype(np.int64) == k
        bins[int(k)] = fit(m) if m.sum() >= min_samples else allday
    corrected = 0
    for tr in traces:
        if tr.geom is None or tr.alt_baro is None:
            continue
        m = ~tr.geom & np.isfinite(tr.alt_baro) & ~tr.on_ground
        if not m.any():
            continue
        ks = np.floor(tr.t[m] / bin_s).astype(np.int64)
        coef = np.array([bins.get(int(k), allday) for k in ks])
        tr.alt = tr.alt.copy()
        tr.alt[m] = tr.alt_baro[m] + coef[:, 0] + coef[:, 1] * tr.alt_baro[m]
        corrected += int(m.sum())
    return {"samples": int(len(tt)), "corrected": corrected, "offset_at_1000m": float(allday[0] + 1000 * allday[1]),
            "bins": len(bins)}
