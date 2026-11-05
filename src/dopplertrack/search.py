"""Candidate observer generation: who could have heard which flight, when, how far."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .adsb import Trace
from .doppler import is_profile, propagation_delay, speed_of_sound
from .geo import lla_to_enu, enu_to_lla


@dataclass
class Grid:
    lat: np.ndarray
    lon: np.ndarray
    elev: np.ndarray
    enu: np.ndarray        # (G,3) relative to (lat0, lon0, 0)
    lat0: float
    lon0: float

    @classmethod
    def make(cls, lat_min, lat_max, lon_min, lon_max, spacing_m, elevation_fn=None, geoid_fn=None):
        """elevation_fn gives terrain height above mean sea level (clipped at 0: water);
        geoid_fn the geoid undulation, so the observer sits at ellipsoidal height
        elev + N, the datum ADS-B geometric altitude uses. elev keeps the MSL value."""
        lat0, lon0 = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
        dlat = spacing_m / 111320.0
        dlon = spacing_m / (111320.0 * np.cos(np.radians(lat0)))
        la, lo = np.meshgrid(np.arange(lat_min, lat_max, dlat), np.arange(lon_min, lon_max, dlon), indexing="ij")
        la, lo = la.ravel(), lo.ravel()
        el = elevation_fn(la, lo) if elevation_fn else np.zeros_like(la)
        el = np.maximum(el, 0.0)
        h = el + (geoid_fn(la, lo) if geoid_fn else 0.0)
        return cls(la, lo, el, lla_to_enu(la, lo, h, lat0, lon0), lat0, lon0)

    @classmethod
    def points(cls, lat, lon, elev, lat0=None, lon0=None, geoid_fn=None):
        lat = np.atleast_1d(np.asarray(lat, float)); lon = np.atleast_1d(np.asarray(lon, float)); elev = np.atleast_1d(np.asarray(elev, float))
        lat0 = float(lat.mean()) if lat0 is None else lat0; lon0 = float(lon.mean()) if lon0 is None else lon0
        h = elev + (geoid_fn(lat, lon) if geoid_fn else 0.0)
        return cls(lat, lon, elev, lla_to_enu(lat, lon, h, lat0, lon0), lat0, lon0)

    def __len__(self):
        return len(self.lat)


@dataclass
class Segment:
    trace_idx: int
    icao: str
    t: np.ndarray       # emission times (unix s), 1 Hz
    enu: np.ndarray     # (N,3)
    speed: np.ndarray   # (N,) m/s


def build_segments(traces: list[Trace], lat0: float, lon0: float, dt: float = 1.0, min_len: int = 20,
                   clip_enu: tuple[float, float, float, float] | None = None,
                   interp: str = "linear") -> list[Segment]:
    """1 Hz ENU segments. clip_enu=(e_min, e_max, n_min, n_max) keeps only the parts of a
    track inside that box (split where it leaves), which bounds memory in cpa_table.
    interp: "linear" or "hermite" (velocity-aware, see Trace.resample)."""
    segs = []
    for i, tr in enumerate(traces):
        for sl in tr.segments():
            t, enu = tr.enu(sl, lat0, lon0, dt, method=interp)
            if len(t) < min_len or not np.all(np.isfinite(enu)):
                continue
            vel = np.gradient(enu, t, axis=0)
            spd = np.linalg.norm(vel, axis=1)
            if clip_enu is None:
                segs.append(Segment(i, tr.icao, t, enu, spd))
                continue
            inside = (enu[:, 0] >= clip_enu[0]) & (enu[:, 0] <= clip_enu[1]) & (enu[:, 1] >= clip_enu[2]) & (enu[:, 1] <= clip_enu[3])
            if not inside.any():
                continue
            edges = np.flatnonzero(np.diff(inside.astype(int))) + 1
            for a, b in zip(np.r_[0, edges], np.r_[edges, len(t)]):
                if inside[a] and b - a >= min_len:
                    segs.append(Segment(i, tr.icao, t[a:b], enu[a:b], spd[a:b]))
    return segs


def cpa_table(grid: Grid, segs: list[Segment], max_range_m: float = 10000.0, c: float | None = None,
              chunk: int = 20000, wind=None) -> pd.DataFrame:
    """For every (grid point, segment) within max_range: CPA emission time, heard time,
    slant distance and ground-track speed at CPA. wind=(east, north) m/s delays or advances
    the heard time along the ray (see doppler.propagation_delay)."""
    c = speed_of_sound() if c is None else c
    rows = []
    tree = cKDTree(grid.enu[:, :2])
    for si, s in enumerate(segs):
        # grid points within max_range (horizontally) of the track polyline, via a KD-tree
        # on track samples spaced <= max_range/4 apart (so the union of balls covers the corridor)
        step = max(1, int(max_range_m / 4 / max(np.median(s.speed), 1.0)))
        probes = s.enu[::step, :2]
        hits = tree.query_ball_point(probes, r=max_range_m * 1.05)
        m = np.unique(np.concatenate([np.asarray(h, int) for h in hits])) if len(hits) else np.array([], int)
        if len(m) == 0:
            continue
        for k in range(0, len(m), chunk):
            idx = m[k:k + chunk]
            # |g-p|^2 = |g|^2 + |p|^2 - 2 g.p as one BLAS matmul per block; coordinates are
            # centred on the chunk so the cancellation stays harmless in float64
            ctr = grid.enu[idx].mean(0)
            g = grid.enu[idx] - ctr                                   # (G,3)
            gg = (g ** 2).sum(1)
            nstep = max(1, int(1e7 // max(len(idx), 1)))
            best = np.full(len(idx), np.inf); j = np.zeros(len(idx), int)
            for n0 in range(0, len(s.t), nstep):
                pts = s.enu[n0:n0 + nstep] - ctr
                d2 = gg[:, None] + (pts ** 2).sum(1)[None, :] - 2.0 * (g @ pts.T)    # (G,n)
                jj = np.argmin(d2, axis=1)
                dd = d2[np.arange(len(idx)), jj]
                upd = dd < best
                best[upd] = dd[upd]; j[upd] = jj[upd] + n0
            r = np.sqrt(np.maximum(best, 0.0))
            ok = (r <= max_range_m) & (j > 0) & (j < len(s.t) - 1)     # interior CPA only
            if not ok.any():
                continue
            if wind is None and not is_profile(c):
                tau = r[ok] / c
            else:
                tau = propagation_delay(s.enu[j[ok]] - grid.enu[idx[ok]], c, wind, z_obs=grid.enu[idx[ok], 2])
            rows.append(pd.DataFrame({
                "gidx": idx[ok].astype(np.int32), "seg": np.int32(si),
                "t_emit": s.t[j[ok]], "t_heard": s.t[j[ok]] + tau,
                "d": r[ok].astype(np.float32), "v": s.speed[j[ok]].astype(np.float32),
            }))
    if not rows:
        return empty_cpa_table()
    return pd.concat(rows, ignore_index=True)


CPA_COLUMNS = ["gidx", "seg", "t_emit", "t_heard", "d", "v"]


def empty_cpa_table() -> pd.DataFrame:
    return pd.DataFrame({"gidx": np.zeros(0, np.int32), "seg": np.zeros(0, np.int32),
                         "t_emit": np.zeros(0), "t_heard": np.zeros(0),
                         "d": np.zeros(0, np.float32), "v": np.zeros(0, np.float32)})


def save_cpa_table(path, cpa: pd.DataFrame, fingerprint: dict):
    np.savez_compressed(path, fingerprint=np.array(repr(fingerprint)), **{k: cpa[k].to_numpy() for k in CPA_COLUMNS})


def load_cpa_table(path, fingerprint: dict) -> pd.DataFrame | None:
    """Return the cached table if it was built from the same grid and segments, else None."""
    z = np.load(path, allow_pickle=False)
    if str(z["fingerprint"]) != repr(fingerprint):
        return None
    return pd.DataFrame({k: z[k] for k in CPA_COLUMNS})


def _offset_join(a: pd.DataFrame, b: pd.DataFrame, dt_offset: float, tol_s: float) -> pd.DataFrame:
    """(grid point, a-row, b-row) where b is heard dt_offset +- tol_s after a."""
    w = 2.0 * tol_s
    a = a.assign(key=np.floor(a.t_heard.to_numpy() / w).astype(np.int64))
    kb = np.floor((b.t_heard.to_numpy() - dt_offset) / w).astype(np.int64)
    parts = [a.merge(b.assign(key=kb + off), on=["gidx", "key"], suffixes=("1", "2")) for off in (-1, 0, 1)]
    m = pd.concat(parts, ignore_index=True).drop(columns="key")
    return m[(m.seg1 != m.seg2) & ((m.t_heard2 - m.t_heard1 - dt_offset).abs() <= tol_s)].copy()


def _stage(m: pd.DataFrame) -> dict:
    return {"triples": int(len(m)), "points": int(m.gidx.nunique()) if len(m) else 0,
            "pairs": int(len(m[["seg1", "seg2"]].drop_duplicates())) if len(m) else 0}


def offset_only_stage(cpa: pd.DataFrame, dt_offset: float, tol_s: float, chunk_points: int = 20000) -> dict:
    """Funnel stage 1 as the video counts it: points that hear *any* two flights dt_offset
    apart, before speed, distance or silence. Joined in blocks of grid points so the
    intermediate table stays small even when this stage keeps most of the region."""
    g = cpa.gidx.to_numpy()
    out = {"triples": 0, "points": 0}
    pairs = set()
    for g0 in range(0, int(g.max()) + 1 if len(g) else 0, chunk_points):
        sub = cpa[(g >= g0) & (g < g0 + chunk_points)]
        if not len(sub):
            continue
        m = _offset_join(sub, sub, dt_offset, tol_s)
        out["triples"] += int(len(m)); out["points"] += int(m.gidx.nunique())
        pairs.update(zip(m.seg1.tolist(), m.seg2.tolist()))
    out["pairs"] = len(pairs)
    return out


def pair_candidates(cpa: pd.DataFrame, dt_offset: float, tol_s: float = 6.0,
                    v1=(0, 1e9), d1=(0, 1e9), v2=(0, 1e9), d2=(0, 1e9),
                    silence_d_m: float | None = 6000.0, silence_margin_s: float = 40.0,
                    funnel: dict | None = None) -> pd.DataFrame:
    """Grid points that hear an event-1-like CPA then, dt_offset later, an event-2-like CPA.

    The join is on (grid point, time bucket) rather than grid point alone: a grid point in a
    busy region hears hundreds of flights a day, and the unconstrained per-point cross product
    is billions of rows before the time filter. Buckets of width 2*tol_s with a +-1 bucket
    overlap find exactly the pairs within tol_s.

    funnel: pass a dict to have it filled with the filter funnel the video reports (points,
    triples and flight pairs after each stage): "offset" (any two flights at the offset; the
    costly one, computed only when the dict has offset=True), "speed_distance" and "silence"."""
    if funnel is not None and funnel.pop("offset", False):
        funnel["offset"] = offset_only_stage(cpa, dt_offset, tol_s)
    a = cpa[(cpa.v.between(*v1)) & (cpa.d.between(*d1))]
    b = cpa[(cpa.v.between(*v2)) & (cpa.d.between(*d2))]
    m = _offset_join(a, b, dt_offset, tol_s)
    if funnel is not None:
        funnel["speed_distance"] = _stage(m)
    if silence_d_m is not None and len(m):
        near = cpa[(cpa.d <= silence_d_m) & cpa.gidx.isin(m.gidx.unique())][["gidx", "seg", "t_heard"]]
        x = m[["gidx", "seg1", "seg2", "t_heard1", "t_heard2"]].reset_index().merge(near, on="gidx")
        bad = x[(x.seg != x.seg1) & (x.seg != x.seg2) & (x.t_heard > x.t_heard1 + silence_margin_s) & (x.t_heard < x.t_heard2 - silence_margin_s)]
        m = m.drop(index=bad["index"].unique())
    if funnel is not None:
        funnel["silence"] = _stage(m)
    return m.reset_index(drop=True)


def grid_lla(grid: Grid, gidx):
    return grid.lat[gidx], grid.lon[gidx], grid.elev[gidx]
