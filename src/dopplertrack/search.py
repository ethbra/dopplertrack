"""Candidate generation and scoring: start from the flights, not from a grid.

Two events heard dt apart at slant distances d1 and d2 mean the two aircraft were within
d1 + d2 of each other at emission times dt apart (give or take the propagation delays). That
is a cheap geometric join over a day's trajectories, and it leaves a few tens of thousands of
flight pairs, each with a small lens of ground where both could have been heard. Ground
points are laid only in those lenses and tested with exact CPA quantities, gates, a silence
filter and locked scoring. A box is a restriction on the join, not a different search; a
single event with a known time is the same lens stage on one flight; a known location is
the lens stage on one point.

Stages, each a function of the previous one's output so it can be cached on its inputs:
  1. DayIndex        a day's prop flights as 1 Hz ENU segments in ONE frame (lla_to_enu is a
                     true ECEF rotation, so slant distances are exact anywhere), plus every
                     trajectory sampled once per time bucket (10 s), sorted by bucket.
  2. pair_join       (segment 1, segment 2, bucket range) for every pair the geometry allows;
     single_join     (segment, bucket range) for one event at a known time.
  3. lens_candidates per pair, a coarse then fine lattice of ground points with exact CPA
                     values, gated, then silence-filtered.
  4. score_candidates locked_score per (point, pair) triple, in a process pool.
  5. rescore_with_extra further flyovers on the top pairs, locked to the same clock.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .adsb import Trace
from .doppler import is_profile, nominal_speed, propagation_delay, speed_of_sound
from .geo import enu_to_lla, lla_to_enu

F0_SCALES = np.arange(0.95, 1.0501, 0.005)
SHIFTS = np.arange(-8, 8.01, 0.5)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_workers() -> int:
    """Process pool size: every core on Linux; 1 on macOS, where forking a process that has
    already used BLAS threads is unsafe (pass workers explicitly to override)."""
    return 1 if sys.platform == "darwin" else (os.cpu_count() or 1)


# ----------------------------------------------------------------------------------------
# segments and the day index
# ----------------------------------------------------------------------------------------

@dataclass
class Segment:
    trace_idx: int
    icao: str
    t: np.ndarray       # emission times (unix s), 1 Hz
    enu: np.ndarray     # (N,3)
    speed: np.ndarray   # (N,) m/s


def build_segments(traces: list[Trace], lat0: float, lon0: float, dt: float = 1.0, min_len: int = 20,
                   interp: str = "linear") -> list[Segment]:
    """1 Hz ENU segments of every airborne stretch of every trace, in the frame at (lat0, lon0).
    interp: "linear" or "hermite" (uses each report's speed and track, see Trace.resample)."""
    segs = []
    for i, tr in enumerate(traces):
        for sl in tr.segments():
            t, enu = tr.enu(sl, lat0, lon0, dt, method=interp)
            if len(t) < min_len or not np.all(np.isfinite(enu)):
                continue
            segs.append(Segment(i, tr.icao, t, enu, np.linalg.norm(np.gradient(enu, t, axis=0), axis=1)))
    return segs


@dataclass
class DayIndex:
    lat0: float
    lon0: float
    bucket_s: float
    off: np.ndarray          # (S+1,) segment offsets into t/enu/spd
    t: np.ndarray            # (T,) 1 Hz emission times, unix s
    enu: np.ndarray          # (T,3)
    spd: np.ndarray          # (T,)
    icao: np.ndarray         # (S,) str
    trace_idx: np.ndarray    # (S,)
    sb: np.ndarray           # (M,) bucket id per sample, sorted
    ss: np.ndarray           # (M,) segment index per sample
    sj: np.ndarray           # (M,) sample index inside the segment
    segs: list[Segment] = field(default_factory=list, repr=False)
    spos: np.ndarray = field(default=None, repr=False)   # (M,3) sample positions
    sspd: np.ndarray = field(default=None, repr=False)   # (M,) sample speeds
    _trees: dict = field(default_factory=dict, repr=False, compare=False)
    _slla: tuple | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not self.segs:
            self.segs = [Segment(int(self.trace_idx[i]), str(self.icao[i]), self.t[a:b], self.enu[a:b], self.spd[a:b])
                         for i, (a, b) in enumerate(zip(self.off[:-1], self.off[1:]))]
        if self.spos is None:
            k = self.off[self.ss] + self.sj
            self.spos = self.enu[k]; self.sspd = self.spd[k]

    @classmethod
    def build(cls, traces: list[Trace], lat0: float | None = None, lon0: float | None = None,
              bucket_s: float = 10.0, min_len: int = 20, interp: str = "linear") -> "DayIndex":
        if lat0 is None:
            lat0 = float(np.mean([np.median(tr.lat) for tr in traces])); lon0 = float(np.mean([np.median(tr.lon) for tr in traces]))
        segs = build_segments(traces, lat0, lon0, min_len=min_len, interp=interp)
        off = np.r_[0, np.cumsum([len(s.t) for s in segs])].astype(np.int64)
        t = np.concatenate([s.t for s in segs]) if segs else np.zeros(0)
        enu = np.concatenate([s.enu for s in segs]) if segs else np.zeros((0, 3))
        spd = np.concatenate([s.speed for s in segs]) if segs else np.zeros(0)
        sb, ss, sj = [], [], []
        for i, s in enumerate(segs):
            b = np.arange(np.ceil((s.t[0] - bucket_s / 2) / bucket_s), np.floor((s.t[-1] - bucket_s / 2) / bucket_s) + 1, dtype=np.int64)
            j = np.rint((b + 0.5) * bucket_s - s.t[0]).astype(np.int64)
            ok = (j >= 0) & (j < len(s.t))
            sb.append(b[ok]); ss.append(np.full(int(ok.sum()), i, np.int32)); sj.append(j[ok].astype(np.int32))
        sb = np.concatenate(sb) if sb else np.zeros(0, np.int64)
        ss = np.concatenate(ss) if ss else np.zeros(0, np.int32)
        sj = np.concatenate(sj) if sj else np.zeros(0, np.int32)
        order = np.argsort(sb, kind="stable")
        return cls(lat0, lon0, bucket_s, off, t, enu, spd, np.array([s.icao for s in segs]), np.array([s.trace_idx for s in segs], np.int32),
                   sb[order], ss[order], sj[order], segs=segs)

    def __len__(self):
        return len(self.segs)

    def save(self, path):
        np.savez(path, lat0=self.lat0, lon0=self.lon0, bucket_s=self.bucket_s, off=self.off, t=self.t, enu=self.enu, spd=self.spd,
                 icao=self.icao, trace_idx=self.trace_idx, sb=self.sb, ss=self.ss, sj=self.sj)

    @classmethod
    def load(cls, path) -> "DayIndex":
        z = np.load(path, allow_pickle=False)
        return cls(float(z["lat0"]), float(z["lon0"]), float(z["bucket_s"]), z["off"], z["t"], z["enu"], z["spd"], z["icao"], z["trace_idx"], z["sb"], z["ss"], z["sj"])

    def fingerprint(self) -> dict:
        return {"segs": len(self.segs), "points": int(len(self.t)), "lat0": float(self.lat0), "lon0": float(self.lon0), "bucket_s": float(self.bucket_s)}

    def bucket_time(self, b) -> float:
        return (b + 0.5) * self.bucket_s

    def bucket_slice(self, b: int) -> slice:
        return slice(int(np.searchsorted(self.sb, b, "left")), int(np.searchsorted(self.sb, b, "right")))

    def tree(self, b: int) -> cKDTree | None:
        """KD-tree of every sample in bucket b (cached); None if the bucket is empty."""
        if b not in self._trees:
            sl = self.bucket_slice(b)
            self._trees[b] = cKDTree(self.spos[sl]) if sl.stop > sl.start else None
        return self._trees[b]

    def window(self, si: int, t_a: float, t_b: float) -> slice:
        """Sample slice of segment si with t in [t_a, t_b]."""
        s = self.segs[si]
        return slice(int(np.searchsorted(s.t, t_a, "left")), int(np.searchsorted(s.t, t_b, "right")))

    def sample_mask_in_bbox(self, bbox, pad_m: float = 0.0) -> np.ndarray:
        """Samples whose position lies inside (lat_min, lat_max, lon_min, lon_max), padded."""
        if self._slla is None:
            self._slla = enu_to_lla(self.spos[:, 0], self.spos[:, 1], self.spos[:, 2], self.lat0, self.lon0)[:2]
        la, lo = self._slla
        dla = pad_m / 111320.0; dlo = pad_m / (111320.0 * np.cos(np.radians((bbox[0] + bbox[1]) / 2)))
        return (la >= bbox[0] - dla) & (la <= bbox[1] + dla) & (lo >= bbox[2] - dlo) & (lo <= bbox[3] + dlo)

    def near(self, obs_enu, t_unix: float, radius_m: float) -> set[int]:
        """Segments with a sample within radius of obs_enu in the buckets around t_unix."""
        b = int(np.floor(t_unix / self.bucket_s))
        out: set[int] = set()
        for bb in (b - 1, b, b + 1):
            tr = self.tree(bb)
            if tr is None:
                continue
            sl = self.bucket_slice(bb)
            hit = tr.query_ball_point(np.asarray(obs_enu, float), radius_m)
            out.update(int(x) for x in self.ss[np.asarray(hit, int) + sl.start])
        return out


# ----------------------------------------------------------------------------------------
# the join
# ----------------------------------------------------------------------------------------

def lag_range(dt_heard: float, d1, d2, c: float, tol_s: float) -> tuple[float, float]:
    """Emission-time gap between the two CPAs allowed by the heard gap dt +- tol and the
    slant-distance gates: t_cpa2 - t_cpa1 = dt_heard + (d1 - d2) / c."""
    return dt_heard + (d1[0] - d2[1]) / c - tol_s, dt_heard + (d1[1] - d2[0]) / c + tol_s


def _empty_pairs():
    return pd.DataFrame({"seg1": np.zeros(0, np.int32), "seg2": np.zeros(0, np.int32), "b_min": np.zeros(0, np.int64), "b_max": np.zeros(0, np.int64), "n": np.zeros(0, np.int64)})


def pair_join(idx: DayIndex, dt_heard: float, v1, d1, v2, d2, tol_s: float = 6.0, c: float | None = None,
              v_slack: float = 0.1, a_mask: np.ndarray | None = None, progress=None) -> pd.DataFrame:
    """Flight pairs that could be heard as event 1 then event 2 by one observer: in some
    bucket k flight 1 (speed in v1) is within d1max + d2max of where flight 2 (speed in v2) is
    at the allowed lag. a_mask restricts the flight-1 samples (a bounding box). Returns one
    row per pair: seg1, seg2, b_min, b_max (buckets of the flight-1 side), n (matched
    buckets). Nothing here is exact; it only has to be complete."""
    c = speed_of_sound() if c is None else c
    W = idx.bucket_s
    lo, hi = lag_range(dt_heard, d1, d2, c, tol_s)
    v1g = (v1[0] * (1 - v_slack), v1[1] * (1 + v_slack)); v2g = (v2[0] * (1 - v_slack), v2[1] * (1 + v_slack))
    R = d1[1] + d2[1] + (v1g[1] + v2g[1]) * W / 2
    A = (idx.sspd >= v1g[0]) & (idx.sspd <= v1g[1])
    if a_mask is not None:
        A &= a_mask
    B = (idx.sspd >= v2g[0]) & (idx.sspd <= v2g[1])
    buckets = np.unique(idx.sb[A])
    btrees: dict[int, tuple] = {}
    out1, out2, outb = [], [], []
    t0 = time.time()
    for n, k in enumerate(buckets):
        sl = idx.bucket_slice(int(k)); a_idx = np.flatnonzero(A[sl]) + sl.start
        ta = cKDTree(idx.spos[a_idx])
        tk = idx.bucket_time(k)
        kb_lo = int(np.floor((tk - W / 2 + lo) / W)); kb_hi = int(np.floor((tk + W / 2 + hi) / W))
        for kb in list(btrees):
            if kb < kb_lo:
                del btrees[kb]
        for kb in range(kb_lo, kb_hi + 1):
            if kb not in btrees:
                slb = idx.bucket_slice(kb); b_idx = np.flatnonzero(B[slb]) + slb.start
                btrees[kb] = (b_idx, cKDTree(idx.spos[b_idx])) if len(b_idx) else (b_idx, None)
            b_idx, tb = btrees[kb]
            if tb is None:
                continue
            m = ta.sparse_distance_matrix(tb, R, output_type="coo_matrix")
            if m.nnz == 0:
                continue
            s1 = idx.ss[a_idx[m.row]]; s2 = idx.ss[b_idx[m.col]]
            keep = s1 != s2
            out1.append(s1[keep]); out2.append(s2[keep]); outb.append(np.full(int(keep.sum()), k, np.int64))
        if progress and (n + 1) % 1000 == 0:
            progress(n + 1, len(buckets), time.time() - t0)
    if not out1:
        return _empty_pairs()
    df = pd.DataFrame({"seg1": np.concatenate(out1), "seg2": np.concatenate(out2), "b": np.concatenate(outb)}).drop_duplicates()
    g = df.groupby(["seg1", "seg2"], sort=False)["b"]
    return g.agg(b_min="min", b_max="max", n="size").reset_index()


def single_join(idx: DayIndex, t_heard: float, v1, d1, tol_s: float = 30.0, c: float | None = None,
                v_slack: float = 0.1, a_mask: np.ndarray | None = None) -> pd.DataFrame:
    """One event heard at a known unix time (+- tol_s): every flight with a speed-gated sample
    in the buckets its CPA could fall in, as pair rows with seg2 = -1."""
    c = speed_of_sound() if c is None else c
    W = idx.bucket_s
    A = (idx.sspd >= v1[0] * (1 - v_slack)) & (idx.sspd <= v1[1] * (1 + v_slack))
    if a_mask is not None:
        A &= a_mask
    b_lo = int(np.floor((t_heard - d1[1] / c - tol_s) / W)); b_hi = int(np.floor((t_heard + tol_s) / W))
    A &= (idx.sb >= b_lo) & (idx.sb <= b_hi)
    if not A.any():
        return _empty_pairs()
    df = pd.DataFrame({"seg1": idx.ss[A], "b": idx.sb[A]})
    g = df.groupby("seg1", sort=False)["b"].agg(b_min="min", b_max="max", n="size").reset_index()
    g.insert(1, "seg2", np.int32(-1))
    return g


# ----------------------------------------------------------------------------------------
# lens lattices, exact gates, silence filter
# ----------------------------------------------------------------------------------------

def observer_enu(lat, lon, elev_msl, lat0, lon0, geoid=None) -> np.ndarray:
    """Observers in the index frame. Terrain is height above mean sea level; ADS-B geometric
    altitude (and so the index) is height above the WGS-84 ellipsoid; geoid (an undulation
    function, see geoid.Geoid) puts the observer on the same datum. Without it the observer
    sits N metres off in height (about +33 m around New York)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float); h = np.asarray(elev_msl, float)
    if geoid is not None:
        h = h + geoid(lat, lon)
    return lla_to_enu(lat, lon, h, lat0, lon0)


def cpa_to_points(P: np.ndarray, X: np.ndarray, T: np.ndarray, V: np.ndarray, c, wind=None):
    """Exact CPA of one trajectory window (X (N,3), T, V) to many points P (G,3):
    (d, t_heard, v, interior); interior is False where the closest sample is the window's first
    or last, i.e. the true CPA lies outside the window. A KD-tree nearest-sample query rather
    than a dense G x N distance matrix: same answer, no multi-megabyte temporaries, so a
    process per core scales instead of fighting over memory bandwidth.

    c is a sound speed or a doppler.Atmosphere (each ray at the mean speed of the air it
    crosses); wind (east, north) m/s moves the heard time (doppler.propagation_delay)."""
    d, j = cKDTree(X).query(P, workers=1)
    if wind is None and not is_profile(c):
        th = T[j] + d / c
    else:
        th = T[j] + propagation_delay(X[j] - P, c, wind, z_obs=P[:, 2])
    return d, th, V[j], (j > 0) & (j < len(T) - 1)


@dataclass
class LensParams:
    dt_heard: float                        # heard gap event 2 - event 1 (two events) or the event's heard unix time (single)
    v1: tuple; d1: tuple
    v2: tuple = (0.0, 0.0); d2: tuple = (0.0, 0.0)
    tol_s: float = 6.0
    c: float = 340.0                       # a sound speed, or a doppler.Atmosphere
    wind: tuple | None = None              # (east, north) m/s
    spacing_m: float = 200.0
    coarse_m: float = 1000.0
    max_range_m: float = 12000.0
    silence_d_m: float | None = 6000.0
    silence_margin_s: float = 40.0
    pad_s: float = 30.0
    single: bool = False                   # one event at a known time: dt_heard is that time
    points: tuple | None = None            # (lat, lon, elev, enu (G,3)) to test instead of a lattice

    def fingerprint(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            if k == "wind" and v is None:        # absent, so caches made before the field existed stay valid
                continue
            if k == "points":
                out[k] = None if v is None else [float(x) for x in np.asarray(v[0]).ravel()[:50]]
            else:
                out[k] = tuple(float(x) for x in v) if isinstance(v, (tuple, list)) else v
        return out


_G: dict = {}      # per-process state for the pools (set before forking)


def _ground_lattice(e, n, lat_c, lon_c, elevation, lat0, lon0, geoid=None):
    """Local tangent-plane offsets (e, n) at (lat_c, lon_c) -> lat, lon, elev (MSL), global ENU
    (on the ellipsoid when geoid is given, see observer_enu)."""
    lat, lon, _ = enu_to_lla(e, n, np.zeros_like(e), lat_c, lon_c)
    el = np.maximum(elevation(lat, lon), 0.0) if elevation is not None else np.zeros_like(lat)
    return lat, lon, el, observer_enu(lat, lon, el, lat0, lon0, geoid)


def _gates(p: LensParams, da, ha, va, ia, db, hb, vb, ib, d1hi, d2hi, slack_d=0.0, slack_t=0.0, v_slack=0.0):
    ok = ia & (da >= p.d1[0] - slack_d) & (da <= d1hi + slack_d) & (va >= p.v1[0] * (1 - v_slack)) & (va <= p.v1[1] * (1 + v_slack))
    if p.single:
        return ok & (np.abs(ha - p.dt_heard) <= p.tol_s + slack_t)
    return (ok & ib & (db >= p.d2[0] - slack_d) & (db <= d2hi + slack_d) & (vb >= p.v2[0] * (1 - v_slack)) & (vb <= p.v2[1] * (1 + v_slack))
            & (np.abs(hb - ha - p.dt_heard) <= p.tol_s + slack_t))


def _lens_one(row) -> pd.DataFrame | None:
    idx: DayIndex = _G["idx"]; p: LensParams = _G["params"]; elevation = _G.get("elevation"); geoid = _G.get("geoid")
    seg1, seg2, b_min, b_max = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    s1 = idx.segs[seg1]
    W, c, cw = idx.bucket_s, nominal_speed(p.c), p.wind      # c: the scalar the gate arithmetic needs
    ca = p.c                                                 # the full model, for exact heard times
    t1a, t1b = b_min * W - p.pad_s, (b_max + 1) * W + p.pad_s
    w1 = idx.window(seg1, t1a, t1b)
    if w1.stop - w1.start < 3:
        return None, 0, 0
    X1, T1, V1 = s1.enu[w1], s1.t[w1], s1.speed[w1]
    d1hi = min(p.d1[1], p.max_range_m); d2hi = min(p.d2[1], p.max_range_m)
    if p.single:
        X2 = T2 = V2 = None
    else:
        lo, hi = lag_range(p.dt_heard, p.d1, p.d2, c, p.tol_s)
        s2 = idx.segs[seg2]
        w2 = idx.window(seg2, t1a + lo - p.pad_s, t1b + hi + p.pad_s)
        if w2.stop - w2.start < 3:
            return None, 0, 0
        X2, T2, V2 = s2.enu[w2], s2.t[w2], s2.speed[w2]

    def cpa_both(G):
        da, ha, va, ia = cpa_to_points(G, X1, T1, V1, ca, cw)
        if p.single:
            z = np.zeros(len(G))
            return da, ha, va, ia, z + np.nan, z, z + np.nan, np.ones(len(G), bool)
        return (da, ha, va, ia) + cpa_to_points(G, X2, T2, V2, ca, cw)

    if p.points is not None:
        lat, lon, el, G = (np.asarray(x, float) for x in p.points)
    else:
        # coarse lattice in the tangent plane at the flight-1 window's centre, within d1hi of its path
        lat_c, lon_c, _ = enu_to_lla(*X1.mean(0), idx.lat0, idx.lon0)
        lat1, lon1, _ = enu_to_lla(X1[:, 0], X1[:, 1], X1[:, 2], idx.lat0, idx.lon0)
        P1 = lla_to_enu(lat1, lon1, np.zeros_like(lat1), lat_c, lon_c)[:, :2]
        cm = p.coarse_m
        e = np.arange(np.floor((P1[:, 0].min() - d1hi) / cm), np.ceil((P1[:, 0].max() + d1hi) / cm) + 1) * cm
        n = np.arange(np.floor((P1[:, 1].min() - d1hi) / cm), np.ceil((P1[:, 1].max() + d1hi) / cm) + 1) * cm
        E, N = np.meshgrid(e, n, indexing="ij"); E, N = E.ravel(), N.ravel()
        near = cKDTree(P1).query(np.stack([E, N], 1), distance_upper_bound=d1hi + cm)[0] <= d1hi + cm
        E, N = E[near], N[near]
        if len(E) == 0:
            return None, 0, 0
        _, _, _, G = _ground_lattice(E, N, lat_c, lon_c, None, idx.lat0, idx.lon0, geoid)
        sd = 0.75 * cm; st = sd * (1.0 / max(p.v1[0] if p.single else min(p.v1[0], p.v2[0]), 10.0) + 1.0 / c)
        ok = _gates(p, *cpa_both(G), d1hi, d2hi, slack_d=sd, slack_t=2 * st, v_slack=0.1)
        if not ok.any():
            return None, 0, 0
        # fine lattice around each surviving coarse cell
        m = int(np.ceil(cm / 2 / p.spacing_m))
        offs = np.arange(-m, m + 1) * p.spacing_m
        OE, ON = np.meshgrid(offs, offs, indexing="ij")
        Ef = (E[ok][:, None] + OE.ravel()[None, :]).ravel(); Nf = (N[ok][:, None] + ON.ravel()[None, :]).ravel()
        key = np.unique(np.rint(np.stack([Ef, Nf], 1) / p.spacing_m).astype(np.int64), axis=0)
        Ef, Nf = key[:, 0] * p.spacing_m, key[:, 1] * p.spacing_m
        lat, lon, el, G = _ground_lattice(Ef, Nf, lat_c, lon_c, elevation, idx.lat0, idx.lon0, geoid)
    da, ha, va, ia, db, hb, vb, ib = cpa_both(G)
    ok = _gates(p, da, ha, va, ia, db, hb, vb, ib, d1hi, d2hi)
    if not ok.any():
        return None, 0, 0
    n_gated = int(ok.sum())
    F = G[ok]; ha, da, va, hb, db, vb = ha[ok], da[ok], va[ok], hb[ok], db[ok], vb[ok]
    lat, lon, el = lat[ok], lon[ok], el[ok]
    # silence: no other flight may pass within silence_d_m with its CPA heard between the events
    if p.silence_d_m is not None and not p.single:
        mg = p.silence_margin_s
        ta, tb = float(ha.min()) + mg, float(hb.max()) - mg
        if tb > ta:
            ctr = F.mean(0); rad = float(np.linalg.norm(F - ctr, axis=1).max())
            others: set[int] = set()
            for b in range(int(np.floor(ta / W)), int(np.floor(tb / W)) + 1):
                tr = idx.tree(b)
                if tr is None:
                    continue
                sl = idx.bucket_slice(b)
                hit = tr.query_ball_point(ctr, rad + p.silence_d_m + 150.0 * W / 2)
                others.update(int(x) for x in idx.ss[np.asarray(hit, int) + sl.start])
            others.discard(seg1); others.discard(seg2)
            bad = np.zeros(len(F), bool)
            for so in others:
                w = idx.window(so, ta - 60.0, tb + 60.0)
                if w.stop - w.start < 3:
                    continue
                s = idx.segs[so]
                do, ho, _, io = cpa_to_points(F, s.enu[w], s.t[w], s.speed[w], ca, cw)
                bad |= io & (do <= p.silence_d_m) & (ho > ha + mg) & (ho < hb - mg)
            if bad.all():
                return None, n_gated, 0
            keep = ~bad
            F, ha, da, va, hb, db, vb, lat, lon, el = (x[keep] for x in (F, ha, da, va, hb, db, vb, lat, lon, el))
    return pd.DataFrame({"lat": lat, "lon": lon, "elev": el, "e": F[:, 0], "n": F[:, 1], "u": F[:, 2],
                         "seg1": np.int32(seg1), "seg2": np.int32(seg2), "t_heard1": ha, "d1": da.astype(np.float32), "v1": va.astype(np.float32),
                         "t_heard2": hb, "d2": db.astype(np.float32), "v2": vb.astype(np.float32)}), n_gated, len(F)


CAND_COLUMNS = ["lat", "lon", "elev", "e", "n", "u", "seg1", "seg2", "t_heard1", "d1", "v1", "t_heard2", "d2", "v2"]


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame({k: np.zeros(0, np.int32 if k in ("seg1", "seg2") else float) for k in CAND_COLUMNS})


def _pool_map(fn, items, workers: int, chunksize: int = 8):
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    from multiprocessing import get_context
    with get_context("fork").Pool(workers) as pool:
        yield from pool.imap_unordered(fn, items, chunksize=chunksize)


def lens_candidates(idx: DayIndex, pairs: pd.DataFrame, params: LensParams, elevation=None,
                    workers: int | None = None, progress=None, geoid=None, funnel: dict | None = None) -> pd.DataFrame:
    """Exact (point, flight pair) candidates for every pair from pair_join / single_join: a
    200 m lattice (params.spacing_m) wherever a 1 km (params.coarse_m) sweep passed loosened
    gates, with the gates and silence filter applied exactly; or just params.points.

    funnel: a dict to fill with the counts the video reports after each filter: flight pairs
    from the join (offset and geometry), then (points, pairs) after the exact speed, distance
    and offset gates, then after the silence filter."""
    workers = default_workers() if workers is None else workers
    _G.update(idx=idx, params=params, elevation=elevation, geoid=geoid)
    rows = pairs[["seg1", "seg2", "b_min", "b_max"]].to_numpy()
    out = []
    n_gate = n_sil = pairs_gate = 0
    t0 = time.time()
    for k, (df, ng, ns) in enumerate(_pool_map(_lens_one, list(rows), workers)):
        n_gate += ng; n_sil += ns; pairs_gate += ng > 0
        if df is not None:
            out.append(df)
        if progress and (k + 1) % 1000 == 0:
            progress(k + 1, len(rows), time.time() - t0)
    if funnel is not None:
        funnel.update(join={"pairs": int(len(rows))}, gates={"points": n_gate, "pairs": int(pairs_gate)},
                      silence={"points": n_sil, "pairs": int(len(out))})
    return pd.concat(out, ignore_index=True) if out else empty_candidates()


def save_candidates(path, cands: pd.DataFrame, fingerprint: dict):
    np.savez_compressed(path, fingerprint=np.array(repr(fingerprint)), **{k: cands[k].to_numpy() for k in CAND_COLUMNS})


def load_candidates(path, fingerprint: dict) -> pd.DataFrame | None:
    z = np.load(path, allow_pickle=False)
    if str(z["fingerprint"]) != repr(fingerprint):
        return None
    return pd.DataFrame({k: z[k] for k in CAND_COLUMNS})


# ----------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------

RESULT_COLUMNS = ["gidx", "lat", "lon", "elev", "seg1", "seg2", "icao1", "icao2", "t_heard1", "d1", "v1", "d2", "v2", "score", "s1", "s2", "shift",
                  "rec_start_unix", "lat0", "lon0"]


def _score_chunk(args):
    from .score import locked_score, modelled_curve
    idx: DayIndex = _G["idx"]; w1 = _G["w1"]; w2 = _G["w2"]; c = _G["c"]; wind = _G.get("wind")
    i0, seg1, seg2, obs, th1 = args
    out = np.empty((len(seg1), 4))
    for k in range(len(seg1)):
        if seg2[k] < 0:
            to, r = modelled_curve(idx.segs[int(seg1[k])], obs[k], c, wind)
            sc = w1.sweep_scores(to + (w1.fit.t_cpa - float(th1[k])), r, SHIFTS, F0_SCALES)
            i, j = np.unravel_index(np.argmax(sc), sc.shape)
            out[k] = (sc[i, j], sc[i, j], 0.0, SHIFTS[i])
        else:
            out[k] = locked_score(w1, w2, idx.segs[int(seg1[k])], idx.segs[int(seg2[k])], obs[k], float(th1[k]), c=c, wind=wind)
    return i0, out


def score_candidates(sp, ev1, ev2, idx: DayIndex, cands: pd.DataFrame, harmonics=(1, 1), score_mode="brightness",
                     c: float | None = None, workers: int | None = None, chunk: int = 200, progress=None, wind=None) -> pd.DataFrame:
    """locked_score for every candidate row (or the single-event sweep where seg2 < 0);
    returns the result table (RESULT_COLUMNS), best first. rec_start_unix is the UTC time of
    the recording's first sample each row implies: the modelled curve is placed at audio
    time = unix + (ev1.t_cpa - t_heard1) + shift."""
    from .score import EventWindow
    c = speed_of_sound() if c is None else c
    workers = default_workers() if workers is None else workers
    w1 = EventWindow.from_fit(sp, ev1, harmonics[0], mode=score_mode)
    w2 = EventWindow.from_fit(sp, ev2, harmonics[1], mode=score_mode) if ev2 is not None else None
    _G.update(idx=idx, w1=w1, w2=w2, c=c, wind=wind)
    obs = cands[["e", "n", "u"]].to_numpy(); s1 = cands.seg1.to_numpy(); s2 = cands.seg2.to_numpy(); th = cands.t_heard1.to_numpy()
    jobs = [(a, s1[a:a + chunk], s2[a:a + chunk], obs[a:a + chunk], th[a:a + chunk]) for a in range(0, len(cands), chunk)]
    sc = np.zeros((len(cands), 4))
    t0 = time.time(); done = 0
    for i0, out in _pool_map(_score_chunk, jobs, workers, chunksize=1):
        sc[i0:i0 + len(out)] = out; done += len(out)
        if progress:
            progress(done, len(cands), time.time() - t0)
    res = pd.DataFrame({
        "gidx": np.arange(len(cands)), "lat": cands.lat.to_numpy(), "lon": cands.lon.to_numpy(), "elev": cands.elev.to_numpy(),
        "seg1": s1, "seg2": s2, "icao1": idx.icao[s1], "icao2": np.where(s2 >= 0, idx.icao[np.maximum(s2, 0)], ""),
        "t_heard1": th, "d1": cands.d1.to_numpy(), "v1": cands.v1.to_numpy(), "d2": cands.d2.to_numpy(), "v2": cands.v2.to_numpy(),
        "score": sc[:, 0], "s1": sc[:, 1], "s2": sc[:, 2], "shift": sc[:, 3], "rec_start_unix": th - float(ev1.t_cpa) - sc[:, 3]})
    res["lat0"] = idx.lat0; res["lon0"] = idx.lon0
    return res.sort_values("score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# the whole thing
# ----------------------------------------------------------------------------------------

def run_search(sp, ev1, ev2, idx: DayIndex, harmonics=(1, 1), tol_s=None, v_tol=0.35, d_tol=(0.4, 2.5),
               silence_d_m=6000.0, max_range_m=12000.0, spacing_m=200.0, coarse_m=1000.0, elevation=None,
               c=None, score_mode="brightness", workers=None, cand_cache=None, bbox=None, points=None,
               t_heard_unix=None, log=_log, wind=None, geoid=None, funnel: dict | None = None) -> pd.DataFrame:
    """Join -> lens candidates -> scores for one day index.

    ev2=None with t_heard_unix: one event at a known time (tol_s defaults to 30 s).
    bbox: restrict the join to flights in (lat_min, lat_max, lon_min, lon_max).
    points: (lat, lon, elev) arrays to test instead of lattices (a known location).
    cand_cache: optional .npz path; the candidate table depends on the index, the events'
    gates and the geometry parameters but not on the spectrogram or scorer, so it is reused
    when those match.
    c: a sound speed or a doppler.Atmosphere; wind: (east, north) m/s or None; geoid: an
    undulation function putting observers on the index's ellipsoidal datum (observer_enu);
    funnel: a dict filled with the counts after each filter (lens_candidates)."""
    c = speed_of_sound() if c is None else c
    cs = nominal_speed(c)
    single = ev2 is None
    if single and t_heard_unix is None:
        raise ValueError("one event needs its heard time (t_heard_unix)")
    tol_s = (30.0 if single else 6.0) if tol_s is None else tol_s
    g = lambda ev: ((ev.v * (1 - v_tol), ev.v * (1 + v_tol)), (ev.d * d_tol[0], min(ev.d * d_tol[1], max_range_m)))
    v1, d1 = g(ev1)
    v2, d2 = g(ev2) if not single else ((0.0, 0.0), (0.0, 0.0))
    pts = None
    if points is not None:
        la, lo, el = (np.atleast_1d(np.asarray(x, float)) for x in points)
        pts = (la, lo, el, observer_enu(la, lo, el, idx.lat0, idx.lon0, geoid))
    params = LensParams(dt_heard=float(t_heard_unix if single else ev2.t_cpa - ev1.t_cpa), v1=v1, d1=d1, v2=v2, d2=d2,
                        tol_s=tol_s, c=c if is_profile(c) else float(c), wind=None if wind is None else tuple(float(x) for x in wind),
                        spacing_m=spacing_m, coarse_m=coarse_m, max_range_m=max_range_m,
                        silence_d_m=silence_d_m, single=single, points=pts)
    fp = {"index": idx.fingerprint(), "params": params.fingerprint(), "dem": elevation is not None, "bbox": None if bbox is None else tuple(float(x) for x in bbox)}
    if geoid is not None:
        fp["geoid"] = True
    cands = None
    if cand_cache and os.path.exists(cand_cache):
        cands = load_candidates(cand_cache, fp)
        log(f"loaded {len(cands)} candidates from {cand_cache}" if cands is not None else f"{cand_cache} is for other inputs; recomputing")
    if cands is None:
        t0 = time.time()
        a_mask = idx.sample_mask_in_bbox(bbox, pad_m=d1[1]) if bbox is not None else None
        if pts is not None:
            a_mask = (a_mask if a_mask is not None else np.ones(len(idx.sb), bool)) & (cKDTree(pts[3]).query(idx.spos, distance_upper_bound=d1[1] + 1500.0)[0] <= d1[1] + 1500.0)
        if single:
            pairs = single_join(idx, params.dt_heard, v1, d1, tol_s=tol_s, c=cs, a_mask=a_mask)
            log(f"{len(pairs)} flights near the event time pass the speed gate ({time.time() - t0:.0f} s); gates v {v1[0]:.0f}-{v1[1]:.0f} m/s, d {d1[0]:.0f}-{d1[1]:.0f} m")
        else:
            pairs = pair_join(idx, params.dt_heard, v1, d1, v2, d2, tol_s=tol_s, c=cs, a_mask=a_mask,
                              progress=lambda k, n, s: log(f"  join: bucket {k}/{n}, {s:.0f} s"))
            log(f"{len(pairs)} flight pairs pass the join ({time.time() - t0:.0f} s); gates v1 {v1[0]:.0f}-{v1[1]:.0f} m/s, d1 {d1[0]:.0f}-{d1[1]:.0f} m, v2 {v2[0]:.0f}-{v2[1]:.0f} m/s, d2 {d2[0]:.0f}-{d2[1]:.0f} m")
        t0 = time.time()
        cands = lens_candidates(idx, pairs, params, elevation=elevation, workers=workers, geoid=geoid, funnel=funnel,
                                progress=lambda k, n, s: log(f"  lens: pair {k}/{n}, {s:.0f} s, {s / k * 1000:.1f} ms each"))
        if funnel:
            log(f"funnel: {funnel['join']['pairs']} pairs from the join -> {funnel['gates']['points']} points on {funnel['gates']['pairs']} pairs "
                f"after the exact gates -> {funnel['silence']['points']} points on {funnel['silence']['pairs']} pairs after silence")
        log(f"{len(cands)} candidate (point, flight) rows on {cands.groupby(['seg1', 'seg2']).ngroups if len(cands) else 0} flight pairs ({time.time() - t0:.0f} s)")
        if cand_cache:
            save_candidates(cand_cache, cands, fp)
    if not len(cands):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    t0 = time.time()
    res = score_candidates(sp, ev1, ev2, idx, cands, harmonics=harmonics, score_mode=score_mode, c=c, workers=workers, wind=wind,
                           progress=lambda k, n, s: log(f"  scored {k}/{n} ({s / k * 1000:.1f} ms each)") if k % 5000 < 200 or k == n else None)
    log(f"scored {len(res)} rows in {time.time() - t0:.0f} s")
    return res


# ----------------------------------------------------------------------------------------
# more events: rescore the top pairs with further flyovers locked to the same clock
# ----------------------------------------------------------------------------------------

def pick_flight(idx: DayIndex, w, obs_enu, t0_map: float, shift: float, c,
                max_range_m: float = 12000.0, exclude=(), wind=None) -> tuple[int, float]:
    """Which flight does an observer at obs_enu hear as the event in window w, given the
    recording's clock (audio = unix + t0_map) and the time shift chosen for event 1? The
    flights within max_range at that time come from the index's bucket trees; the best
    locked score wins. Returns (segment index, score), (-1, 0) if nothing is near."""
    from .score import modelled_curve
    t_unix = w.fit.t_cpa - t0_map
    best, best_s = -1, 0.0
    for si in idx.near(obs_enu, t_unix, max_range_m + 150.0 * idx.bucket_s) - set(exclude):
        to, r = modelled_curve(idx.segs[si], obs_enu, c, wind)
        sc = float(w.sweep_scores(to + t0_map, r, [shift], F0_SCALES).max())
        if sc > best_s:
            best, best_s = si, sc
    return best, best_s


def _rescore_pair(args):
    from .score import modelled_curve
    idx: DayIndex = _G["idx"]; t1 = _G["ev1_t_cpa"]; xw = _G["xw"]; c = _G["c"]; mr = _G["max_range_m"]; wind = _G.get("wind")
    i0, obs, th1, shift, seg1, seg2 = args
    k = int(np.argmax(_G["score"][i0:i0 + len(obs)]))                      # the pair's best point chooses the flights
    t0_map = t1 - float(th1[k])
    chosen = [pick_flight(idx, w, obs[k], t0_map, float(shift[k]), c, mr, exclude=(seg1, seg2), wind=wind)[0] for w in xw]
    out = np.zeros((len(obs), len(xw)))
    for j, (w, si) in enumerate(zip(xw, chosen)):
        if si < 0:
            continue
        seg = idx.segs[si]
        for n in range(len(obs)):
            to, r = modelled_curve(seg, obs[n], c, wind)
            out[n, j] = w.sweep_scores(to + (t1 - float(th1[n])), r, [float(shift[n])], F0_SCALES).max()
    return i0, out, chosen


def rescore_with_extra(sp, ev1, extra, idx: DayIndex, res: pd.DataFrame, extra_harmonics=None, top_pairs: int = 200,
                       score_mode="brightness", c=None, max_range_m: float = 12000.0,
                       workers: int | None = None, log=_log, wind=None, geoid=None) -> pd.DataFrame:
    """Add further measured events to the top pairs of a search result. For each of the best
    `top_pairs` flight pairs, each extra event is matched to a flight once, at the pair's best
    point (pick_flight), then scored at every point of that pair with that point's own time
    shift, so every event is locked to one clock. Returns those pairs' rows re-ranked by
    score = s1 + s2 + s3 + ..., with the two-event score kept as score12 and the chosen
    flights as x_icao1, x_icao2, ..."""
    from .score import EventWindow
    c = speed_of_sound() if c is None else c
    workers = default_workers() if workers is None else workers
    extra = list(extra); extra_harmonics = list(extra_harmonics or [1] * len(extra))
    xw = [EventWindow.from_fit(sp, ev, h, mode=score_mode) for ev, h in zip(extra, extra_harmonics)]
    best = res.sort_values("score", ascending=False).drop_duplicates(["seg1", "seg2"]).head(top_pairs)
    keys = set(zip(best.seg1.tolist(), best.seg2.tolist()))
    sel = res[[k in keys for k in zip(res.seg1.tolist(), res.seg2.tolist())]].sort_values(["seg1", "seg2"]).reset_index(drop=True)
    obs = observer_enu(sel.lat.to_numpy(), sel.lon.to_numpy(), sel.elev.to_numpy(), idx.lat0, idx.lon0, geoid)
    _G.update(idx=idx, ev1_t_cpa=float(ev1.t_cpa), xw=xw, c=c, max_range_m=max_range_m, score=sel.score.to_numpy(), wind=wind)
    s1a, s2a = sel.seg1.to_numpy(), sel.seg2.to_numpy()
    bounds = np.flatnonzero(np.r_[True, (s1a[1:] != s1a[:-1]) | (s2a[1:] != s2a[:-1]), True])
    jobs = [(int(a), obs[a:b], sel.t_heard1.to_numpy()[a:b], sel["shift"].to_numpy()[a:b], int(s1a[a]), int(s2a[a])) for a, b in zip(bounds[:-1], bounds[1:])]
    xs = np.zeros((len(sel), len(xw))); chosen = np.full((len(sel), len(xw)), -1, int)
    t0 = time.time()
    for n, (i0, out, ch) in enumerate(_pool_map(_rescore_pair, jobs, workers, chunksize=1)):
        xs[i0:i0 + len(out)] = out; chosen[i0:i0 + len(out)] = ch
        if (n + 1) % 50 == 0:
            log(f"  rescored {n + 1}/{len(jobs)} pairs ({time.time() - t0:.0f} s)")
    sel["score12"] = sel.score
    for j in range(len(xw)):
        sel[f"s{j + 3}"] = xs[:, j]
        sel[f"x_icao{j + 1}"] = np.where(chosen[:, j] >= 0, idx.icao[np.maximum(chosen[:, j], 0)], "")
    sel["score"] = sel.score12 + xs.sum(1)
    log(f"rescored {len(sel)} points on {len(jobs)} pairs with {len(xw)} extra events in {time.time() - t0:.0f} s")
    return sel.sort_values("score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# a small lat/lon grid, for refine's fine lattice around one result
# ----------------------------------------------------------------------------------------

@dataclass
class Grid:
    lat: np.ndarray
    lon: np.ndarray
    elev: np.ndarray

    @classmethod
    def make(cls, lat_min, lat_max, lon_min, lon_max, spacing_m, elevation_fn=None):
        lat0 = (lat_min + lat_max) / 2
        dlat = spacing_m / 111320.0
        dlon = spacing_m / (111320.0 * np.cos(np.radians(lat0)))
        la, lo = np.meshgrid(np.arange(lat_min, lat_max, dlat), np.arange(lon_min, lon_max, dlon), indexing="ij")
        la, lo = la.ravel(), lo.ravel()
        el = np.maximum(elevation_fn(la, lo), 0.0) if elevation_fn else np.zeros_like(la)
        return cls(la, lo, el)

    def __len__(self):
        return len(self.lat)
