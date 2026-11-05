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
    alt: np.ndarray      # metres: geometric (WGS-84 ellipsoid) when available, else barometric (pressure altitude)
    gs: np.ndarray       # m/s (nan if missing)
    on_ground: np.ndarray
    # what each report said, beyond the values above (None for traces saved before these existed)
    alt_src: np.ndarray | None = field(default=None, repr=False)   # ALT_SOURCES index of alt
    alt_baro: np.ndarray | None = field(default=None, repr=False)  # pressure altitude, m (nan if none)
    pos_src: np.ndarray | None = field(default=None, repr=False)   # POS_SOURCES index of the position
    flags: np.ndarray | None = field(default=None, repr=False)     # readsb row flags (2: new leg after a gap, 8: slot 3 is geometric)
    track: np.ndarray | None = field(default=None, repr=False)     # true track over ground, deg (nan if missing)
    vrate: np.ndarray | None = field(default=None, repr=False)     # vertical rate, m/s (nan if missing)

    def __len__(self):
        return len(self.t)

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

    def resample(self, sl: slice, dt: float = 1.0):
        """Uniformly resampled (t, lat, lon, alt) over one segment."""
        t = self.t[sl]
        tq = np.arange(t[0], t[-1], dt)
        return (tq, np.interp(tq, t, self.lat[sl]), np.interp(tq, t, self.lon[sl]), np.interp(tq, t, self.alt[sl]))

    def enu(self, sl: slice, lat0: float, lon0: float, dt: float = 1.0):
        tq, la, lo, al = self.resample(sl, dt)
        return tq, lla_to_enu(la, lo, al, lat0, lon0)


ALT_SOURCES = ("none", "geometric", "barometric")
POS_SOURCES = ("unknown", "adsb", "adsr", "tisb", "mlat", "adsc", "other")
FLAG_GEOMETRIC_ALT = 8      # readsb: the altitude in slot 3 is geometric, not barometric


def _pos_source(kind) -> int:
    """readsb's slot-9 source ("adsb_icao", "mlat", "tisb_trackfile", ...) as a POS_SOURCES index."""
    if not isinstance(kind, str):
        return 0
    for i, name in enumerate(POS_SOURCES[1:], 1):
        if kind.startswith(name):
            return i
    return POS_SOURCES.index("other")


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
    num = (int, float)
    t, lat, lon, alt, gs, grd = [], [], [], [], [], []
    asrc, baro, psrc, flg, trk, vr = [], [], [], [], [], []
    for r in rows:
        if r[1] is None or r[2] is None:
            continue
        a = r[3]
        on_ground = a == "ground"
        geom = r[10] if len(r) > 10 else None
        f = r[6] if len(r) > 6 and isinstance(r[6], int) else 0
        a_is_geom = bool(f & FLAG_GEOMETRIC_ALT)
        if isinstance(geom, num):
            am, src = geom * FT, 1
        elif isinstance(a, num):
            am, src = a * FT, 1 if a_is_geom else 2
        elif on_ground:
            am, src = np.nan, 0
        else:
            continue
        t.append(base + r[0]); lat.append(r[1]); lon.append(r[2]); alt.append(am)
        gs.append(r[4] * KT if isinstance(r[4], num) else np.nan); grd.append(on_ground)
        asrc.append(src); baro.append(a * FT if isinstance(a, num) and not a_is_geom else np.nan)
        psrc.append(_pos_source(r[9] if len(r) > 9 else None)); flg.append(f)
        trk.append(r[5] if len(r) > 5 and isinstance(r[5], num) else np.nan)
        vr.append(r[7] * FT / 60.0 if len(r) > 7 and isinstance(r[7], num) else np.nan)
    if not t:
        return None
    return Trace(d.get("icao", fallback_icao), d.get("r"), d.get("t"), d.get("desc"),
                 np.array(t), np.array(lat), np.array(lon), np.array(alt), np.array(gs), np.array(grd),
                 np.array(asrc, np.int8), np.array(baro, float), np.array(psrc, np.int8), np.array(flg, np.int32),
                 np.array(trk, float), np.array(vr, float))


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


_EXTRA = ("alt_src", "alt_baro", "pos_src", "flags", "track", "vrate")


def save_traces(traces: list[Trace], path: str | Path):
    """One (6 or 12, N) array per trace: t, lat, lon, alt, gs, on_ground, then _EXTRA when the
    trace has them. load_traces reads either."""
    def rows(t):
        base = [t.t, t.lat, t.lon, t.alt, t.gs, t.on_ground.astype(float)]
        if t.alt_src is None:
            return np.stack(base)
        return np.stack(base + [np.asarray(getattr(t, k), float) for k in _EXTRA])
    np.savez_compressed(path, n=len(traces),
                        meta=np.array([json.dumps([t.icao, t.reg, t.typ, t.desc]) for t in traces]),
                        **{f"a{i}": rows(t) for i, t in enumerate(traces)})


def load_traces(path: str | Path) -> list[Trace]:
    z = np.load(path, allow_pickle=False)
    meta = z["meta"]                      # read once: indexing z[...] decompresses the member each time
    out = []
    for i in range(int(z["n"])):
        icao, reg, typ, desc = json.loads(str(meta[i]))
        a = z[f"a{i}"]
        extra = ()
        if a.shape[0] >= 12:
            extra = (a[6].astype(np.int8), a[7], a[8].astype(np.int8), a[9].astype(np.int32), a[10], a[11])
        out.append(Trace(icao, reg, typ, desc, a[0], a[1], a[2], a[3], a[4], a[5].astype(bool), *extra))
    return out
