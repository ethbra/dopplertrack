"""dopplertrack command line: spectrogram, detect, fit, fetch, search, refine, and the tools
around them: explain, trace-line, audit, scrub, export, plot."""
from __future__ import annotations

import argparse
import json
import os
import time

# One BLAS thread per process: the heavy stages run a process per core, and the matrices
# are small, so BLAS threads on top of that only contend. Set before numpy loads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from . import pipeline
from .adsb import INTERP_METHODS
from .audio import Spectrogram
from .doppler import speed_of_sound
from .score import SCORE_MODES
from .search import DayIndex, rescore_with_extra, run_search
from .terrain import Terrain
from .verdict import verdict
from .tracks import EventFit

BBOX = dict(nargs=4, type=float, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
SCORE = dict(choices=SCORE_MODES, default="brightness",
             help="brightness: brightness under the modelled curve; ridge: how closely it follows the ridge traced in the window "
                  "(for faint curves); line: how closely it follows the line stored in each event by trace-line")


def load_event(path) -> EventFit:
    with open(path) as fh:
        return EventFit(**json.load(fh))


def load_events(paths) -> list[EventFit]:
    """Events from JSON files holding one event (fit) or a list of them (detect --out)."""
    out = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        out.extend(EventFit(**e) for e in (d if isinstance(d, list) else [d]))
    return out


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def parse_utc(s: str) -> float:
    from datetime import datetime, timezone
    try:
        return float(s)
    except ValueError:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()




def _add_medium(sp, weather=True):
    sp.add_argument("--temp", type=float, metavar="C", help="air temperature for the speed of sound (default 15 C, 340.3 m/s)")
    sp.add_argument("--wind", nargs=2, type=float, metavar=("SPEED", "FROM_DEG"),
                    help="wind along the sound paths: speed m/s and the direction it blows from (deg from north)")
    sp.add_argument("--lapse", action="store_true",
                    help="sound speed falls with height (standard lapse from the --temp surface temperature) instead of one value for every ray")
    if weather:
        sp.add_argument("--weather", action="store_true",
                        help="that hour's temperature profile and mean wind below 3 km (Open-Meteo pressure levels), so each ray gets the "
                             "sound speed of the air it crosses; --temp/--wind override")




def _add_geoid(sp):
    sp.add_argument("--no-geoid", action="store_true",
                    help="treat terrain height (MSL) as ellipsoidal height, as results before the geoid fix did")


def _geoid(a):
    """EGM96 undulation function, or None with --no-geoid. A failed download is fatal: a
    silent fallback would move every observer ~30 m without saying so."""
    if getattr(a, "no_geoid", False):
        return None
    from .geoid import Geoid
    try:
        return Geoid().undulation
    except Exception as e:
        raise SystemExit(f"could not load the EGM96 geoid grid ({e}); pass --no-geoid to run without it")



def _medium(a, lat=None, lon=None, unix_t=None):
    """(c, wind) from --temp, --lapse, --wind and --weather. c is a sound speed, or a
    doppler.Atmosphere when the sound speed varies with height (--lapse, --weather)."""
    from .doppler import Atmosphere, describe_c
    from .weather import wind_vector
    c, wind = None, None
    if getattr(a, "weather", False) and lat is not None and unix_t is not None:
        from .weather import fetch_profile
        try:
            w = fetch_profile(lat, lon, unix_t)
            c, wind = w["atmosphere"], w["wind"]
            pipeline.log(f"weather {w['time']} UTC at {lat:.3f}, {lon:.3f}: {describe_c(c)}; mean wind below 3 km {wind}")
        except Exception as e:      # network or API trouble must not kill a long search
            pipeline.log(f"weather lookup failed ({e}); using --temp/--lapse/--wind or 15 C still air")
    if a.temp is not None or c is None:
        t = 15.0 if a.temp is None else a.temp
        c = Atmosphere.standard(t) if getattr(a, "lapse", False) else speed_of_sound(t)
    if a.wind:
        wind = wind_vector(*a.wind)
    return c, wind



def _c(a):
    """The scalar sound speed for commands that fit ideal curves (fit, detect, audit, scrub)."""
    return speed_of_sound(15.0 if getattr(a, "temp", None) is None else a.temp)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dopplertrack", description="Locate a recording from the Doppler curves of aircraft it caught, matched against historical ADS-B.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spectrogram", help="audio/video file -> whitened low-band spectrogram (.npz)")
    s.add_argument("audio"); s.add_argument("out"); s.add_argument("--hop", type=float, default=0.25)
    s.add_argument("--fmin", type=float, default=40); s.add_argument("--fmax", type=float, default=450)

    d = sub.add_parser("detect", help="scan a whole spectrogram for Doppler events and rank them")
    d.add_argument("spec"); d.add_argument("--band", nargs=2, type=float, required=True, metavar=("FLO", "FHI"))
    d.add_argument("--harmonics", type=int, default=1); d.add_argument("--top", type=int, default=20)
    d.add_argument("--window", type=float, default=240.0); d.add_argument("--out", help="write ranked fits as a JSON list")
    d.add_argument("--max-burst", type=float, help="drop events whose surroundings are mostly broadband burst (a passing train or truck sweeps like an aircraft); e.g. 0.15")
    d.add_argument("--temp", type=float, metavar="C", help="air temperature for the speed of sound (default 15 C)")
    d.add_argument("--null", action="store_true", help="also scan the time-reversed spectrogram and mark events above that chance level")

    f = sub.add_parser("fit", help="fit an ideal Doppler curve inside a time/frequency window")
    f.add_argument("spec"); f.add_argument("--window", nargs=2, type=float, required=True, metavar=("T0", "T1"))
    f.add_argument("--band", nargs=2, type=float, required=True, metavar=("FLO", "FHI"))
    f.add_argument("--harmonics", type=int, default=1); f.add_argument("--out")
    f.add_argument("--objective", choices=("brightness", "contrast"), default="brightness",
                   help="brightness: mean brightness under the curve (the docs/results fits); contrast: the sweep-weighted ridge contrast detect uses")
    f.add_argument("--temp", type=float, metavar="C", help="air temperature for the speed of sound (default 15 C)")

    fe = sub.add_parser("fetch", help="one UTC day of adsb.lol history -> the day index a search runs on (streamed from GitHub, no disk needed)")
    fe.add_argument("date", help="YYYY-MM-DD (UTC day)"); fe.add_argument("out", help="day index .npz")
    fe.add_argument("--bbox", **BBOX, help="keep only flights touching this box (default: the whole day, everywhere)")
    fe.add_argument("--archive", metavar="DIR", help="read an already extracted day archive instead of streaming")
    fe.add_argument("--keep-jets", action="store_true"); fe.add_argument("--pod", default="planes-readsb-prod-0")
    fe.add_argument("--origin", nargs=2, type=float, metavar=("LAT0", "LON0"), help="frame origin (default: the flights' mean position)")
    fe.add_argument("--traces", metavar="NPZ", help="also save the raw traces here")
    fe.add_argument("--from-traces", nargs="+", metavar="NPZ", help="index trace files saved earlier (by --traces or an older fetch-day) instead of streaming")
    fe.add_argument("--interp", choices=INTERP_METHODS, default="linear",
                    help="paths between ADS-B reports: linear, or hermite (uses each report's speed and track, so turns curve)")
    fe.add_argument("--calibrate-baro", action="store_true",
                    help="move baro-only altitudes onto the GNSS datum with an offset fitted from the aircraft that report both")
    _add_geoid(fe)

    r = sub.add_parser("search", help="rank candidate locations over a day index: pair join, lens lattices, locked scoring")
    r.add_argument("spec"); r.add_argument("--events", nargs="+", required=True, metavar="EV_JSON", help="two events, or one with --utc")
    r.add_argument("--day", required=True, help="day index .npz from fetch"); r.add_argument("--out", required=True)
    r.add_argument("--utc", help="one event only: its heard CPA time, unix seconds or ISO-8601 UTC")
    r.add_argument("--bbox", **BBOX, help="only flights in this box (default: the whole index)")
    r.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"), help="score this one known point instead of searching")
    r.add_argument("--cands", metavar="NPZ", help="cache the candidate table here; reused when the index, gates and geometry match")
    r.add_argument("--harmonics", nargs="+", type=int, default=[1, 1], help="harmonics per event")
    r.add_argument("--tol", type=float, help="heard-gap tolerance, s (default 6; 30 with --utc)")
    r.add_argument("--v-tol", type=float, default=0.35, help="speed gate, as a fraction of each event's fitted speed")
    r.add_argument("--d-tol", nargs=2, type=float, default=(0.4, 2.5), metavar=("LO", "HI"), help="distance gate, as multiples of each event's fitted CPA distance")
    r.add_argument("--max-range", type=float, default=12000.0); r.add_argument("--spacing", type=float, default=200.0)
    r.add_argument("--coarse", type=float, default=1000.0, help="coarse sweep spacing, m")
    r.add_argument("--no-silence", action="store_true"); r.add_argument("--no-dem", action="store_true")
    r.add_argument("--dem-zoom", type=int, default=9, help="terrain tile zoom (9: ~300 m pixels, a few hundred tiles for a continent)")
    r.add_argument("--workers", type=int, help="process pool size (default: all cores on Linux, 1 on macOS)")
    r.add_argument("--score", **SCORE); r.add_argument("--top", type=int, default=20)
    r.add_argument("--extra", nargs="*", default=[], metavar="EV_JSON", help="further events (from detect or fit): the top pairs are rescored with them locked to the same clock")
    r.add_argument("--extra-harmonics", nargs="*", type=int, default=[])
    r.add_argument("--top-pairs", type=int, default=200, help="how many of the best two-event pairs to rescore with --extra")
    r.add_argument("--out-extra", metavar="CSV", help="where the rescored top pairs go (default: OUT with _extra before .csv)")
    _add_medium(r); _add_geoid(r)

    rf = sub.add_parser("refine", help="fine grid around one search result, optionally adding more events")
    rf.add_argument("spec"); rf.add_argument("results", help="CSV written by search")
    rf.add_argument("--events", nargs=2, required=True, metavar=("EV1_JSON", "EV2_JSON"))
    rf.add_argument("--day", required=True, help="the day index the search used"); rf.add_argument("--out", required=True)
    rf.add_argument("--rank", type=int, default=1, help="which flight pair to refine, by rank in the results (1 = best)")
    rf.add_argument("--spacing", type=float, default=50.0); rf.add_argument("--half", type=float, default=600.0, help="half-width of the fine grid, m")
    rf.add_argument("--harmonics", nargs=2, type=int, default=(1, 1))
    rf.add_argument("--extra", nargs="*", default=[], metavar="EV_JSON", help="further events (from detect or fit) to score locked to the same clock")
    rf.add_argument("--extra-harmonics", nargs="*", type=int, default=[])
    rf.add_argument("--no-dem", action="store_true"); rf.add_argument("--score", **SCORE)
    _add_medium(rf); _add_geoid(rf)

    x = sub.add_parser("explain", help="the video's step-by-step Doppler panel for one flight, observer and reception time")
    x.add_argument("--day", required=True, help="day index .npz from fetch")
    x.add_argument("--icao", required=True); x.add_argument("--point", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    x.add_argument("--utc", required=True, help="reception time, unix seconds or ISO-8601 UTC")
    x.add_argument("--f0", type=float, default=100.0, help="rest frequency, Hz"); x.add_argument("--elev", type=float, help="observer elevation, m (default: DEM)")
    x.add_argument("--plot", metavar="PNG", help="also plot the curve against emission time, against reception time, and the reception-time model")
    _add_medium(x, weather=False)

    tl = sub.add_parser("trace-line", help="store a traced line in an event JSON (automatic ridge, or a hand-traced CSV) for --score line")
    tl.add_argument("spec"); tl.add_argument("event"); tl.add_argument("--out", help="default: overwrite the event JSON")
    tl.add_argument("--csv", help="hand-traced line: rows of t (recording s), f (Hz); header optional")
    tl.add_argument("--harmonic", type=int, default=1, help="the CSV traces this harmonic (its frequencies are divided by it)")
    tl.add_argument("--min-db", type=float, default=3.0); tl.add_argument("--min-unique-db", type=float, default=3.0)

    au = sub.add_parser("audit", help="does this recording leak its location? detected flyovers against the chance level")
    au.add_argument("spec"); au.add_argument("--band", nargs=2, type=float, default=(60.0, 300.0), metavar=("FLO", "FHI"))
    au.add_argument("--harmonics", type=int, default=1); au.add_argument("--top", type=int, default=20)
    au.add_argument("--window", type=float, default=240.0); au.add_argument("--out", help="write the credible events as a JSON list (input to scrub)")

    sc = sub.add_parser("scrub", help="remove the Doppler tones of given events from an audio/video file (writes WAV)")
    sc.add_argument("audio"); sc.add_argument("out"); sc.add_argument("--events", nargs="+", required=True, help="event JSONs (fit, detect or audit output)")
    sc.add_argument("--mode", choices=("fill", "mute", "cut"), default="fill")
    sc.add_argument("--sr", type=int, default=16000); sc.add_argument("--width", type=float, default=0.04, help="half-width of the removed band, fraction of frequency")
    sc.add_argument("--pad", type=float, default=30.0, help="seconds beyond each event window"); sc.add_argument("--fmax", type=float, default=1500.0)
    sc.add_argument("--temp", type=float, metavar="C")
    sc.add_argument("--band", nargs=2, type=float, metavar=("FLO", "FHI"),
                    help="after scrubbing, audit the result in this band and scrub what is still above chance (see --passes)")
    sc.add_argument("--passes", type=int, default=2, help="with --band: audit-and-scrub rounds after the first (default 2)")

    ex = sub.add_parser("export", help="candidates as GeoJSON, KML or an HTML map, with Street View links")
    ex.add_argument("results", help="CSV from search or refine"); ex.add_argument("out", help=".geojson, .kml or .html")
    ex.add_argument("--top", type=int, default=20); ex.add_argument("--all-points", action="store_true", help="do not keep only the best point per flight pair")
    ex.add_argument("--truth", nargs=2, type=float, metavar=("LAT", "LON"))

    pl = sub.add_parser("plot", help="figures: event fit, score map, best score per flight pair, modelled curves of a candidate")
    pl.add_argument("kind", choices=("event", "map", "pairs", "candidate")); pl.add_argument("out", help="PNG path")
    pl.add_argument("--spec"); pl.add_argument("--event"); pl.add_argument("--results")
    pl.add_argument("--events", nargs=2, metavar=("EV1_JSON", "EV2_JSON")); pl.add_argument("--day", help="day index .npz the search used")
    pl.add_argument("--rank", type=int, default=1); pl.add_argument("--harmonics", nargs="+", type=int, default=[1])
    pl.add_argument("--truth", nargs=2, type=float, metavar=("LAT", "LON")); pl.add_argument("--top", type=int, default=30)
    pl.add_argument("--temp", type=float, metavar="C")
    _add_geoid(pl)

    a = ap.parse_args(argv)
    if a.cmd == "spectrogram":
        sp = pipeline.make_spectrogram(a.audio, a.out, hop_s=a.hop, fmin=a.fmin, fmax=a.fmax)
        print(f"wrote {a.out}: {sp.S.shape[0]} bins x {sp.S.shape[1]} frames")
    elif a.cmd == "fit":
        from .tracks import fit_event
        sp = Spectrogram.load(a.spec)
        ef = fit_event(sp, a.window[0], a.window[1], a.band[0], a.band[1], harmonics=a.harmonics, objective=a.objective,
                       **({} if a.temp is None else {"c": _c(a)}))
        js = json.dumps(ef.to_dict(), indent=1)
        print(js)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(js)
    elif a.cmd == "detect":
        from .tracks import detect_events
        sp = Spectrogram.load(a.spec)
        evs = detect_events(sp, a.band[0], a.band[1], c=_c(a), harmonics=a.harmonics, top=a.top, window_s=a.window,
                            progress=lambda k, n: print(f"\r  window {k}/{n}", end="", flush=True))
        print()
        if a.max_burst is not None:
            evs = [e for e in evs if e.burst <= a.max_burst]
        chance = None
        if a.null:
            from .privacy import reversed_spectrogram
            null = detect_events(reversed_spectrogram(sp), a.band[0], a.band[1], c=_c(a), harmonics=a.harmonics, top=3, window_s=a.window)
            chance = max((e.score for e in null), default=0.0)
            print(f"chance level (best score on the time-reversed spectrogram): {chance:.2f}")
        print(f"{'rank':>4} {'t_cpa s':>9} {'f0 Hz':>7} {'v m/s':>6} {'d m':>7} {'score':>6} {'burst':>5}")
        for i, e in enumerate(evs):
            mark = "" if chance is None else ("  above chance" if e.score > chance else "")
            print(f"{i + 1:>4} {e.t_cpa:>9.1f} {e.f0:>7.1f} {e.v:>6.0f} {e.d:>7.0f} {e.score:>6.2f} {e.burst:>5.2f}{mark}")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump([e.to_dict() for e in evs], fh, indent=1)
    elif a.cmd == "fetch":
        pipeline.fetch(a.date, a.out, bbox=a.bbox, archive=a.archive, prop_only=not a.keep_jets, pod=a.pod,
                       origin=a.origin, traces_path=a.traces, from_traces=a.from_traces, interp=a.interp,
                       calibrate=a.calibrate_baro, geoid=_geoid(a))
    elif a.cmd == "search":
        if len(a.events) not in (1, 2) or (len(a.events) == 1) != (a.utc is not None):
            ap.error("give two events, or one event with --utc")
        t0 = time.time()
        sp = Spectrogram.load(a.spec)
        evs = [load_event(p) for p in a.events]
        ev1, ev2 = evs[0], (evs[1] if len(evs) == 2 else None)
        harm = tuple(a.harmonics) + (1,) * (2 - len(a.harmonics))
        idx = DayIndex.load(a.day)
        pipeline.log(f"index: {len(idx)} segments, {len(idx.t)} points, loaded in {time.time() - t0:.0f} s")
        elev = None if a.no_dem else Terrain(zoom=a.dem_zoom).elevation
        points = None
        if a.point:
            la, lo = np.array([a.point[0]]), np.array([a.point[1]])
            points = (la, lo, np.maximum(elev(la, lo), 0.0) if elev else np.zeros(1))
        geoid = _geoid(a)
        t_utc = parse_utc(a.utc) if a.utc else None
        mlat, mlon = (a.point if a.point else ((a.bbox[0] + a.bbox[1]) / 2, (a.bbox[2] + a.bbox[3]) / 2) if a.bbox else (idx.lat0, idx.lon0))
        c, wind = _medium(a, mlat, mlon, t_utc if t_utc is not None else float(np.median(idx.t)))
        res = run_search(sp, ev1, ev2, idx, harmonics=harm, tol_s=a.tol, v_tol=a.v_tol, d_tol=tuple(a.d_tol),
                         silence_d_m=None if a.no_silence else 6000.0, max_range_m=a.max_range, spacing_m=a.spacing, coarse_m=a.coarse,
                         elevation=elev, score_mode=a.score, workers=a.workers, cand_cache=a.cands, bbox=a.bbox, points=points,
                         t_heard_unix=t_utc, c=c, wind=wind, geoid=geoid, funnel={})
        res.to_csv(a.out, index=False)
        pipeline.log(f"wrote {a.out}")
        print(pipeline.best_per_pair(res).head(a.top).to_string())
        print(verdict(res))
        if len(res):
            from datetime import datetime, timezone
            print(f"the best row implies the recording started {datetime.fromtimestamp(float(res.rec_start_unix.iloc[0]), timezone.utc).isoformat()}")
        if a.extra:
            extra = [load_event(p) for p in a.extra]
            xh = list(a.extra_harmonics) + [1] * (len(extra) - len(a.extra_harmonics))
            res = rescore_with_extra(sp, ev1, extra, idx, res, extra_harmonics=xh, top_pairs=a.top_pairs, score_mode=a.score,
                                     max_range_m=a.max_range, workers=a.workers, c=c, wind=wind, geoid=geoid)
            out = a.out_extra or (a.out[:-4] + "_extra.csv" if a.out.endswith(".csv") else a.out + "_extra.csv")
            res.to_csv(out, index=False)
            pipeline.log(f"wrote {out}")
            print(pipeline.best_per_pair(res).head(a.top).to_string())
            print(verdict(res))
    elif a.cmd == "refine":
        import pandas as pd
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        extra = [load_event(p) for p in a.extra]
        xh = list(a.extra_harmonics) + [1] * (len(extra) - len(a.extra_harmonics))
        row = pipeline.best_per_pair(pd.read_csv(a.results)).iloc[a.rank - 1]
        idx = DayIndex.load(a.day)
        if idx.icao[int(row.seg1)] != row.icao1 or idx.icao[int(row.seg2)] != row.icao2:
            ap.error(f"{a.results} was not made from {a.day} (segment {row.seg1} is {idx.icao[int(row.seg1)]}, not {row.icao1})")
        elev = None if a.no_dem else Terrain().elevation
        print(f"refining rank {a.rank}: {row.icao1}/{row.icao2} at {row.lat:.5f}, {row.lon:.5f} (score {row.score:.2f})")
        c, wind = _medium(a, float(row.lat), float(row.lon), float(row.t_heard1))
        fine = pipeline.refine(sp, ev1, ev2, idx, row, spacing_m=a.spacing, half_m=a.half, harmonics=tuple(a.harmonics),
                               elevation=elev, extra=extra, extra_harmonics=xh, score_mode=a.score, c=c, wind=wind, geoid=_geoid(a))
        fine.to_csv(a.out, index=False)
        b = fine.iloc[0]
        print(f"peak score {b.score:.2f} at {b.lat:.6f}, {b.lon:.6f}" + (f"; extra flights {fine.attrs['extra_icao']}" if extra else ""))
        for f, (n, rmax) in pipeline.peak_spread(fine).items():
            print(f"  {n} fine points within {f:.0%} of the peak, the farthest {rmax:.0f} m from it")
        print(fine.head(10).to_string())

    elif a.cmd == "explain":
        _explain(a)
    elif a.cmd == "trace-line":
        _trace_line(a)
    elif a.cmd == "audit":
        _audit(a)
    elif a.cmd == "scrub":
        _scrub(a)
    elif a.cmd == "export":
        _export(a)
    elif a.cmd == "plot":
        _plot(a, ap)




def _explain(a):
    from .doppler import curve_from_trajectory, describe_c, explain
    from .search import observer_enu
    idx = DayIndex.load(a.day)
    t_r = parse_utc(a.utc)
    segs = [s for s in idx.segs if s.icao == a.icao and s.t[0] <= t_r <= s.t[-1] + 120]
    if not segs:
        raise SystemExit(f"{a.icao} has no airborne segment in {a.day} around that time")
    seg = segs[0]
    lat, lon = a.point
    el = a.elev if a.elev is not None else max(float(Terrain().elevation(np.array([lat]), np.array([lon]))[0]), 0.0)
    geoid = _geoid(a)
    obs = observer_enu(np.array([lat]), np.array([lon]), np.array([el]), idx.lat0, idx.lon0, geoid)[0]
    c, wind = _medium(a)
    d = explain(t_r, seg.t, seg.enu, obs, a.f0, c=c, wind=wind)
    p, v = d["pos"], d["vel"]
    print(f"Building the Doppler curve, step by step ({a.icao}, observer {lat:.6f}, {lon:.6f}, {el:.0f} m above sea level)")
    print(f"  1. reception time        t_r = {d['t_r']:.2f}")
    print(f"  2. solve t_r = t_e + r(t_e)/c, c = {describe_c(c)}" + (f" (along this ray {d['c_eff']:.1f} m/s)" if wind or not isinstance(c, float) else ""))
    print(f"     emission time         t_e = {d['t_e']:.3f}  (sound in flight {d['delay_s']:.3f} s)")
    print(f"  3. aircraft at t_e       position E {p[0] - obs[0]:.0f}, N {p[1] - obs[1]:.0f}, U {p[2] - obs[2]:.0f} m from the observer; velocity ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) m/s")
    print(f"     slant range           r = {d['r']:.0f} m")
    print(f"     radial speed          v_r = {d['v_r']:.2f} m/s (positive receding)")
    print(f"  4. Doppler               f = f0 c / (c + v_r) = {a.f0:.1f} * {d['c_eff']:.1f} / ({d['c_eff']:.1f} + {d['v_r']:.2f}) = {d['f']:.2f} Hz")
    print(f"  (reception-time model: v_r = {d['v_r_reception_model']:.2f} m/s, f = {d['f_reception_model']:.2f} Hz, "
          f"{d['f_reception_model'] - d['f']:+.2f} Hz off)")
    if a.plot:
        import matplotlib.pyplot as plt
        to, f, _ = curve_from_trajectory(seg.t, seg.enu, obs, a.f0, c, wind)
        _, fw, _ = curve_from_trajectory(seg.t, seg.enu, obs, a.f0, c, wind, model="reception")
        m = np.abs(to - t_r) < 300
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(seg.t[m] - t_r, f[m], label="at emission time (position)")
        ax.plot(to[m] - t_r, f[m], label="heard (delayed)")
        ax.plot(to[m] - t_r, fw[m], "--", label="reception-time model (wrong): the undelayed curve")
        ax.axvline(0, color="k", lw=0.5); ax.axvline(d["t_e"] - t_r, color="k", lw=0.5, ls=":")
        ax.set_xlabel("s from t_r"); ax.set_ylabel("Hz"); ax.legend(); fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")


def _trace_line(a):
    from .score import auto_line
    sp = Spectrogram.load(a.spec)
    ev = load_event(a.event)
    if a.csv:
        rows = []
        with open(a.csv) as fh:
            for ln in fh:
                parts = ln.replace(";", ",").replace("\t", ",").split(",")
                try:
                    rows.append([float(parts[0]), float(parts[1]) / a.harmonic])
                except (ValueError, IndexError):
                    continue                # header or blank line
        line = sorted(rows)
    else:
        line = auto_line(sp, ev, a.min_db, a.min_unique_db)
    if len(line) < 5:
        raise SystemExit(f"only {len(line)} line points; nothing to score against")
    ev.line = line
    save_json(a.out or a.event, ev.to_dict())
    span = (max(line[0][0], ev.t_start), min(line[-1][0], ev.t_end))
    cover = max(span[1] - span[0], 0.0) / max(ev.t_end - ev.t_start, 1e-9)
    print(f"{len(line)} line points spanning {line[0][0]:.1f}-{line[-1][0]:.1f} s ({cover:.0%} of the fit window); wrote {a.out or a.event}")


def _audit(a):
    from .privacy import audit
    sp = Spectrogram.load(a.spec)
    r = audit(sp, a.band[0], a.band[1], harmonics=a.harmonics, top=a.top, window_s=a.window,
              progress=lambda k, n: print(f"\r  window {k}/{n}", end="", flush=True))
    print()
    print(f"chance level (best Doppler-curve score in the time-reversed recording): {r['chance']:.2f}")
    print(f"{'rank':>4} {'t_cpa s':>9} {'f0 Hz':>7} {'v m/s':>6} {'d m':>7} {'score':>6}")
    for i, e in enumerate(r["events"]):
        print(f"{i + 1:>4} {e.t_cpa:>9.1f} {e.f0:>7.1f} {e.v:>6.0f} {e.d:>7.0f} {e.score:>6.2f}" + ("  above chance" if e.score > r["chance"] else ""))
    n = len(r["credible"])
    msg = {"high": f"{n} flyovers above chance: two of them, with the time between them intact, are enough to search without knowing when or where",
           "medium": "1 flyover above chance: enough to locate the recording if its time is known",
           "low": "no flyover above chance"}[r["risk"]]
    print(f"risk: {r['risk'].upper()} - {msg}")
    if a.out:
        save_json(a.out, [e.to_dict() for e in r["credible"]])


def _scrub(a):
    from .audio import load_audio
    from .privacy import scrub_audio, write_wav
    evs = load_events(a.events)
    x, sr = load_audio(a.audio, a.sr)
    c = speed_of_sound(15.0 if a.temp is None else a.temp)
    y = scrub_audio(x, sr, evs, mode=a.mode, width=a.width, pad_s=a.pad, fmax=a.fmax, c=c)
    print(f"{len(evs)} events scrubbed ({a.mode})")
    if a.band and a.mode == "fill":
        # a detected fit can be far enough off the real tone that the first pass misses it; the
        # audit of the result fits the remaining tone afresh, so scrub what it still finds
        from math import gcd
        import scipy.signal as ss
        from .audio import compute_spectrogram
        from .privacy import audit
        for k in range(a.passes):
            g = gcd(sr, 4000)
            y4 = ss.resample_poly(y, 4000 // g, sr // g).astype(np.float32) if sr != 4000 else y
            r = audit(compute_spectrogram(y4, 4000), a.band[0], a.band[1], c=c, top=10)
            print(f"audit after pass {k + 1}: risk {r['risk']}, {len(r['credible'])} flyovers above chance {r['chance']:.2f}")
            if not r["credible"]:
                break
            y = scrub_audio(y, sr, r["credible"], mode="fill", width=a.width, pad_s=a.pad, fmax=a.fmax, c=c, seed=k + 1)
    write_wav(a.out, y, sr)
    print(f"wrote {a.out}: {len(y) / sr:.1f} s")
    if not a.band:
        print("check it: run `spectrogram` and `audit` on the output (or pass --band); flyovers you did not pass in are still there")


def _export(a):
    import pandas as pd
    from . import export
    res = pd.read_csv(a.results)
    c = export.candidates(res, top=a.top, per_pair=not a.all_points)
    ext = a.out.rsplit(".", 1)[-1].lower()
    text = {"geojson": export.to_geojson, "json": export.to_geojson, "kml": export.to_kml}.get(ext)
    out = export.to_html(c, tuple(a.truth) if a.truth else None) if ext in ("html", "htm") else (text(c) if text else None)
    if out is None:
        raise SystemExit("output must end in .geojson, .kml or .html")
    with open(a.out, "w") as fh:
        fh.write(out)
    print(f"wrote {len(c)} candidates to {a.out}")
    for _, r in c.head(5).iterrows():
        print(f"  #{int(r['rank'])} {r.lat:.6f}, {r.lon:.6f}  score {r.score:.3f}  {export.streetview_url(r.lat, r.lon)}")


def _plot(a, ap):
    import matplotlib.pyplot as plt
    import pandas as pd
    from . import plots
    need = {"event": ("spec", "event"), "map": ("results",), "pairs": ("results",),
            "candidate": ("spec", "results", "events", "day")}[a.kind]
    missing = [k for k in need if getattr(a, k) is None]
    if missing:
        ap.error(f"plot {a.kind} needs " + ", ".join("--" + k for k in missing))
    if a.kind == "event":
        fig, ax = plt.subplots(figsize=(9, 4))
        plots.plot_event_fit(Spectrogram.load(a.spec), load_event(a.event), harmonics=a.harmonics[0], ax=ax)
    elif a.kind == "map":
        fig, ax = plt.subplots(figsize=(7, 7))
        plots.plot_score_map(pd.read_csv(a.results), truth=a.truth, ax=ax)
    elif a.kind == "pairs":
        fig, ax = plt.subplots(figsize=(9, 4))
        plots.plot_pair_scores(pd.read_csv(a.results), top=a.top, ax=ax)
    else:
        from .search import observer_enu
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        row = pipeline.best_per_pair(pd.read_csv(a.results)).iloc[a.rank - 1]
        idx = DayIndex.load(a.day)
        el = max(float(Terrain().elevation(np.array([row.lat]), np.array([row.lon]))[0]), 0.0)
        obs = observer_enu(np.array([row.lat]), np.array([row.lon]), np.array([el]), idx.lat0, idx.lon0, _geoid(a))[0]
        off = ev1.t_cpa - float(row.t_heard1) + float(row["shift"])
        h = list(a.harmonics) + [a.harmonics[-1]] * (2 - len(a.harmonics))
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        for ax, ev, k, hh, lab in ((axs[0], ev1, "seg1", h[0], row.icao1), (axs[1], ev2, "seg2", h[1], row.icao2)):
            plots.plot_modelled(sp, ev, idx.segs[int(row[k])], obs, off, harmonics=hh, ax=ax, c=_c(a),
                                label=f"{lab} at {row.lat:.5f}, {row.lon:.5f}")
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

