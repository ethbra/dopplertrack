"""dopplertrack command line."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import pipeline
from .adsb import INTERP_METHODS, calibrate_baro, load_many
from .audio import Spectrogram
from .doppler import speed_of_sound
from .score import SCORE_MODES
from .search import Grid
from .terrain import Terrain
from .tracks import EventFit


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
    if weather:
        sp.add_argument("--weather", action="store_true",
                        help="look up temperature and wind for the region and time (Open-Meteo ERA5 archive); --temp/--wind override")


def _add_traces(sp):
    sp.add_argument("--traces", required=True, nargs="+", help="one or more trace files (e.g. consecutive UTC days)")
    sp.add_argument("--interp", choices=INTERP_METHODS, default="linear",
                    help="trajectory between ADS-B reports: linear (as the results were made) or hermite (uses each report's speed and track, so turns curve)")
    sp.add_argument("--calibrate-baro", action="store_true",
                    help="move baro-only altitudes onto the geometric datum using aircraft that report both")


def _traces(a):
    traces = load_many(a.traces)
    if getattr(a, "calibrate_baro", False):
        st = calibrate_baro(traces)
        pipeline.log(f"baro calibration: {st}")
    return traces


def _medium(a, lat=None, lon=None, unix_t=None):
    """(c, wind) from --temp/--wind/--weather."""
    temp, wind = a.temp, None
    if getattr(a, "weather", False) and lat is not None and unix_t is not None:
        from .weather import Medium, fetch_weather
        try:
            w = fetch_weather(lat, lon, unix_t)
            m = Medium.from_surface(w["temp_c"], w["wind_speed"], w["wind_from_deg"])
            pipeline.log(f"weather {w['time']} UTC at {lat:.3f}, {lon:.3f}: {w['temp_c']:.1f} C, wind {w['wind_speed']:.1f} m/s from {w['wind_from_deg']:.0f} deg"
                         f" -> c = {m.c:.1f} m/s, wind at path height {m.wind}")
            c, wind = m.c, m.wind
        except Exception as e:      # network or API trouble must not kill a long search
            pipeline.log(f"weather lookup failed ({e}); using --temp/--wind or still air")
            c = None
        if a.temp is not None:
            c = speed_of_sound(a.temp)
        if a.wind:
            from .weather import wind_vector
            wind = wind_vector(*a.wind)
        return (c if c is not None else speed_of_sound(15.0 if temp is None else temp)), wind
    c = speed_of_sound(15.0 if temp is None else temp)
    if a.wind:
        from .weather import wind_vector
        wind = wind_vector(*a.wind)
    return c, wind


def _median_time(traces):
    return float(np.median(np.concatenate([t.t for t in traces]))) if traces else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dopplertrack")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spectrogram", help="audio/video file -> whitened low-band spectrogram (.npz)")
    s.add_argument("audio"); s.add_argument("out"); s.add_argument("--hop", type=float, default=0.25)
    s.add_argument("--fmin", type=float, default=40); s.add_argument("--fmax", type=float, default=450)

    f = sub.add_parser("fit", help="fit an ideal Doppler curve inside a time/frequency window")
    f.add_argument("spec"); f.add_argument("--window", nargs=2, type=float, required=True, metavar=("T0", "T1"))
    f.add_argument("--band", nargs=2, type=float, required=True, metavar=("FLO", "FHI"))
    f.add_argument("--harmonics", type=int, default=1); f.add_argument("--out")
    f.add_argument("--objective", choices=("brightness", "contrast"), default="brightness",
                   help="brightness: mean brightness under the curve (the docs/results fits); contrast: the sweep-weighted ridge contrast detect uses")
    f.add_argument("--temp", type=float, metavar="C", help="air temperature for the speed of sound (default 15 C)")

    d = sub.add_parser("detect", help="scan a whole spectrogram for Doppler events and rank them")
    d.add_argument("spec"); d.add_argument("--band", nargs=2, type=float, required=True, metavar=("FLO", "FHI"))
    d.add_argument("--harmonics", type=int, default=1); d.add_argument("--top", type=int, default=20)
    d.add_argument("--window", type=float, default=240.0); d.add_argument("--out", help="write ranked fits as a JSON list")
    d.add_argument("--temp", type=float, metavar="C", help="air temperature for the speed of sound (default 15 C)")
    d.add_argument("--null", action="store_true", help="also scan the time-reversed spectrogram and mark events above that chance level")

    e = sub.add_parser("extract-region", help="pull traces touching a bbox out of an adsb.lol day archive")
    e.add_argument("archive_dir"); e.add_argument("out")
    e.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    e.add_argument("--keep-jets", action="store_true")

    fd = sub.add_parser("fetch-day", help="stream one UTC day of adsb.lol history from GitHub, keeping traces in a bbox (no disk needed)")
    fd.add_argument("date", help="YYYY-MM-DD (UTC day)"); fd.add_argument("out")
    fd.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    fd.add_argument("--keep-jets", action="store_true"); fd.add_argument("--pod", default="planes-readsb-prod-0")

    r = sub.add_parser("search", help="rank candidate locations for a two-event recording")
    r.add_argument("spec"); r.add_argument("--events", nargs=2, required=True, metavar=("EV1_JSON", "EV2_JSON"))
    _add_traces(r); r.add_argument("--out", required=True)
    r.add_argument("--bbox", nargs=4, type=float, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    r.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"), help="score a single known point instead of a grid")
    r.add_argument("--spacing", type=float, default=200.0); r.add_argument("--harmonics", nargs=2, type=int, default=(1, 1))
    r.add_argument("--no-dem", action="store_true"); r.add_argument("--no-silence", action="store_true")
    r.add_argument("--tol", type=float, default=6.0)
    r.add_argument("--v-tol", type=float, default=0.35, help="speed gate, as a fraction of each event's fitted speed")
    r.add_argument("--d-tol", nargs=2, type=float, default=(0.4, 2.5), metavar=("LO", "HI"), help="distance gate, as multiples of each event's fitted CPA distance")
    r.add_argument("--cpa-cache", metavar="NPZ", help="cache the (point, flight) CPA table here; reused on reruns with the same grid and traces")
    r.add_argument("--score", choices=SCORE_MODES, default="brightness",
                   help="brightness: brightness under the modelled curve; ridge: how closely it follows the ridge traced in the window (for faint curves); line: how closely it follows the line stored in each event by trace-line")
    r.add_argument("--funnel", action="store_true", help="also count the offset-only stage of the filter funnel (as the video reports it); slower on big grids")
    _add_medium(r)

    rf = sub.add_parser("refine", help="fine grid around one search result, optionally adding more events")
    rf.add_argument("spec"); rf.add_argument("results", help="CSV written by search")
    rf.add_argument("--events", nargs=2, required=True, metavar=("EV1_JSON", "EV2_JSON"))
    _add_traces(rf); rf.add_argument("--out", required=True)
    rf.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"), help="the bbox the search used")
    rf.add_argument("--rank", type=int, default=1, help="which flight pair to refine, by rank in the results (1 = best)")
    rf.add_argument("--spacing", type=float, default=50.0); rf.add_argument("--half", type=float, default=600.0, help="half-width of the fine grid, m")
    rf.add_argument("--harmonics", nargs=2, type=int, default=(1, 1))
    rf.add_argument("--extra", nargs="*", default=[], metavar="EV_JSON", help="further events (from detect or fit) to score locked to the same clock")
    rf.add_argument("--extra-harmonics", nargs="*", type=int, default=[])
    rf.add_argument("--no-dem", action="store_true")
    rf.add_argument("--score", choices=SCORE_MODES, default="brightness", help="as for search")
    _add_medium(rf)

    g = sub.add_parser("search-single", help="rank locations for ONE event whose absolute time is known")
    g.add_argument("spec"); g.add_argument("--event", required=True); _add_traces(g)
    g.add_argument("--utc", required=True, help="heard CPA time of the event, unix seconds or ISO-8601 UTC")
    g.add_argument("--tol", type=float, default=30.0); g.add_argument("--out", required=True)
    g.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    g.add_argument("--spacing", type=float, default=200.0); g.add_argument("--harmonics", type=int, default=1)
    g.add_argument("--score", choices=SCORE_MODES, default="brightness", help="as for search")
    _add_medium(g)
    g.add_argument("--v-tol", type=float, default=0.35, help="speed gate, as a fraction of the fitted speed; a faint or turning pass fits badly, so open it (e.g. 1.0)")
    g.add_argument("--d-tol", nargs=2, type=float, default=(0.4, 2.5), metavar=("LO", "HI"), help="distance gate, as multiples of the fitted CPA distance (e.g. 0.05 20)")
    g.add_argument("--no-dem", action="store_true")

    x = sub.add_parser("explain", help="the video's step-by-step Doppler panel for one flight, observer and reception time")
    _add_traces(x)
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
    ex.add_argument("results", help="CSV from search, refine or search-single"); ex.add_argument("out", help=".geojson, .kml or .html")
    ex.add_argument("--top", type=int, default=20); ex.add_argument("--all-points", action="store_true", help="do not keep only the best point per flight pair")
    ex.add_argument("--truth", nargs=2, type=float, metavar=("LAT", "LON"))

    pl = sub.add_parser("plot", help="figures: event fit, score map, best score per flight pair, modelled curves of a candidate")
    pl.add_argument("kind", choices=("event", "map", "pairs", "candidate")); pl.add_argument("out", help="PNG path")
    pl.add_argument("--spec"); pl.add_argument("--event"); pl.add_argument("--results")
    pl.add_argument("--events", nargs=2, metavar=("EV1_JSON", "EV2_JSON")); pl.add_argument("--traces", nargs="+")
    pl.add_argument("--bbox", nargs=4, type=float, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    pl.add_argument("--rank", type=int, default=1); pl.add_argument("--harmonics", nargs="+", type=int, default=[1])
    pl.add_argument("--truth", nargs=2, type=float, metavar=("LAT", "LON")); pl.add_argument("--top", type=int, default=30)
    pl.add_argument("--interp", choices=INTERP_METHODS, default="linear")
    pl.add_argument("--temp", type=float, metavar="C")

    a = ap.parse_args(argv)
    if a.cmd == "spectrogram":
        sp = pipeline.make_spectrogram(a.audio, a.out, hop_s=a.hop, fmin=a.fmin, fmax=a.fmax)
        print(f"wrote {a.out}: {sp.S.shape[0]} bins x {sp.S.shape[1]} frames")
    elif a.cmd == "fit":
        sp = Spectrogram.load(a.spec)
        ef = pipeline.fit_event(sp, a.window[0], a.window[1], a.band[0], a.band[1], harmonics=a.harmonics,
                                objective=a.objective, **({} if a.temp is None else {"c": speed_of_sound(a.temp)}))
        js = json.dumps(ef.to_dict(), indent=1)
        print(js)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(js)
    elif a.cmd == "detect":
        from .tracks import detect_events
        sp = Spectrogram.load(a.spec)
        c = speed_of_sound(15.0 if a.temp is None else a.temp)
        evs = detect_events(sp, a.band[0], a.band[1], c=c, harmonics=a.harmonics, top=a.top, window_s=a.window,
                            progress=lambda k, n: print(f"\r  window {k}/{n}", end="", flush=True))
        print()
        chance = None
        if a.null:
            from .privacy import reversed_spectrogram
            null = detect_events(reversed_spectrogram(sp), a.band[0], a.band[1], c=c, harmonics=a.harmonics, top=3,
                                 window_s=a.window)
            chance = max((e.score for e in null), default=0.0)
            print(f"chance level (best score on the time-reversed spectrogram): {chance:.2f}")
        print(f"{'rank':>4} {'t_cpa s':>9} {'f0 Hz':>7} {'v m/s':>6} {'d m':>7} {'score':>6}")
        for i, e in enumerate(evs):
            mark = "" if chance is None else ("  above chance" if e.score > chance else "")
            print(f"{i + 1:>4} {e.t_cpa:>9.1f} {e.f0:>7.1f} {e.v:>6.0f} {e.d:>7.0f} {e.score:>6.2f}{mark}")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump([e.to_dict() for e in evs], fh, indent=1)
    elif a.cmd == "extract-region":
        pipeline.extract_region(a.archive_dir, tuple(a.bbox), a.out, prop_only=not a.keep_jets)
    elif a.cmd == "fetch-day":
        pipeline.fetch_region(a.date, tuple(a.bbox), a.out, prop_only=not a.keep_jets, pod=a.pod)
    elif a.cmd == "search":
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        traces = _traces(a)
        elev = None if a.no_dem else Terrain().elevation
        if a.point:
            el = elev(np.array([a.point[0]]), np.array([a.point[1]])) if elev else np.zeros(1)
            grid = Grid.points([a.point[0]], [a.point[1]], el)
        elif a.bbox:
            grid = Grid.make(*a.bbox, a.spacing, elev)
        else:
            ap.error("search needs --bbox or --point")
        c, wind = _medium(a, grid.lat0, grid.lon0, _median_time(traces))
        funnel = {"offset": a.funnel}
        res, segs = pipeline.run_search(sp, ev1, ev2, traces, grid, harmonics=tuple(a.harmonics),
                                        silence_d_m=None if a.no_silence else 6000.0, tol_s=a.tol, cpa_cache=a.cpa_cache,
                                        v_tol=a.v_tol, d_tol=tuple(a.d_tol), score_mode=a.score, c=c, wind=wind,
                                        interp=a.interp, funnel=funnel)
        res["lat0"] = grid.lat0; res["lon0"] = grid.lon0
        res.to_csv(a.out, index=False)
        print(res.head(20).to_string())
        if len(res):
            from datetime import datetime, timezone
            b = res.iloc[0]
            print(f"best: {b.icao1}/{b.icao2} at {b.lat:.6f}, {b.lon:.6f}; recording starts "
                  f"{datetime.fromtimestamp(b.rec_start_unix, timezone.utc).isoformat()}")
    elif a.cmd == "refine":
        import pandas as pd
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        extra = [load_event(p) for p in a.extra]
        xh = list(a.extra_harmonics) + [1] * (len(extra) - len(a.extra_harmonics))
        res = pd.read_csv(a.results)
        best = res.sort_values("score", ascending=False).drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)
        row = best.iloc[a.rank - 1]
        traces = _traces(a)
        elev = None if a.no_dem else Terrain().elevation
        segs = pipeline.segments_for_search(traces, tuple(a.bbox), float(row.lat0), float(row.lon0), interp=a.interp)
        row = pipeline.relocate_row(row, segs, ev2.t_cpa - ev1.t_cpa)
        print(f"refining rank {a.rank}: {row.icao1}/{row.icao2} at {row.lat:.5f}, {row.lon:.5f} (score {row.score:.2f})")
        c, wind = _medium(a, float(row.lat), float(row.lon), float(row.t_heard1))
        fine = pipeline.refine(sp, ev1, ev2, segs, row, spacing_m=a.spacing, half_m=a.half, harmonics=tuple(a.harmonics),
                               elevation=elev, extra=extra, extra_harmonics=xh, score_mode=a.score, c=c, wind=wind)
        fine.to_csv(a.out, index=False)
        b = fine.iloc[0]
        print(f"peak score {b.score:.2f} at {b.lat:.6f}, {b.lon:.6f}" + (f"; extra flights {fine.attrs['extra_icao']}" if extra else ""))
        for f, (n, rmax) in pipeline.peak_spread(fine).items():
            print(f"  {n} fine points within {f:.0%} of the peak, the farthest {rmax:.0f} m from it")
        print(fine.head(10).to_string())
    elif a.cmd == "search-single":
        sp = Spectrogram.load(a.spec)
        ev = load_event(a.event)
        traces = _traces(a)
        t_unix = parse_utc(a.utc)
        elev = None if a.no_dem else Terrain().elevation
        grid = Grid.make(*a.bbox, a.spacing, elev)
        c, wind = _medium(a, grid.lat0, grid.lon0, t_unix)
        res, segs = pipeline.run_single_search(sp, ev, traces, grid, t_unix, tol_s=a.tol, harmonics=a.harmonics,
                                               v_tol=a.v_tol, d_tol=tuple(a.d_tol), score_mode=a.score, c=c, wind=wind,
                                               interp=a.interp)
        res["lat0"] = grid.lat0; res["lon0"] = grid.lon0
        res.to_csv(a.out, index=False)
        print(res.head(20).to_string())
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
    from .doppler import curve_from_trajectory, explain
    from .geo import lla_to_enu
    traces = [t for t in _traces(a) if t.icao == a.icao]
    if not traces:
        raise SystemExit(f"{a.icao} is not in these traces")
    t_r = parse_utc(a.utc)
    lat, lon = a.point
    el = a.elev if a.elev is not None else float(Terrain().elevation(np.array([lat]), np.array([lon]))[0])
    c, wind = _medium(a)
    from .search import build_segments
    segs = [s for s in build_segments(traces, lat, lon, interp=a.interp) if s.t[0] <= t_r <= s.t[-1] + 120]
    if not segs:
        raise SystemExit(f"{a.icao} has no airborne segment around that time")
    seg = segs[0]
    obs = lla_to_enu(lat, lon, el, lat, lon)
    d = explain(t_r, seg.t, seg.enu, obs, a.f0, c=c, wind=wind)
    p, v = d["pos"], d["vel"]
    print(f"Building the Doppler curve, step by step ({a.icao}, observer {lat:.6f}, {lon:.6f}, {el:.0f} m)")
    print(f"  1. reception time        t_r = {d['t_r']:.2f}")
    print(f"  2. solve t_r = t_e + r(t_e)/c, c = {c:.1f} m/s" + (f" (along the ray {d['c_eff']:.1f} m/s with wind {wind})" if wind else ""))
    print(f"     emission time         t_e = {d['t_e']:.3f}  (sound in flight {d['delay_s']:.3f} s)")
    print(f"  3. aircraft at t_e       position E {p[0]:.0f}, N {p[1]:.0f}, U {p[2]:.0f} m; velocity ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) m/s")
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
            "candidate": ("spec", "results", "events", "traces", "bbox")}[a.kind]
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
        from .geo import lla_to_enu
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        res = pd.read_csv(a.results)
        best = res.sort_values("score", ascending=False).drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)
        row = best.iloc[a.rank - 1]
        segs = pipeline.segments_for_search(load_many(a.traces), tuple(a.bbox), float(row.lat0), float(row.lon0), interp=a.interp)
        row = pipeline.relocate_row(row, segs, ev2.t_cpa - ev1.t_cpa)
        el = float(Terrain().elevation(np.array([row.lat]), np.array([row.lon]))[0])
        obs = lla_to_enu(row.lat, row.lon, el, row.lat0, row.lon0)
        off = ev1.t_cpa - float(row.t_heard1) + float(row["shift"])
        h = list(a.harmonics) + [a.harmonics[-1]] * (2 - len(a.harmonics))
        c = speed_of_sound(15.0 if a.temp is None else a.temp)
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        for ax, ev, k, hh, lab in ((axs[0], ev1, "seg1", h[0], row.icao1), (axs[1], ev2, "seg2", h[1], row.icao2)):
            plots.plot_modelled(sp, ev, segs[int(row[k])], obs, off, harmonics=hh, ax=ax, c=c,
                                label=f"{lab} at {row.lat:.5f}, {row.lon:.5f}")
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
