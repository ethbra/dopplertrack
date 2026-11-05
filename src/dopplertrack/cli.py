"""dopplertrack command line: spectrogram, detect, fit, fetch, search, refine, audit, scrub."""
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
from .audio import Spectrogram
from .search import DayIndex, rescore_with_extra, run_search
from .terrain import Terrain
from .verdict import verdict
from .tracks import EventFit

BBOX = dict(nargs=4, type=float, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
SCORE = dict(choices=("brightness", "ridge"), default="brightness",
             help="brightness: brightness under the modelled curve; ridge: how closely it follows the ridge traced in the window (for faint curves)")


def load_event(path) -> EventFit:
    with open(path) as fh:
        return EventFit(**json.load(fh))


def load_events(paths) -> list[EventFit]:
    """Events from JSON files holding one event (fit) or a list of them (detect, audit)."""
    out = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        out.extend(EventFit(**e) for e in (d if isinstance(d, list) else [d]))
    return out


def parse_utc(s: str) -> float:
    from datetime import datetime, timezone
    try:
        return float(s)
    except ValueError:
        return datetime.fromisoformat(s).astimezone(timezone.utc).timestamp()


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

    f = sub.add_parser("fit", help="fit an ideal Doppler curve inside a time/frequency window")
    f.add_argument("spec"); f.add_argument("--window", nargs=2, type=float, required=True, metavar=("T0", "T1"))
    f.add_argument("--band", nargs=2, type=float, required=True, metavar=("FLO", "FHI"))
    f.add_argument("--harmonics", type=int, default=1); f.add_argument("--out")
    f.add_argument("--objective", choices=("brightness", "contrast"), default="brightness",
                   help="brightness: mean brightness under the curve (the docs/results fits); contrast: the sweep-weighted ridge contrast detect uses")

    fe = sub.add_parser("fetch", help="one UTC day of adsb.lol history -> the day index a search runs on (streamed from GitHub, no disk needed)")
    fe.add_argument("date", help="YYYY-MM-DD (UTC day)"); fe.add_argument("out", help="day index .npz")
    fe.add_argument("--bbox", **BBOX, help="keep only flights touching this box (default: the whole day, everywhere)")
    fe.add_argument("--archive", metavar="DIR", help="read an already extracted day archive instead of streaming")
    fe.add_argument("--keep-jets", action="store_true"); fe.add_argument("--pod", default="planes-readsb-prod-0")
    fe.add_argument("--origin", nargs=2, type=float, metavar=("LAT0", "LON0"), help="frame origin (default: the flights' mean position)")
    fe.add_argument("--traces", metavar="NPZ", help="also save the raw traces here")

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

    au = sub.add_parser("audit", help="does a recording leak its location? detected flyovers against the chance level")
    au.add_argument("spec"); au.add_argument("--band", nargs=2, type=float, default=(60.0, 300.0), metavar=("FLO", "FHI"))
    au.add_argument("--harmonics", type=int, default=1); au.add_argument("--top", type=int, default=20)
    au.add_argument("--window", type=float, default=240.0); au.add_argument("--out", help="write the flyovers above chance as a JSON list (scrub reads it)")

    sc = sub.add_parser("scrub", help="remove the Doppler tones of given flyovers from an audio or video file (writes WAV)")
    sc.add_argument("audio"); sc.add_argument("out"); sc.add_argument("--events", nargs="+", required=True, help="event JSON (fit, detect or audit)")
    sc.add_argument("--mode", choices=("fill", "mute", "cut"), default="fill")
    sc.add_argument("--sr", type=int, default=16000); sc.add_argument("--width", type=float, default=0.04, help="half-width of the removed band, fraction of frequency")
    sc.add_argument("--pad", type=float, default=30.0, help="seconds beyond each event window"); sc.add_argument("--fmax", type=float, default=1500.0)
    sc.add_argument("--band", nargs=2, type=float, metavar=("FLO", "FHI"), help="after scrubbing, audit the result in this band and scrub what is still above chance")
    sc.add_argument("--passes", type=int, default=2, help="with --band: audit-and-scrub rounds after the first")

    a = ap.parse_args(argv)
    if a.cmd == "spectrogram":
        sp = pipeline.make_spectrogram(a.audio, a.out, hop_s=a.hop, fmin=a.fmin, fmax=a.fmax)
        print(f"wrote {a.out}: {sp.S.shape[0]} bins x {sp.S.shape[1]} frames")
    elif a.cmd == "fit":
        from .tracks import fit_event
        sp = Spectrogram.load(a.spec)
        ef = fit_event(sp, a.window[0], a.window[1], a.band[0], a.band[1], harmonics=a.harmonics, objective=a.objective)
        js = json.dumps(ef.to_dict(), indent=1)
        print(js)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(js)
    elif a.cmd == "detect":
        from .tracks import detect_events
        sp = Spectrogram.load(a.spec)
        evs = detect_events(sp, a.band[0], a.band[1], harmonics=a.harmonics, top=a.top, window_s=a.window,
                            progress=lambda k, n: print(f"\r  window {k}/{n}", end="", flush=True))
        print()
        if a.max_burst is not None:
            evs = [e for e in evs if e.burst <= a.max_burst]
        print(f"{'rank':>4} {'t_cpa s':>9} {'f0 Hz':>7} {'v m/s':>6} {'d m':>7} {'score':>6} {'burst':>5}")
        for i, e in enumerate(evs):
            print(f"{i + 1:>4} {e.t_cpa:>9.1f} {e.f0:>7.1f} {e.v:>6.0f} {e.d:>7.0f} {e.score:>6.2f} {e.burst:>5.2f}")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump([e.to_dict() for e in evs], fh, indent=1)
    elif a.cmd == "fetch":
        pipeline.fetch(a.date, a.out, bbox=a.bbox, archive=a.archive, prop_only=not a.keep_jets, pod=a.pod,
                       origin=a.origin, traces_path=a.traces)
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
            points = (la, lo, elev(la, lo) if elev else np.zeros(1))
        res = run_search(sp, ev1, ev2, idx, harmonics=harm, tol_s=a.tol, v_tol=a.v_tol, d_tol=tuple(a.d_tol),
                         silence_d_m=None if a.no_silence else 6000.0, max_range_m=a.max_range, spacing_m=a.spacing, coarse_m=a.coarse,
                         elevation=elev, score_mode=a.score, workers=a.workers, cand_cache=a.cands, bbox=a.bbox, points=points,
                         t_heard_unix=parse_utc(a.utc) if a.utc else None)
        res.to_csv(a.out, index=False)
        pipeline.log(f"wrote {a.out}")
        print(pipeline.best_per_pair(res).head(a.top).to_string())
        print(verdict(res))
        if a.extra:
            extra = [load_event(p) for p in a.extra]
            xh = list(a.extra_harmonics) + [1] * (len(extra) - len(a.extra_harmonics))
            res = rescore_with_extra(sp, ev1, extra, idx, res, extra_harmonics=xh, top_pairs=a.top_pairs, score_mode=a.score,
                                     max_range_m=a.max_range, workers=a.workers)
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
        fine = pipeline.refine(sp, ev1, ev2, idx, row, spacing_m=a.spacing, half_m=a.half, harmonics=tuple(a.harmonics),
                               elevation=elev, extra=extra, extra_harmonics=xh, score_mode=a.score)
        fine.to_csv(a.out, index=False)
        b = fine.iloc[0]
        print(f"peak score {b.score:.2f} at {b.lat:.6f}, {b.lon:.6f}" + (f"; extra flights {fine.attrs['extra_icao']}" if extra else ""))
        print(fine.head(10).to_string())
    elif a.cmd == "audit":
        _audit(a)
    elif a.cmd == "scrub":
        _scrub(a)


def _audit(a):
    from .privacy import audit
    r = audit(Spectrogram.load(a.spec), a.band[0], a.band[1], harmonics=a.harmonics, top=a.top, window_s=a.window,
              progress=lambda k, n: print(f"\r  window {k}/{n}", end="", flush=True))
    print()
    print(f"chance level (best score in the time-reversed recording): {r['chance']:.2f}")
    print(f"{'rank':>4} {'t_cpa s':>9} {'f0 Hz':>7} {'v m/s':>6} {'d m':>7} {'score':>6}")
    for i, e in enumerate(r["events"]):
        print(f"{i + 1:>4} {e.t_cpa:>9.1f} {e.f0:>7.1f} {e.v:>6.0f} {e.d:>7.0f} {e.score:>6.2f}" + ("  above chance" if e.score > r["chance"] else ""))
    print({"high": "risk: high, two or more flyovers above chance; with the time between them a search needs no clock",
           "medium": "risk: medium, one flyover above chance; enough with a known recording time",
           "low": "risk: low, no flyover above chance"}[r["risk"]])
    if a.out:
        with open(a.out, "w") as fh:
            json.dump([e.to_dict() for e in r["credible"]], fh, indent=1)


def _scrub(a):
    from .audio import compute_spectrogram, load_audio
    from .privacy import audit, scrub_audio, write_wav
    evs = load_events(a.events)
    x, sr = load_audio(a.audio, a.sr)
    y = scrub_audio(x, sr, evs, mode=a.mode, width=a.width, pad_s=a.pad, fmax=a.fmax)
    print(f"{len(evs)} events scrubbed ({a.mode})")
    if a.band and a.mode == "fill":
        # a detected fit can be far enough off the real tone that the mask misses it; the audit
        # of the result fits what is left afresh, so scrub that too
        import scipy.signal as ss
        from math import gcd
        for k in range(a.passes):
            g = gcd(sr, 4000)
            y4 = ss.resample_poly(y, 4000 // g, sr // g).astype(np.float32) if sr != 4000 else y
            r = audit(compute_spectrogram(y4, 4000), a.band[0], a.band[1], top=10)
            print(f"audit after pass {k + 1}: risk {r['risk']}, {len(r['credible'])} above chance {r['chance']:.2f}")
            if not r["credible"]:
                break
            y = scrub_audio(y, sr, r["credible"], mode="fill", width=a.width, pad_s=a.pad, fmax=a.fmax, seed=k + 1)
    write_wav(a.out, y, sr)
    print(f"wrote {a.out}: {len(y) / sr:.1f} s")


if __name__ == "__main__":
    main()
