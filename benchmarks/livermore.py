"""Livermore benchmark: 40 real single flyovers (Pathak 2026, Zenodo record 23424278, CC BY 4.0).

For each pass the dataset gives the smartphone audio, the author's traced frequency curve with
UTC timestamps, the aircraft's ADS-B track (EGM96 MSL altitudes) and, in its results table, the
true horizontal (L), vertical (H) and slant (R) distance from the recorder to the track at
closest approach. The recorder's coordinates are not published and it moved between sessions,
so this scores each model configuration by the slant distance it recovers when it locates the
recorder from that one traced curve (free clock shift and rest frequency), not by a position.
It tests the propagation model alone: the measurement is the author's, not this repository's.

    uv run python benchmarks/livermore.py download     # ~130 MB into data/livermore (git-ignored)
    uv run python benchmarks/livermore.py weather      # that week's Open-Meteo profiles
    uv run python benchmarks/livermore.py run          # writes data/livermore/per_pass.csv, prints the table
"""
import csv, glob, re
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT = Path("data/livermore")

def ts(s):
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00")).timestamp()

def kml(path):
    s = open(path, errors="replace").read()
    w = [ts(x) for x in re.findall(r"<when\s*>([^<]+)</when>", s)]
    c = [list(map(float, x.split())) for x in re.findall(r"<gx:coord\s*>([^<]+)</gx:coord>", s)]
    t = np.array(w); c = np.array(c)
    o = np.argsort(t, kind="stable"); t, c = t[o], c[o]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], c[keep, 1], c[keep, 0], c[keep, 2]        # t, lat, lon, alt (m MSL, EGM96)

def cases():
    rows = {r["case"]: r for r in csv.DictReader(open(ROOT / "raw/final_results_40_cases.csv"))}
    out = []
    for d in sorted(glob.glob(str(ROOT / "x/*/*"))):
        name = Path(d).name
        if name not in rows:
            continue
        k = glob.glob(d + "/*.kml")[0]
        f = np.array([(ts(a), float(b)) for a, b in csv.reader(open(d + "/frequency_vs_time_used.csv")) if a != "time_utc"])
        out.append(dict(name=name, dir=d, row=rows[name], track=kml(k), line=f, start=ts(rows[name]["received_start_utc"])))
    return out


# --- experiment ------------------------------------------------------------
import sys, json, csv
from datetime import datetime, timezone
import numpy as np
from dopplertrack import doppler
from dopplertrack.adsb import Trace
from dopplertrack.geo import lla_to_enu
from dopplertrack.geoid import Geoid
from dopplertrack.terrain import Terrain
from dopplertrack.weather import profile_from_response

def _setup():
    global C, rec, LAT0, LON0, W, geo, ter, N0, mlat, mlon
    C = cases()
    rec = np.load(ROOT / "recorder.npy"); LAT0, LON0 = float(rec[0]), float(rec[1])
    W = json.load(open(ROOT / "weather.json"))
    geo, ter = Geoid(), Terrain()
    N0 = float(geo.undulation([LAT0], [LON0])[0])
    mlat, mlon = 111320.0, 111320.0 * np.cos(np.radians(LAT0))

def track_enu(c, interp):
    t, la, lo, al = c["track"]
    hae = al + geo.undulation(la, lo)                       # KML is EGM96 MSL; frame is ellipsoidal
    tr = Trace("x", None, None, None, t, la, lo, hae, np.full(len(t), np.nan), np.zeros(len(t), bool))
    t0, t1 = c["line"][0, 0] - 300, c["line"][-1, 0] + 300
    m = (t >= t0) & (t <= t1)
    idx = np.flatnonzero(m)
    sl = slice(max(idx[0] - 1, 0), min(idx[-1] + 2, len(t)))
    return tr.enu(sl, LAT0, LON0, 1.0, method=interp)

def misfit(te, pos, obs, c, wind, lt, ly):
    to, f = doppler.curve_from_trajectory(te, pos, obs, 1.0, c, wind)[:2]
    lf = np.log(f); sh = 0.0
    for _ in range(4):
        m = np.interp(lt - sh, to, lf); d = np.gradient(m, lt)
        A = np.stack([-d, np.ones_like(lt)], 1); x = np.linalg.lstsq(A, ly - m, rcond=None)[0]; sh += x[0]
    r = ly - np.interp(lt - sh, to, lf)
    return float(np.sqrt(np.mean((r - r.mean()) ** 2)))

def fit(te, pos, c, wind, lt, ly, geoid_obs):
    def obs_at(E, Nn):
        la, lo = LAT0 + Nn / mlat, LON0 + E / mlon
        h = np.maximum(ter.elevation(la, lo), 0.0) + 1.5 + (geo.undulation(la, lo) if geoid_obs else 0.0)
        return np.stack([E, Nn, np.zeros_like(E)], 1), la, lo, h
    # start from the track point nearest the all-day recorder estimate
    k = int(np.argmin(np.hypot(pos[:, 0], pos[:, 1]))); ce, cn = pos[k, 0], pos[k, 1]
    best = None
    for half, step in ((8000, 400), (600, 50)):
        if best is not None:
            ce, cn = best[1], best[2]
        g = np.arange(-half, half + 1e-9, step)
        E, Nn = [a.ravel() for a in np.meshgrid(ce + g, cn + g)]
        _, la, lo, h = obs_at(E, Nn)
        pts = lla_to_enu(la, lo, h, LAT0, LON0)
        for p in pts:
            v = misfit(te, pos, p, c, wind, lt, ly)
            if best is None or v < best[0]:
                best = (v, p[0], p[1], p)
    p = best[3]
    d = pos - p; i = int(np.argmin(np.linalg.norm(d, axis=1)))
    return best[0], float(np.hypot(d[i, 0], d[i, 1])), float(d[i, 2]), float(np.linalg.norm(d[i]))

configs = [("old: 15 C, linear, old datum", "linear", "const15", False),
           ("15 C, linear, geoid", "linear", "const15", True),
           ("15 C, hermite, geoid", "hermite", "const15", True),
           ("surface temp, hermite, geoid", "hermite", "surface", True),
           ("profile, hermite, geoid", "hermite", "profile", True),
           ("profile + wind, hermite, geoid", "hermite", "profile+wind", True)]
def run():
    import time
    T0 = time.time()
    _setup()
    out = csv.writer(open(ROOT / "per_pass.csv", "w", newline=""))
    out.writerow(["case", "category", "config", "rms_hz", "L_fit", "H_fit", "R_fit", "L_true", "H_true", "R_true"])
    for c in C:
        day = datetime.fromtimestamp(c["start"], timezone.utc).strftime("%Y-%m-%d")
        prof = profile_from_response(W[day], float(np.median(c["line"][:, 0])))
        lt, lf = c["line"][:, 0], c["line"][:, 1]
        ly = np.log(lf)
        Lt, Ht, Rt = (float(c["row"][k]) for k in ("L_ADSB_m", "H_ADSB_above_recorder_m", "R_ADSB_m"))
        for label, interp, cm, gobs in configs:
            te, pos = track_enu(c, interp)
            cc = {"const15": 340.3, "surface": float(prof["atmosphere"].speed(N0 + 150)), "profile": prof["atmosphere"], "profile+wind": prof["atmosphere"]}[cm]
            wind = prof["wind"] if cm == "profile+wind" else None
            v, L, H, R = fit(te, pos, cc, wind, lt, ly, gobs)
            out.writerow([c["name"], c["row"]["category"], label, v * float(np.mean(lf)), L, H, R, Lt, Ht, Rt])
        print(c["name"], "done", round(time.time() - T0), "s", flush=True)


ZENODO = "https://zenodo.org/api/records/23424278/files/{}/content"
FILES = ["final_results_40_cases.csv", "Raw data_V_constant_H_constant.zip", "Raw data_V_constant_H_varying.zip",
         "Raw data_V_varying_H_constant.zip", "Raw data_V_varying_H_varying.zip"]
DAYS = ("2026-05-09", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-16", "2026-05-17")


def download():
    """Fetch the dataset into data/livermore/raw and unpack it into data/livermore/x."""
    import shutil, urllib.parse, zipfile, requests
    raw = ROOT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (raw / f).exists():
            r = requests.get(ZENODO.format(urllib.parse.quote(f)), timeout=600); r.raise_for_status()
            (raw / f).write_bytes(r.content)
        if f.endswith(".zip"):
            zipfile.ZipFile(raw / f).extractall(ROOT / "x")
    shutil.rmtree(ROOT / "x" / "__MACOSX", ignore_errors=True)


def weather():
    """That week's hourly pressure-level profiles at the site, one Open-Meteo request per day."""
    from dopplertrack.weather import PROFILE_URL, PRESSURE_LEVELS, _get
    var = ["temperature_2m", "wind_speed_10m", "wind_direction_10m"]
    for p in PRESSURE_LEVELS:
        var += [f"temperature_{p}hPa", f"geopotential_height_{p}hPa", f"wind_speed_{p}hPa", f"wind_direction_{p}hPa"]
    out = {d: _get(None, PROFILE_URL, {"latitude": "37.6675", "longitude": "-121.7822", "start_date": d, "end_date": d,
                                       "hourly": ",".join(var), "wind_speed_unit": "ms", "timezone": "GMT"}, 60).json() for d in DAYS}
    json.dump(out, open(ROOT / "weather.json", "w"))


def recorder():
    """A frame origin near the recorder: the point that best reproduces every pass's L and H.
    It is not the truth (the recorder moved between sessions) and nothing is scored against it."""
    from scipy.optimize import minimize
    from dopplertrack.geo import enu_to_lla
    C = cases()
    lat0 = np.median([np.median(c["track"][1]) for c in C]); lon0 = np.median([np.median(c["track"][2]) for c in C])
    L = np.array([float(c["row"]["L_ADSB_m"]) for c in C]); H = np.array([float(c["row"]["H_ADSB_above_recorder_m"]) for c in C])
    enus = []
    for c in C:
        t, la, lo, al = c["track"]; tq = np.arange(t[0], t[-1], 0.5)
        enus.append(lla_to_enu(np.interp(tq, t, la), np.interp(tq, t, lo), np.interp(tq, t, al), lat0, lon0))

    def cost(x):
        s = 0.0
        for e, l, h in zip(enus, L, H):
            d = e - x; i = np.argmin(np.linalg.norm(d, axis=1)); s += (np.hypot(d[i, 0], d[i, 1]) - l) ** 2 + (d[i, 2] - h) ** 2
        return s
    _, e0, n0 = min((cost([e, n, 150.0]), e, n) for e in np.arange(-8000, 8001, 500) for n in np.arange(-8000, 8001, 500))
    r = minimize(cost, [e0, n0, 150.0], method="Nelder-Mead", options={"xatol": 0.1, "fatol": 1e-3, "maxiter": 4000})
    la, lo, al = enu_to_lla(r.x[0], r.x[1], r.x[2], lat0, lon0)
    np.save(ROOT / "recorder.npy", np.array([float(la), float(lo), float(al)]))


def summary():
    import pandas as pd
    from scipy.stats import wilcoxon
    d = pd.read_csv(ROOT / "per_pass.csv")
    d["R_err"] = (d.R_fit - d.R_true).abs()
    order = list(dict.fromkeys(d.config))
    base = d[d.config == order[0]].set_index("case").R_err
    print(f"{'configuration':<32} {'median |dR| m':>13} {'mean |dR| m':>11} {'better than old':>16} {'paired p':>9}")
    for cf in order:
        x = d[d.config == cf].set_index("case").R_err.loc[base.index]
        p = wilcoxon(base, x).pvalue if cf != order[0] else float("nan")
        print(f"{cf:<32} {x.median():13.0f} {x.mean():11.0f} {(x < base).sum():>10d} / {len(x)} {p:9.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "download":
        download()
    elif cmd == "weather":
        weather()
    elif cmd == "run":
        if not (ROOT / "recorder.npy").exists():
            recorder()
        run(); summary()
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit("usage: livermore.py download|weather|run|summary")
