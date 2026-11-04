"""The parts of colsto's method (and of its discussion) added on top of the core search:
the step-by-step Doppler panel, the reception-time model it warns about, wind and
temperature, velocity-aware trajectories, the baro datum, the filter funnel, the recording
clock, traced-line scoring and the exports."""
import json

import numpy as np
import pandas as pd
import pytest

from dopplertrack import adsb, doppler, export, pipeline, search, weather
from dopplertrack.score import EventWindow, auto_line
from dopplertrack.tracks import EventFit

from conftest import C, LAT0, LON0, T0, synth_trace


# --- the interactive panel ------------------------------------------------------------------

def _onscreen_example():
    """The video's 'Building the Doppler Curve, Step by Step' panel: c = 343, f0 = 130 Hz,
    at t_e = 16.910 s the aircraft is at (2367, 1264, 1020) m moving (140, 16.8, 0) m/s, the
    slant range is 2775 m and the radial speed 121.16 m/s. The panel does not show the
    observer; put one on the ground consistent with r and v_r."""
    p = np.array([2367.0, 1264.0, 1020.0]); v = np.array([140.0, 16.8, 0.0]); r, vr = 2775.0, 121.16
    rh = np.sqrt(r * r - p[2] ** 2)
    a = np.arctan2(v[1], v[0]) + np.arccos(vr * r / (rh * np.hypot(v[0], v[1])))
    obs = p - np.array([rh * np.cos(a), rh * np.sin(a), p[2]])
    te = np.arange(-200, 200, 0.5)
    return te, p + np.outer(te - 16.910, v), obs


def test_explain_reproduces_the_onscreen_panel():
    te, pos, obs = _onscreen_example()
    d = doppler.explain(25.00, te, pos, obs, 130.0, c=343.0)
    assert d["t_e"] == pytest.approx(16.910, abs=1e-3)
    assert d["r"] == pytest.approx(2775, abs=0.5)
    assert d["v_r"] == pytest.approx(121.16, abs=0.01)
    assert d["f"] == pytest.approx(96.07, abs=0.01)
    # the reception-time model takes the radial speed 8 s later in the pass: a different answer
    assert abs(d["f_reception_model"] - d["f"]) > 1.0


def test_emission_time_is_the_unique_root():
    te, pos, obs = _onscreen_example()
    for tr in (-50.0, 0.0, 25.0, 100.0):
        t_e = doppler.emission_time(tr, te, pos, obs, c=343.0)
        p = np.array([np.interp(t_e, te, pos[:, k]) for k in range(3)])
        assert t_e + np.linalg.norm(p - obs) / 343.0 == pytest.approx(tr, abs=1e-6)
    with pytest.raises(ValueError):
        doppler.emission_time(1e6, te, pos, obs, c=343.0)


def test_reception_model_bends_the_curve():
    te = np.arange(0, 1000, 0.5)
    pos = np.stack([110.0 * (te - 500), np.full_like(te, 2000.0), np.full_like(te, 1500.0)], -1)
    to, f, _ = doppler.curve_from_trajectory(te, pos, np.zeros(3), 100.0, 340.0)
    _, fw, _ = doppler.curve_from_trajectory(te, pos, np.zeros(3), 100.0, 340.0, model="reception")
    tq = np.arange(400, 650, 0.5)
    right = np.interp(tq, to, f)
    # no time shift makes the wrong model match
    best = min(np.max(np.abs(np.interp(tq - s, to, fw) - right)) for s in np.arange(-15, 15, 0.25))
    assert best > 0.3


# --- the atmosphere -------------------------------------------------------------------------

def _exact_moving_medium(te, pos, obs, f0, c, wind, tr):
    """Ground truth for a uniform wind: in the air's frame sound travels at c in straight
    lines and both source and observer drift at -wind. Solve |p_s(t_e) - p_o(t_r)| = c (t_r - t_e)
    for each t_r, then f = f0 dt_e/dt_r."""
    from scipy.optimize import brentq
    w = np.array([wind[0], wind[1], 0.0])

    def src(t):
        return np.array([np.interp(t, te, pos[:, k]) for k in range(3)]) - w * t

    out = []
    for t in tr:
        po = obs - w * t
        out.append(brentq(lambda e: np.linalg.norm(src(e) - po) - c * (t - e), t - 60, t - 1e-3))
    out = np.array(out)
    return f0 * np.gradient(out, tr)


def test_wind_model_matches_exact_moving_medium():
    c, f0, wind = 340.0, 100.0, (12.0, -5.0)
    te = np.arange(0, 1000, 0.25)
    pos = np.stack([100.0 * (te - 500), np.full_like(te, 1500.0), np.full_like(te, 800.0)], -1)
    obs = np.zeros(3)
    tr = np.arange(420, 600, 0.5)
    exact = _exact_moving_medium(te, pos, obs, f0, c, wind, tr)
    to, f, _ = doppler.curve_from_trajectory(te, pos, obs, f0, c, wind=wind)
    to0, f0c, _ = doppler.curve_from_trajectory(te, pos, obs, f0, c)
    err_wind = np.max(np.abs(np.interp(tr, to, f) - exact)[2:-2])
    err_still = np.max(np.abs(np.interp(tr, to0, f0c) - exact)[2:-2])
    assert err_still > 0.3              # a 13 m/s wind is visible on the curve
    assert err_wind < 0.1 * err_still   # and the first-order model removes most of it


def test_wind_moves_heard_time_both_ways():
    te = np.arange(0, 600, 1.0)
    pos = np.stack([100.0 * (te - 300), np.full_like(te, 3000.0), np.full_like(te, 1000.0)], -1)
    still = doppler.cpa_from_trajectory(te, pos, np.zeros(3), c=340.0)[1]
    south = doppler.cpa_from_trajectory(te, pos, np.zeros(3), c=340.0, wind=(0.0, -10.0))[1]   # blows towards the observer
    north = doppler.cpa_from_trajectory(te, pos, np.zeros(3), c=340.0, wind=(0.0, 10.0))[1]
    assert south < still < north


def test_weather_helpers():
    assert weather.wind_vector(10.0, 0.0) == pytest.approx((0.0, -10.0))      # from the north: air moves south
    assert weather.wind_vector(10.0, 90.0) == pytest.approx((-10.0, 0.0), abs=1e-9)
    m = weather.Medium.from_surface(20.0, 5.0, 270.0, path_mid_height_m=500.0)
    assert m.c == pytest.approx(doppler.speed_of_sound(20.0 - 3.25))
    assert m.wind[0] > 5.0 and abs(m.wind[1]) < 1e-9                       # from the west, stronger aloft
    assert weather.Medium.from_surface(20.0).wind is None

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"hourly": {"time": ["2026-09-08T19:00", "2026-09-08T20:00", "2026-09-08T21:00"],
                               "temperature_2m": [22.0, 21.0, 20.0], "wind_speed_10m": [3.0, 4.0, 5.0],
                               "wind_direction_10m": [180.0, 190.0, 200.0]}}

    class Session:
        def get(self, url, params, timeout):
            assert params["start_date"] == "2026-09-08" and "temperature_2m" in params["hourly"]
            return Resp()

    w = weather.fetch_weather(40.87, -73.52, 1788900058.6, session=Session())   # 20:40:58 UTC
    assert w["time"] == "2026-09-08T21:00" and w["temp_c"] == 20.0 and w["wind_from_deg"] == 200.0


# --- trajectories ---------------------------------------------------------------------------

def _turning_trace(report_dt=8.0):
    """A level 1.5 deg/s turn at 60 m/s, reported every report_dt s with exact speed and track."""
    om, v, R = np.radians(1.5), 60.0, 60.0 / np.radians(1.5)
    t = T0 + np.arange(0, 240, report_dt)
    tau = t - T0
    e = R * np.sin(om * tau); n = R * (1 - np.cos(om * tau))
    trk = (90.0 - np.degrees(om * tau)) % 360                    # heading east, turning left
    enu = np.stack([e, n, np.full_like(e, 900.0)], -1)
    tr = synth_trace("turn01", t, enu, 0.0)
    tr.gs = np.full(len(t), v); tr.track = trk
    truth = lambda tq: np.stack([R * np.sin(om * (tq - T0)), R * (1 - np.cos(om * (tq - T0))), np.full_like(tq, 900.0)], -1)
    return tr, truth


def test_hermite_follows_a_turn_between_reports():
    tr, truth = _turning_trace()
    obs = np.array([1500.0, 4000.0, 0.0])
    sl = slice(0, len(tr))
    out = {}
    for method in ("linear", "hermite"):
        tq, enu = tr.enu(sl, LAT0, LON0, 1.0, method=method)
        _, f, _ = doppler.curve_from_trajectory(tq, enu, obs, 100.0, C)
        _, ft, _ = doppler.curve_from_trajectory(tq, truth(tq), obs, 100.0, C)
        out[method] = (np.max(np.linalg.norm(enu - truth(tq), axis=1)), np.max(np.abs(f - ft)[2:-2]))
    assert out["hermite"][0] < 0.2 * out["linear"][0]          # position between reports
    assert out["hermite"][1] < 0.25 * out["linear"][1]         # and the Doppler curve (range-rate staircase)
    assert out["linear"][1] > 0.1


def test_hermite_falls_back_without_speed_and_track():
    tr, truth = _turning_trace()
    tr.gs = np.full(len(tr), np.nan); tr.track = None
    tq, enu = tr.enu(slice(0, len(tr)), LAT0, LON0, 1.0, method="hermite")
    assert np.all(np.isfinite(enu)) and np.max(np.linalg.norm(enu - truth(tq), axis=1)) < 30.0


def test_parse_keeps_track_vrate_and_both_altitudes():
    from pathlib import Path
    tr = adsb.parse_trace_file(Path(__file__).parent / "data" / "trace_full_a2f38b.json")
    air = ~tr.on_ground
    assert np.isfinite(tr.track[air]).mean() > 0.9 and np.isfinite(tr.vrate[air]).mean() > 0.5
    both = tr.geom & np.isfinite(tr.alt_baro)
    # QNH 1026 hPa that day: geometric altitude sits ~250 ft above pressure altitude
    assert np.median(tr.alt[both] - tr.alt_baro[both]) == pytest.approx(76, abs=15)
    segs = tr.segments()
    t1, e1 = tr.enu(segs[0], 40.1, -74.1, method="linear")
    t2, e2 = tr.enu(segs[0], 40.1, -74.1, method="hermite")
    assert np.array_equal(t1, t2) and np.median(np.linalg.norm(e1 - e2, axis=1)) < 20


def test_save_load_round_trip_new_fields_and_old_format(tmp_path):
    from pathlib import Path
    tr = adsb.parse_trace_file(Path(__file__).parent / "data" / "trace_full_a2f38b.json")
    adsb.save_traces([tr], tmp_path / "t.npz")
    back = adsb.load_traces(tmp_path / "t.npz")[0]
    assert np.allclose(back.track, tr.track, equal_nan=True) and np.array_equal(back.geom, tr.geom)
    assert np.allclose(back.alt_baro, tr.alt_baro, equal_nan=True)
    # a six-row file from before these fields still loads
    np.savez_compressed(tmp_path / "old.npz", n=1, meta=np.array([json.dumps([tr.icao, tr.reg, tr.typ, tr.desc])]),
                        a0=np.stack([tr.t, tr.lat, tr.lon, tr.alt, tr.gs, tr.on_ground.astype(float)]))
    old = adsb.load_traces(tmp_path / "old.npz")[0]
    assert old.track is None and old.enu(old.segments()[0], 40.1, -74.1, method="hermite")[1].shape[1] == 3
    assert len(adsb.load_many([tmp_path / "t.npz", tmp_path / "old.npz"])) == 2


def test_calibrate_baro_lifts_baro_only_aircraft():
    t = T0 + np.arange(0, 3600, 10.0)
    def tr(icao, alt, geom):
        x = synth_trace(icao, t, np.stack([1.0 * (t - T0), np.zeros_like(t), np.full_like(t, alt)], -1), 100)
        x.geom = np.full(len(t), geom)
        x.alt_baro = x.alt - 76.0 - 0.01 * x.alt          # truth: geometric = baro + 76 m + 1 %
        if not geom:
            x.alt = x.alt_baro.copy()
        return x
    ref = [tr(f"r{k}", 600.0 + 300 * k, True) for k in range(5)]
    lone = tr("lone", 1200.0, False)                  # really at 1200 m, reports only baro (1112 m)
    assert np.max(np.abs(lone.alt - 1200.0)) > 80
    st = adsb.calibrate_baro(ref + [lone])
    assert st["corrected"] == len(t) and st["samples"] == 5 * len(t)
    assert np.max(np.abs(lone.alt - 1200.0)) < 2.0
    for k, r in enumerate(ref):                       # aircraft with a geometric altitude are untouched
        assert np.allclose(r.alt, 600.0 + 300 * k, atol=2.0)


# --- the search -----------------------------------------------------------------------------

def test_funnel_and_recording_clock(synthetic):
    s = synthetic
    grid = search.Grid.make(LAT0 - 0.05, LAT0 + 0.05, LON0 - 0.06, LON0 + 0.06, 500.0)
    funnel = {"offset": True}
    res, segs = pipeline.run_search(s["sp"], s["ev1"], s["ev2"], s["traces"], grid, c=C, tol_s=8.0,
                                    silence_d_m=None, funnel=funnel, progress=False)
    assert set(funnel) == {"offset", "speed_distance", "silence"}
    assert funnel["offset"]["points"] >= funnel["speed_distance"]["points"] >= funnel["silence"]["points"] > 0
    # the synthetic recording's first sample is at T0
    assert res.rec_start_unix.iloc[0] == pytest.approx(T0, abs=2.0)


def test_search_with_hermite_and_wind_still_finds_the_observer(synthetic):
    s = synthetic
    grid = search.Grid.make(LAT0 - 0.03, LAT0 + 0.03, LON0 - 0.04, LON0 + 0.04, 500.0)
    res, _ = pipeline.run_search(s["sp"], s["ev1"], s["ev2"], s["traces"], grid, c=C, tol_s=8.0, silence_d_m=None,
                                 interp="hermite", wind=(1.0, 0.5), progress=False)
    b = res.iloc[0]
    assert np.hypot((b.lat - LAT0) * 111320, (b.lon - LON0) * 111320 * np.cos(np.radians(LAT0))) < 600


def test_peak_spread():
    fine = pd.DataFrame({"lat": [40.0, 40.0, 40.001, 40.01], "lon": [-73.0, -73.001, -73.0, -73.0],
                         "score": [10.0, 9.95, 9.6, 5.0]})
    sp = pipeline.peak_spread(fine)
    assert sp[0.01][0] == 2 and sp[0.01][1] == pytest.approx(84, abs=2)
    assert sp[0.05][0] == 3 and sp[0.05][1] == pytest.approx(111, abs=2)


# --- traced-line scoring --------------------------------------------------------------------

def _line_case():
    t = np.arange(0, 200, 0.25); f = np.arange(60, 260, 0.5)
    from dopplertrack.audio import Spectrogram
    S = np.random.default_rng(0).normal(0, 1, (len(f), len(t))).astype(np.float32)
    ridge = 150.0 - 50.0 / (1 + np.exp(-(t - 100) / 8))
    line = [[float(tk), float(fk)] for tk, fk in zip(t[::8], ridge[::8]) if 50 <= tk <= 150]
    fit = EventFit(t_cpa=100, t_cpa_emit=97, f0=1.0, v=80, d=1500, t_start=40, t_end=160, score=0,
                   f_pre=150, f_post=100, line=line)
    return Spectrogram(t, f, S), fit, t, ridge


def test_line_mode_scores_against_the_traced_line_on_pure_noise():
    sp, fit, t, ridge = _line_case()          # the spectrogram has no curve at all: only the line counts
    w = EventWindow.from_fit(sp, fit, mode="line")
    s = w.curve_score(t, ridge)
    assert s == pytest.approx(100 / 120, abs=0.03)        # the line covers 100 s of the 120 s window
    assert w.sweep_scores(t, ridge, [0.0], [1.05])[0, 0] < 0.1
    assert w.sweep_scores(t, ridge, [15.0], [1.0])[0, 0] < 0.6 * s
    with pytest.raises(ValueError):
        EventWindow.from_fit(sp, EventFit(**{**fit.to_dict(), "line": None}), mode="line")


def test_event_json_round_trip_with_and_without_line():
    _, fit, _, _ = _line_case()
    assert EventFit(**json.loads(json.dumps(fit.to_dict()))).line == fit.line
    plain = EventFit(**{**fit.to_dict(), "line": None})
    assert "line" not in plain.to_dict()


def test_auto_line_follows_the_ridge(synthetic):
    s = synthetic
    line = np.array(auto_line(s["sp"], s["ev1"]))
    assert len(line) > 100
    segs = s["segs"]
    to, ratio, _ = doppler.curve_from_trajectory(segs[0].t, segs[0].enu, np.zeros(3), 120.0, C)
    truth = np.interp(line[:, 0] + T0, to, ratio)
    assert np.median(np.abs(line[:, 1] - truth)) < 1.0


# --- exports --------------------------------------------------------------------------------

def test_exports(tmp_path):
    res = pd.DataFrame({"lat": [40.87, 40.88, 40.5], "lon": [-73.52, -73.53, -73.0], "icao1": ["a", "a", "b"],
                        "icao2": ["c", "c", "d"], "score": [3.0, 2.0, 1.0], "rec_start_unix": [1788898650.0] * 3})
    c = export.candidates(res)
    assert list(c.icao1) == ["a", "b"] and list(c["rank"]) == [1, 2]
    gj = json.loads(export.to_geojson(c))
    p = gj["features"][0]
    assert p["geometry"]["coordinates"] == [-73.52, 40.87]
    assert p["properties"]["streetview"].endswith("viewpoint=40.870000,-73.520000")
    assert p["properties"]["rec_start_utc"].startswith("2026-09-08T20:17:30")
    assert "<Placemark>" in export.to_kml(c) and "leaflet" in export.to_html(c, (40.876, -73.525))


def test_sparse_hand_trace_is_interpolated_but_not_across_long_gaps():
    sp, fit, t, ridge = _line_case()
    clicks = [[float(tk), float(np.interp(tk, t, ridge))] for tk in (40, 60, 80, 100, 120, 160)]   # 40 s gap at 120-160
    w = EventWindow.from_fit(sp, EventFit(**{**fit.to_dict(), "line": clicks}), mode="line")
    ts, fs, ws = w.ridge_data
    assert ws[0][(ts > 45) & (ts < 115)].all() and not ws[0][(ts > 125) & (ts < 155)].any()


def test_cli_trace_line_csv_and_export(tmp_path):
    from dopplertrack import cli
    sp, fit, t, ridge = _line_case()
    sp.save(tmp_path / "rec.npz")
    (tmp_path / "ev.json").write_text(json.dumps({**fit.to_dict(), "line": None}))
    (tmp_path / "hand.csv").write_text("time,freq\n" + "".join(f"{tk},{2 * np.interp(tk, t, ridge)}\n" for tk in range(50, 151, 10)))
    cli.main(["trace-line", str(tmp_path / "rec.npz"), str(tmp_path / "ev.json"), "--csv", str(tmp_path / "hand.csv"), "--harmonic", "2"])
    ev = json.loads((tmp_path / "ev.json").read_text())
    assert len(ev["line"]) == 11 and ev["line"][0][1] == pytest.approx(np.interp(50, t, ridge))
    pd.DataFrame({"lat": [40.87], "lon": [-73.52], "icao": ["abc"], "score": [0.8]}).to_csv(tmp_path / "r.csv", index=False)
    cli.main(["export", str(tmp_path / "r.csv"), str(tmp_path / "c.geojson")])
    assert json.loads((tmp_path / "c.geojson").read_text())["features"][0]["properties"]["icao"] == "abc"
