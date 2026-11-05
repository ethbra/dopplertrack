import numpy as np
import pytest

from dopplertrack import doppler, geo


def test_enu_round_trip():
    lat0, lon0 = 40.8765, -73.525
    lat, lon, alt = 40.95, -73.40, 1500.0
    e, n, u = geo.lla_to_enu(lat, lon, alt, lat0, lon0)
    la, lo, al = geo.enu_to_lla(e, n, u, lat0, lon0)
    assert la == pytest.approx(lat, abs=1e-8)
    assert lo == pytest.approx(lon, abs=1e-8)
    assert al == pytest.approx(alt, abs=1e-3)
    # ~10 km NE, altitude ~1.5 km
    assert 8000 < np.hypot(e, n) < 14000
    assert 1400 < u < 1500 + 20


def test_enu_matches_haversine_horizontally():
    lat0, lon0 = 40.0, -74.0
    lat, lon = 40.05, -74.0
    e, n, u = geo.lla_to_enu(lat, lon, 0.0, lat0, lon0)
    assert n == pytest.approx(geo.haversine_m(lat0, lon0, lat, lon), rel=2e-3)
    assert abs(e) < 1.0


def test_ideal_curve_limits_and_symmetry():
    c = 340.0
    f0, v, d, t0 = 130.0, 112.0, 3460.0, 1000.0
    t = np.linspace(t0 - 500, t0 + 500, 2001)
    f = doppler.ideal_curve(t, f0, v, d, t0, c=c)
    assert np.nanmax(f) == pytest.approx(f0 * c / (c - v), rel=1e-3)
    assert np.nanmin(f) == pytest.approx(f0 * c / (c + v), rel=1e-3)
    # monotonically decreasing
    assert np.all(np.diff(f[np.isfinite(f)]) <= 1e-9)
    # at the *heard* CPA (t0 + d/c) the frequency equals f0
    assert doppler.ideal_curve([t0 + d / c], f0, v, d, t0, c=c)[0] == pytest.approx(f0, rel=1e-4)


def test_endpoints_invert():
    c = 340.0
    f0, v = 131.5, 112.0
    fpre, fpost = f0 * c / (c - v), f0 * c / (c + v)
    f0b, vb = doppler.ideal_curve_params_from_endpoints(fpre, fpost, c=c)
    assert f0b == pytest.approx(f0) and vb == pytest.approx(v)


def test_trajectory_curve_matches_closed_form_for_straight_pass():
    c = 340.0
    f0, v, d, t0 = 130.0, 100.0, 3000.0, 500.0
    te = np.arange(0, 1000, 1.0)
    # straight east-bound pass with CPA slant distance d (altitude 1800, lateral 2400)
    pos = np.stack([v * (te - t0), np.full_like(te, 2400.0), np.full_like(te, 1800.0)], -1)
    obs = np.zeros(3)
    tobs, f, r = doppler.curve_from_trajectory(te, pos, obs, f0=f0, c=c)
    ref = doppler.ideal_curve(tobs[5:-5], f0, v, d, t0, c=c)
    assert np.nanmax(np.abs(f[5:-5] - ref)) < 0.05


def test_emission_time_delay_shifts_curve_not_just_in_time():
    """Plotting f against emission time (the naive method) differs from plotting
    it against observation time by more than a pure time shift."""
    c = 340.0
    f0, v, d, t0 = 130.0, 110.0, 3000.0, 500.0
    te = np.arange(0, 1000, 0.5)
    pos = np.stack([v * (te - t0), np.full_like(te, 2000.0), np.full_like(te, 2236.0)], -1)
    tobs, f, r = doppler.curve_from_trajectory(te, pos, np.zeros(3), f0=f0, c=c)
    tq = np.arange(300, 700, 0.5)
    right = np.interp(tq, tobs, f)
    best = min(np.max(np.abs(np.interp(tq - s, te, f) - right)) for s in np.arange(0, 20, 0.25))
    assert best > 0.5


def test_cpa_from_trajectory():
    te = np.arange(0, 600, 1.0)
    pos = np.stack([100.0 * (te - 300), np.full_like(te, 3000.0), np.full_like(te, 1000.0)], -1)
    tcpa, theard, slant, speed = doppler.cpa_from_trajectory(te, pos, np.zeros(3), c=340.0)
    assert tcpa == 300.0
    assert slant == pytest.approx(np.hypot(3000, 1000))
    assert theard == pytest.approx(300 + slant / 340.0)
    assert speed == pytest.approx(100.0)


def _onscreen_example():
    """The video's 'Building the Doppler Curve, Step by Step' panel: c = 343 m/s, f0 = 130 Hz;
    at t_e = 16.910 s the aircraft is at (2367, 1264, 1020) m moving (140, 16.8, 0) m/s, the
    slant range is 2775 m and the radial speed 121.16 m/s. The panel does not show the
    observer, so one is put on the ground where r and v_r come out as shown."""
    p = np.array([2367.0, 1264.0, 1020.0]); v = np.array([140.0, 16.8, 0.0]); r, vr = 2775.0, 121.16
    rh = np.sqrt(r * r - p[2] ** 2)
    a = np.arctan2(v[1], v[0]) + np.arccos(vr * r / (rh * np.hypot(v[0], v[1])))
    te = np.arange(-200, 200, 0.5)
    return te, p + np.outer(te - 16.910, v), p - np.array([rh * np.cos(a), rh * np.sin(a), p[2]])


def test_explain_reproduces_the_videos_panel():
    te, pos, obs = _onscreen_example()
    d = doppler.explain(25.00, te, pos, obs, 130.0, c=343.0)
    assert d["t_e"] == pytest.approx(16.910, abs=1e-3)
    assert d["r"] == pytest.approx(2775, abs=0.5)
    assert d["v_r"] == pytest.approx(121.16, abs=0.01)
    assert d["f"] == pytest.approx(96.07, abs=0.01)
    assert abs(d["f_reception_model"] - d["f"]) > 1.0


def test_emission_time_solves_the_delay_equation():
    te, pos, obs = _onscreen_example()
    for tr in (-50.0, 0.0, 25.0, 100.0):
        t_e = doppler.emission_time(tr, te, pos, obs, c=343.0)
        p = np.array([np.interp(t_e, te, pos[:, k]) for k in range(3)])
        assert t_e + np.linalg.norm(p - obs) / 343.0 == pytest.approx(tr, abs=1e-6)
    with pytest.raises(ValueError):
        doppler.emission_time(1e6, te, pos, obs, c=343.0)


def test_reception_time_model_bends_the_curve():
    te = np.arange(0, 1000, 0.5)
    pos = np.stack([110.0 * (te - 500), np.full_like(te, 2000.0), np.full_like(te, 1500.0)], -1)
    to, f, _ = doppler.curve_from_trajectory(te, pos, np.zeros(3), 100.0, 340.0)
    _, fw, _ = doppler.curve_from_trajectory(te, pos, np.zeros(3), 100.0, 340.0, model="reception")
    tq = np.arange(400, 650, 0.5)
    right = np.interp(tq, to, f)
    best = min(np.max(np.abs(np.interp(tq - s, to, fw) - right)) for s in np.arange(-15, 15, 0.25))
    assert best > 0.3                          # no time shift makes it match
    assert np.allclose(np.interp(tq, to, fw), np.interp(tq, te, f), atol=0.05)     # it is the undelayed curve (resampling differs by < 0.05 Hz)


def test_explain_cli_on_a_day_index(tmp_path, capsys):
    from dopplertrack import cli
    from dopplertrack.adsb import Trace
    from dopplertrack.search import DayIndex
    lat0, lon0 = 40.0, -74.0
    t = 1.7e9 + np.arange(0, 600, 5.0)
    enu = np.stack([100.0 * (t - t[0] - 300), np.full_like(t, 2000.0), np.full_like(t, 1500.0)], -1)
    la, lo, al = geo.enu_to_lla(enu[:, 0], enu[:, 1], enu[:, 2], lat0, lon0)
    DayIndex.build([Trace("abc123", None, None, None, t, la, lo, al, np.full_like(t, 100.0), np.zeros(len(t), bool))],
                   lat0, lon0).save(tmp_path / "day.npz")
    cli.main(["explain", "--day", str(tmp_path / "day.npz"), "--icao", "abc123", "--point", str(lat0), str(lon0),
              "--elev", "0", "--utc", str(1.7e9 + 300 + 2500 / 340.3), "--f0", "100"])
    out = capsys.readouterr().out
    assert "slant range     r = 2500 m" in out and "f = f0 c / (c + v_r) = 100.00 Hz" in out
