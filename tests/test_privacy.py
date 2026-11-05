"""audit and scrub, and the countermeasure claims from the video's last part, on synthetic
recordings with known flyovers. The two slow checks (about 2.5 min together) run only with
DOPPLERTRACK_SLOW=1."""
import json
import os

import numpy as np
import pytest
import scipy.signal as ss

from dopplertrack import audio, doppler, privacy, search, tracks
from dopplertrack.adsb import Trace
from dopplertrack.geo import enu_to_lla
from dopplertrack.score import EventWindow

C, LAT0, LON0, T0 = 340.0, 40.8765, -73.525, 1_700_000_000.0
slow = pytest.mark.skipif(not os.environ.get("DOPPLERTRACK_SLOW"), reason="set DOPPLERTRACK_SLOW=1 to run")


def _segment(t, enu, icao):
    lat, lon, alt = enu_to_lla(enu[:, 0], enu[:, 1], enu[:, 2], LAT0, LON0)
    tr = Trace(icao, None, "C172", None, t, lat, lon, alt, np.full_like(t, 100.0), np.zeros(len(t), bool))
    return search.build_segments([tr], LAT0, LON0)[0]


def _pass_east(t_end=900.0):
    t = T0 + np.arange(0, t_end, 5.0)
    return _segment(t, np.stack([110.0 * (t - (T0 + 300)), np.full_like(t, 2500.0), np.full_like(t, 1500.0)], -1), "aaa001")


def _tones(segs, f0s, sr, dur, harmonics=(1,), amp=0.05, noise=0.4, seed=1):
    tt = np.arange(0, dur, 1 / sr)
    x = noise * np.random.default_rng(seed).standard_normal(len(tt)).astype(np.float32)
    for seg, f0 in zip(segs, f0s):
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, np.zeros(3), 1.0, C)
        finst = np.interp(tt + T0, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        for h in harmonics:
            x[m] += amp * np.sin(h * ph[m])
    return x


def _spec(x, sr, fmin=60, fmax=300):
    return audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=fmin, fmax=fmax, whiten_window_s=None)


@pytest.fixture(scope="module")
def one_pass():
    """Ten minutes, one 120 Hz flyover heard at CPA 308.6 s."""
    seg = _pass_east()
    x = _tones([seg], [120.0], 4000, 600)
    sp = _spec(x, 4000)
    return {"x": x, "sr": 4000, "sp": sp, "seg": seg, "ev": tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")}


def _contrast(sp, ev, seg, f0=120.0):
    """Brightness under the true curve minus the mean under copies 10 % above and below it."""
    to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, np.zeros(3), 1.0, C)
    sc = EventWindow.from_fit(sp, ev).sweep_scores(to - T0, ratio * f0 / ev.f0, [0.0], [0.9, 1.0, 1.1])[0]
    return sc[1] - 0.5 * (sc[0] + sc[2])


def test_high_pass_and_boost_do_not_hide_the_curve(one_pass):
    """A 4th-order high-pass at 200 Hz, above the tone, then +20 dB: whitening per frequency bin
    undoes the filter's gain, so the fit lands on the same curve."""
    s = one_pass
    y = (10.0 * ss.sosfilt(ss.butter(4, 200.0, "highpass", fs=s["sr"], output="sos"), s["x"])).astype(np.float32)
    sp = _spec(y, s["sr"])
    ev = tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")
    assert abs(ev.t_cpa - s["ev"].t_cpa) < 3 and abs(ev.v - s["ev"].v) < 0.1 * s["ev"].v
    assert _contrast(sp, ev, s["seg"]) > 0.7 * _contrast(s["sp"], s["ev"], s["seg"])


def test_scrub_fill_removes_the_tone_and_leaves_no_dark_curve(one_pass):
    s = one_pass
    y = privacy.scrub_audio(s["x"], s["sr"], [s["ev"]], mode="fill", fmax=1900, c=C)
    assert len(y) == len(s["x"])
    a = int(470 * s["sr"])                                        # past the scrubbed span: untouched
    assert np.allclose(y[a:], s["x"][a:], atol=1e-4)
    sp2 = _spec(y, s["sr"])
    before, after = _contrast(s["sp"], s["ev"], s["seg"]), _contrast(sp2, s["ev"], s["seg"])
    assert before > 8 and abs(after) < 0.1 * before, (before, after)
    # filled, not notched: the curve's path sits at the noise floor, not below it
    to, ratio, _ = doppler.curve_from_trajectory(s["seg"].t, s["seg"].enu, np.zeros(3), 1.0, C)
    tq = np.linspace(s["ev"].t_start, s["ev"].t_end, 200)
    assert np.median(sp2.sample(tq, 120.0 * np.interp(tq + T0, to, ratio))) > -2.0


def test_scrub_mute_and_cut(one_pass):
    s = one_pass
    sr, ev = s["sr"], s["ev"]
    m = privacy.scrub_audio(s["x"], sr, [ev], mode="mute", pad_s=10)
    assert len(m) == len(s["x"]) and np.all(m[int((ev.t_start - 10) * sr) + 1:int((ev.t_end + 10) * sr) - 1] == 0)
    assert np.allclose(m[int(500 * sr):], s["x"][int(500 * sr):])
    c = privacy.scrub_audio(s["x"], sr, [ev], mode="cut", pad_s=10)
    assert abs(len(c) - (len(s["x"]) - (ev.t_end - ev.t_start + 20) * sr)) <= 2


def test_scrub_cli_writes_a_wav(one_pass, tmp_path):
    from dopplertrack import cli
    s = one_pass
    privacy.write_wav(tmp_path / "in.wav", s["x"][: int(500 * s["sr"])], s["sr"])
    (tmp_path / "ev.json").write_text(json.dumps([s["ev"].to_dict()]))
    cli.main(["scrub", str(tmp_path / "in.wav"), str(tmp_path / "out.wav"), "--events", str(tmp_path / "ev.json"),
              "--mode", "mute", "--sr", "4000", "--pad", "10"])
    y, sr = audio.load_audio(tmp_path / "out.wav", 4000)
    assert sr == 4000 and abs(len(y) - 500 * 4000) <= 4 and np.all(np.abs(y[int(260 * 4000):int(380 * 4000)]) < 1e-3)


@slow
def test_phone_band_keeps_the_curve_in_the_harmonics():
    """300-3400 Hz removes a 120 Hz fundamental; its 4th harmonic traces the same curve."""
    x = _tones([_pass_east(600.0)], [120.0], 8000, 600, harmonics=(1, 2, 3, 4))
    y = ss.sosfilt(ss.butter(6, (300.0, 3400.0), "bandpass", fs=8000, output="sos"), x).astype(np.float32)
    ev = tracks.fit_event(_spec(y, 8000, 300, 900), 230, 400, 400, 720, c=C, objective="brightness")
    assert abs(ev.f0 - 480) < 0.03 * 480
    assert abs(ev.v - 110) < 20 and abs(ev.t_cpa - (300 + np.hypot(2500, 1500) / C)) < 5


@slow
def test_audit_rates_two_flyovers_high_and_noise_low():
    t2 = T0 + np.arange(900, 1800, 5.0)
    tau = t2 - (T0 + 1220)
    seg2 = _segment(t2, np.stack([3000.0 + 0.02 * tau**2, 95.0 * tau, np.full_like(t2, 1200.0)], -1), "aaa002")
    sp = _spec(_tones([_pass_east(), seg2], [120.0, 130.0], 4000, 1800), 4000)
    r = privacy.audit(sp, 90, 180, c=C, top=2)
    assert r["risk"] == "high" and len(r["credible"]) == 2
    cpas = sorted(e.t_cpa for e in r["credible"])
    assert abs(cpas[0] - (300 + np.hypot(2500, 1500) / C)) < 25 and abs(cpas[1] - (1220 + 3000 / C)) < 25
    assert min(e.score for e in r["credible"]) > 1.5 * r["chance"]
    noise = audio.Spectrogram(sp.t[:1800], sp.f, np.random.default_rng(3).normal(0, 1, (len(sp.f), 1800)).astype(np.float32))
    assert privacy.audit(noise, 90, 180, c=C, top=2)["risk"] != "high"
