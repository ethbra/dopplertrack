"""The video's countermeasure claims, checked on the synthetic recording, and the audit and
scrub tools built on them."""
import numpy as np
import scipy.signal as ss

from dopplertrack import audio, doppler, privacy, search, tracks
from dopplertrack.score import EventWindow

from conftest import C, T0, LAT0, LON0, synth_trace, tone_audio


F0 = (120.0, 130.0)        # the synthetic flights' rest frequencies


def _contrast(sp, ev, seg, f0):
    """Brightness under the true curve minus the mean under copies 10 % above and below it."""
    to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, np.zeros(3), 1.0, C)
    w = EventWindow.from_fit(sp, ev)
    sc = w.sweep_scores(to - T0, ratio * f0 / ev.f0, [0.0], [0.9, 1.0, 1.1])[0]
    return sc[1] - 0.5 * (sc[0] + sc[2])


def test_high_pass_and_boost_do_not_hide_the_curve(synthetic):
    s = synthetic
    sos = ss.butter(4, 200.0, "highpass", fs=s["sr"], output="sos")      # cut-off above both tones
    y = (10.0 * ss.sosfilt(sos, s["x"])).astype(np.float32)               # then a 20 dB boost
    sp = audio.compute_spectrogram(y, s["sr"], nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)
    ev = tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")
    assert abs(ev.t_cpa - s["ev1"].t_cpa) < 3 and abs(ev.v - s["ev1"].v) < 0.1 * s["ev1"].v
    assert _contrast(sp, ev, s["segs"][0], F0[0]) > 0.7 * _contrast(s["sp"], s["ev1"], s["segs"][0], F0[0])


def test_phone_band_keeps_the_curve_in_the_harmonics():
    """300-3400 Hz removes the 120 Hz fundamental; its 4th harmonic traces the same curve."""
    sr = 8000
    t1 = T0 + np.arange(0, 600, 5.0)
    enu = np.stack([110.0 * (t1 - (T0 + 300)), np.full_like(t1, 2500.0), np.full_like(t1, 1500.0)], -1)
    segs = search.build_segments([synth_trace("aaa001", t1, enu, 214)], LAT0, LON0)
    _, x = tone_audio(segs, (120.0,), sr, 600, harmonics=(1, 2, 3, 4), amp=0.05, noise=0.4)
    y = ss.sosfilt(ss.butter(6, (300.0, 3400.0), "bandpass", fs=sr, output="sos"), x).astype(np.float32)
    sp = audio.compute_spectrogram(y, sr, nperseg=8192, hop_s=0.5, fmin=300, fmax=900, whiten_window_s=None)
    assert np.median(sp.S[np.abs(sp.f - 500) < 100]) < 1.0       # whitened: nothing stands out but the tones
    ev = tracks.fit_event(sp, 230, 400, 400, 720, c=C, objective="brightness")
    d = np.hypot(2500, 1500)
    assert abs(ev.f0 - 480) < 0.03 * 480
    assert abs(ev.v - 110) < 20 and abs(ev.t_cpa - (300 + d / C)) < 5


def test_scrub_fill_removes_the_tone_and_leaves_the_rest(synthetic):
    s = synthetic
    evs = [s["ev1"], s["ev2"]]
    y = privacy.scrub_audio(s["x"], s["sr"], evs, mode="fill", fmax=1900, c=C)
    assert len(y) == len(s["x"])
    a, b = int(700 * s["sr"]), int(1000 * s["sr"])          # between the two event spans
    assert np.allclose(y[a:b], s["x"][a:b], atol=1e-4)
    sp2 = audio.compute_spectrogram(y, s["sr"], nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)
    for ev, seg, f0 in zip(evs, s["segs"], F0):
        before = _contrast(s["sp"], ev, seg, f0)
        after = _contrast(sp2, ev, seg, f0)
        assert before > 8 and abs(after) < 0.1 * before, (before, after)
        # filling (not notching) leaves no dark curve either: the curve's bins sit at the floor
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, np.zeros(3), 1.0, C)
        tq = np.linspace(ev.t_start, ev.t_end, 200)
        assert np.median(sp2.sample(tq, f0 * np.interp(tq + T0, to, ratio))) > -2.0


def test_scrub_mute_and_cut(synthetic):
    s = synthetic
    sr = s["sr"]
    ev = s["ev1"]
    m = privacy.scrub_audio(s["x"], sr, [ev], mode="mute", pad_s=10)
    assert len(m) == len(s["x"]) and np.all(m[int((ev.t_start - 10) * sr) + 1:int((ev.t_end + 10) * sr) - 1] == 0)
    assert np.allclose(m[int(1000 * sr):], s["x"][int(1000 * sr):])
    c = privacy.scrub_audio(s["x"], sr, [ev], mode="cut", pad_s=10)
    assert abs(len(c) - (len(s["x"]) - (ev.t_end - ev.t_start + 20) * sr)) <= 2


def test_audit_rates_the_synthetic_recording_high_and_noise_low(synthetic):
    s = synthetic
    r = privacy.audit(s["sp"], 90, 180, c=C, top=2)
    assert r["risk"] == "high" and len(r["credible"]) == 2
    cpas = sorted(e.t_cpa for e in r["credible"])
    assert abs(cpas[0] - (300 + np.hypot(2500, 1500) / C)) < 25 and abs(cpas[1] - (1220 + 3000 / C)) < 25
    assert min(e.score for e in r["credible"]) > 1.5 * r["chance"]
    sp = s["sp"].crop(0, 900)
    noise = audio.Spectrogram(sp.t, sp.f, np.random.default_rng(3).normal(0, 1, sp.S.shape).astype(np.float32))
    rn = privacy.audit(noise, 90, 180, c=C, top=2)
    assert rn["risk"] != "high"
