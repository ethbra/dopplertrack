"""Shared synthetic recording: two prop flights over a known observer, as in
test_pipeline_synthetic, built once per session."""
import numpy as np
import pytest

from dopplertrack import audio, doppler, search
from dopplertrack.adsb import Trace
from dopplertrack.geo import enu_to_lla

C = 340.0
LAT0, LON0 = 40.8765, -73.525
T0 = 1_700_000_000.0


def synth_trace(icao, t, enu, gs):
    lat, lon, alt = enu_to_lla(enu[:, 0], enu[:, 1], enu[:, 2], LAT0, LON0)
    return Trace(icao, None, "C172", "synthetic", t, lat, lon, alt, np.full_like(t, gs), np.zeros(len(t), bool))


def tone_audio(segs, f0s, sr, dur, harmonics=(1,), amp=0.05, noise=0.4, seed=1, obs=None, c=C):
    obs = np.zeros(3) if obs is None else obs
    tt = np.arange(0, dur, 1 / sr)
    x = noise * np.random.default_rng(seed).standard_normal(len(tt)).astype(np.float32)
    for seg, f0 in zip(segs, f0s):
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, obs, f0=1.0, c=c)
        finst = np.interp(tt + T0, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        for h in harmonics:
            x[m] += amp * np.sin(h * ph[m])
    return tt, x


@pytest.fixture(scope="session")
def synthetic():
    sr = 4000
    t1 = T0 + np.arange(0, 900, 5.0)
    enu1 = np.stack([110.0 * (t1 - (T0 + 300)), np.full_like(t1, 2500.0), np.full_like(t1, 1500.0)], -1)
    t2 = T0 + np.arange(900, 1800, 5.0)
    tau = t2 - (T0 + 1220)
    enu2 = np.stack([3000.0 + 0.02 * tau**2, 95.0 * tau, np.full_like(t2, 1200.0)], -1)
    traces = [synth_trace("aaa001", t1, enu1, 110 / 0.5144), synth_trace("aaa002", t2, enu2, 95 / 0.5144)]
    segs = search.build_segments(traces, LAT0, LON0)
    _, x = tone_audio(segs, (120.0, 130.0), sr, 1800)
    sp = audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)
    from dopplertrack import tracks
    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")
    ev2 = tracks.fit_event(sp, 1150, 1310, 100, 180, c=C, objective="brightness")
    return {"sr": sr, "x": x, "sp": sp, "traces": traces, "segs": segs, "ev1": ev1, "ev2": ev2}
