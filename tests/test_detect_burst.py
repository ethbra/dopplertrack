"""A broadband burst (a train passing the microphone) is flagged on the events it surrounds."""
import numpy as np

from dopplertrack.audio import Spectrogram
from dopplertrack.tracks import EventFit, burst_fraction, burst_mask


def test_burst_mask_marks_the_loud_stretch_only():
    rng = np.random.default_rng(0)
    t = np.arange(0, 3600, 0.25); f = np.arange(60, 300, 2.0)
    S = rng.normal(0, 1, (len(f), len(t))).astype(np.float32)
    S[:, (t > 1000) & (t < 1090)] += 12.0                      # 90 s where the whole band is loud
    m = burst_mask(Spectrogram(t, f, S))
    assert m[(t > 1010) & (t < 1080)].mean() > 0.95 and m[(t < 900) | (t > 1200)].mean() < 0.02
    assert burst_fraction(m, t, 1045.0) > 0.45 and burst_fraction(m, t, 2000.0) < 0.05


def test_event_fit_loads_without_the_burst_field():
    e = EventFit(t_cpa=1, t_cpa_emit=0, f0=100, v=50, d=1000, t_start=0, t_end=2, score=1, f_pre=110, f_post=90)
    assert e.burst == 0.0 and "burst" in e.to_dict()
