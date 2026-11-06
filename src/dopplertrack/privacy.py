"""Countermeasures: find out whether a recording leaks its location, and remove the leak.

The video's last part asks what defeats the method and finds that the usual tools do not: a
high-pass filter only lowers the tone (whitening per frequency bin brings it back, as does a
boost), talking over it adds energy elsewhere, and a phone call's 300-3400 Hz band removes
the fundamental but keeps the harmonics, which trace the same curve (every harmonic is
shifted by the same factor). What does work is to remove the tone itself, or the timing.

`audit` runs the detector and compares each event with what the detector finds in the
*time-reversed* spectrogram. A Doppler curve only ever falls, so reversed it is a rising
curve the model cannot fit, while noise, bursts, hum and speech look the same either way:
the best reversed score is the level chance reaches in this very recording.

`scrub` rewrites the audio around each event so the tone is gone: in "fill" mode the
short-time spectrum along the curve (all harmonics up to fmax, following the actual peak
near the fitted curve) is replaced by noise at that bin's own level. A notch (zeroing those bins) would leave a dark curve of the
same shape, which leaks exactly as much. "mute" silences each event window; "cut" deletes
it, which also destroys the time between events that the two-event search relies on.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .audio import Spectrogram
from .doppler import ideal_curve, speed_of_sound
from .tracks import EventFit, detect_events

SCRUB_MODES = ("fill", "mute", "cut")


def reversed_spectrogram(sp: Spectrogram) -> Spectrogram:
    return Spectrogram(sp.t, sp.f, np.ascontiguousarray(sp.S[:, ::-1]))


def audit(sp: Spectrogram, f_lo: float, f_hi: float, harmonics: int = 1, top: int = 20,
          window_s: float = 240.0, c: float | None = None, workers: int | None = None,
          progress=None) -> dict:
    """Detected events, the chance level (best score on the reversed spectrogram) and a risk
    level: "high" with two or more events above chance (the two-event search needs no clock),
    "medium" with one (locatable if the recording time is known), else "low"."""
    evs = detect_events(sp, f_lo, f_hi, c=c, harmonics=harmonics, top=top, window_s=window_s,
                        workers=workers, progress=progress)
    # refined like the events, so both scores are on the same scale (the coarse stage scores a
    # frequency max-filtered spectrogram and reads higher)
    null = detect_events(reversed_spectrogram(sp), f_lo, f_hi, c=c, harmonics=harmonics, top=3,
                         window_s=window_s, workers=workers)
    chance = max((e.score for e in null), default=0.0)
    credible = [e for e in evs if e.score > chance]
    risk = "high" if len(credible) >= 2 else "medium" if credible else "low"
    return {"events": evs, "chance": float(chance), "credible": credible, "risk": risk}


def _curve_mask(times, freqs, ev: EventFit, c: float, width: float, fmax: float, pad_s: float, df: float,
                rel=None, search: float = 0.12):
    """(F, T) boolean mask of the bins along every harmonic of the event's curve.

    The fitted ideal curve can be several percent off the real tone (a turning pass, or a
    window edge on the approach), so with rel (|Z| over its per-bin median, (F, T)) the mask
    also follows the strongest bin within +-search of each harmonic in every frame. Masking a
    noise peak where there is no tone is harmless: noise is replaced with noise."""
    m = np.zeros((len(freqs), len(times)), bool)
    inside = (times >= ev.t_start - pad_s) & (times <= ev.t_end + pad_s)
    if not inside.any():
        return m
    tt = times[inside]
    base = [ideal_curve(tt, ev.f0, ev.v, ev.d, ev.t_cpa_emit, c=c)]
    if ev.line:
        ln = np.asarray(ev.line, float)
        base.append(np.interp(tt, ln[:, 0], ln[:, 1], left=np.nan, right=np.nan))
    # before and after the measured span the tone keeps its asymptotes
    for b in base:
        b[np.isnan(b) & (tt < ev.t_cpa)] = ev.f_pre
        b[np.isnan(b) & (tt >= ev.t_cpa)] = ev.f_post
    sub = np.zeros((len(freqs), len(tt)), bool)
    r = None if rel is None else rel[:, inside]
    for b in base:
        h = 1
        while h * ev.f0 * 0.9 <= fmax:
            fh = h * b
            sub |= np.abs(freqs[:, None] - fh[None, :]) <= np.maximum(width * fh, 2.0 * df)[None, :]
            if r is not None:
                near = np.abs(freqs[:, None] - fh[None, :]) <= search * fh[None, :]
                pk = freqs[np.argmax(np.where(near, r, -np.inf), axis=0)]
                sub |= np.abs(freqs[:, None] - pk[None, :]) <= np.maximum(width * pk, 2.0 * df)[None, :]
            h += 1
    m[:, inside] = sub
    return m


def scrub_audio(x: np.ndarray, sr: int, events: list[EventFit], mode: str = "fill", width: float = 0.04,
                pad_s: float = 30.0, fmax: float = 1500.0, c: float | None = None, seed: int = 0,
                nperseg: int | None = None) -> np.ndarray:
    """Audio with each event's Doppler tone removed (see the module docstring for modes)."""
    if mode not in SCRUB_MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of {SCRUB_MODES}")
    c = speed_of_sound() if c is None else c
    x = np.asarray(x, np.float32).copy()
    spans = sorted((max(0.0, e.t_start - pad_s), min(len(x) / sr, e.t_end + pad_s)) for e in events)
    if mode in ("mute", "cut"):
        keep = np.ones(len(x), bool)
        for a, b in spans:
            keep[int(a * sr):int(b * sr)] = False
        if mode == "cut":
            return x[keep]
        from scipy.ndimage import uniform_filter1d
        fade = int(0.2 * sr)
        g = keep.astype(np.float32)
        if fade > 1:
            g = uniform_filter1d(g, fade, mode="nearest")
        return x * g * keep          # fades out just before each span, exactly zero inside it
    rng = np.random.default_rng(seed)
    nper = nperseg or int(2 ** np.round(np.log2(sr * 0.5)))
    hop = nper // 4
    margin = 2 * nper / sr
    for ev in events:
        a = max(0, int((ev.t_start - pad_s - margin) * sr)); b = min(len(x), int((ev.t_end + pad_s + margin) * sr))
        if b - a < 2 * nper:
            continue
        f, t, Z = ss.stft(x[a:b], sr, nperseg=nper, noverlap=nper - hop, window="hann")
        mag = np.abs(Z)
        rel = mag / np.maximum(np.median(mag, axis=1, keepdims=True), 1e-20)
        m = _curve_mask(t + a / sr, f, ev, c, width, fmax, pad_s, f[1] - f[0], rel=rel)
        if not m.any():
            continue
        floor = np.where(m, np.nan, mag)
        with np.errstate(all="ignore"):
            med = np.nanmedian(floor, axis=1)
        med = np.where(np.isfinite(med), med, np.nanmedian(mag, axis=1))
        # the fill is the STFT of real white noise, scaled per bin to that bin's median level:
        # frames with independent random phases would partly cancel when overlap-added and
        # leave the curve's path darker than its surroundings, a negative copy of the curve
        _, _, Zn = ss.stft(rng.standard_normal(b - a), sr, nperseg=nper, noverlap=nper - hop, window="hann")
        Zn = Zn[:, :Z.shape[1]] / np.median(np.abs(Zn), axis=1, keepdims=True)
        Z = np.where(m, med[:, None] * Zn, Z)
        _, y = ss.istft(Z, sr, nperseg=nper, noverlap=nper - hop, window="hann")
        x[a:b] = y[: b - a].astype(np.float32)
    return x


def write_wav(path, x: np.ndarray, sr: int):
    import soundfile as sf
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    sf.write(str(path), x / peak * 0.99 if peak > 1.0 else x, sr, subtype="PCM_16")
