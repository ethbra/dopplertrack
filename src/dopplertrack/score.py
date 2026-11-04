"""Score candidate observers by modelling the exact Doppler curve they would hear."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import Spectrogram
from .doppler import curve_from_trajectory, speed_of_sound
from .search import Segment
from .tracks import EventFit, local_whiten, ridge

RIDGE_MIN_DB = 3.0          # ridge frames dimmer than this (whitened dB) carry no evidence
RIDGE_MIN_UNIQUE_DB = 3.0   # ... and so do frames where another path through the window is nearly as bright
RIDGE_EXCLUDE_BINS = 3      # half-width of the band blanked around the ridge to find that other path
RIDGE_SIGMA = 0.015         # frequency tolerance of the ridge score, as a fraction of the ridge frequency
SCORE_MODES = ("brightness", "ridge", "line")
LINE_MAX_GAP_S = 30.0       # mode "line" does not interpolate the traced line across longer gaps


@dataclass
class EventWindow:
    """Whitened spectrogram crop for one event plus its measured parameters.

    mode="brightness" scores a modelled curve by the brightness under it. mode="ridge" first
    extracts the ridge of the measured curve (a Viterbi path over the fit window; harmonics
    are multiples of it) and scores a modelled curve by how closely it follows that ridge:
    each reliable frame contributes exp(-0.5 (df / (RIDGE_SIGMA f))^2), so the score is the
    fraction of the window where the model runs along the ridge, in [0, 1]. A frame is
    reliable when the ridge is at least RIDGE_MIN_DB bright and at least RIDGE_MIN_UNIQUE_DB
    brighter than the best path that avoids it: a bird burst or a second tone offers several
    equally good paths and says nothing about where the aircraft's tone is. That is what a
    faint curve needs: brightness cannot separate candidates on a tone a few dB above the
    noise, but the ridge's position can.

    mode="line" is the same kernel against the line stored in the event (EventFit.line, the
    fundamental, from `dopplertrack trace-line`): a hand-traced line, as in the video, or an
    automatic ridge a person has checked and trimmed. Every frame the line covers counts."""
    fit: EventFit
    sp: Spectrogram
    harmonics: int = 1
    mode: str = "brightness"
    ridge_sigma: float = RIDGE_SIGMA
    ridge_data: tuple | None = field(default=None, repr=False, compare=False)   # (t, f[h], w[h])

    @classmethod
    def from_fit(cls, sp: Spectrogram, fit: EventFit, harmonics: int = 1, pad_s: float = 40.0,
                 mode: str = "brightness", ridge_min_db: float = RIDGE_MIN_DB,
                 ridge_min_unique_db: float = RIDGE_MIN_UNIQUE_DB, ridge_sigma: float = RIDGE_SIGMA):
        if mode not in SCORE_MODES:
            raise ValueError(f"unknown score mode {mode!r}; use one of {SCORE_MODES}")
        w = sp.crop(fit.t_start - pad_s, fit.t_end + pad_s, max(fit.f_post * 0.8, sp.f[0]),
                    min(fit.f_pre * 1.25 * harmonics, sp.f[-1]))
        ew = cls(fit, local_whiten(w), harmonics, mode, ridge_sigma)
        if mode == "ridge":
            ew.extract_ridge(ridge_min_db, ridge_min_unique_db)
        elif mode == "line":
            ew.use_line()
        return ew

    def use_line(self):
        """Ridge data from the event's stored line: frames of the fit window, the line's
        frequency where it is defined (linear between its points, no extrapolation)."""
        if not self.fit.line:
            raise ValueError(f"event at {self.fit.t_cpa:.1f} s has no traced line; run `dopplertrack trace-line` first")
        ln = np.asarray(self.fit.line, float)
        ln = ln[np.argsort(ln[:, 0])]
        ts = np.arange(self.fit.t_start, self.fit.t_end + 1e-9, self.sp.dt)
        fr = np.interp(ts, ln[:, 0], ln[:, 1], left=np.nan, right=np.nan)
        # a hand-traced line is often a few clicks; interpolate between them, but not across a
        # gap of more than LINE_MAX_GAP_S (a stretch the tracer left out on purpose)
        gap = np.diff(ln[:, 0]) > LINE_MAX_GAP_S
        if gap.any():
            k = np.searchsorted(ln[:, 0], ts) - 1
            inside = (k >= 0) & (k < len(gap))
            fr[inside & gap[np.clip(k, 0, len(gap) - 1)]] = np.nan
        ok = np.isfinite(fr)
        fs = np.array([h * fr for h in range(1, self.harmonics + 1)])
        ws = np.array([ok.astype(float)] * self.harmonics)
        self.ridge_data = (ts, fs, ws)

    def extract_ridge(self, min_db: float = RIDGE_MIN_DB, min_unique_db: float = RIDGE_MIN_UNIQUE_DB):
        """Ridge of the fundamental over the fit window, within the band the fit's asymptotes
        bound (with margin). Harmonic h is h times that ridge (the harmonics of one pass are
        one event; tracking them separately lets a harmonic band latch onto the brighter
        fundamental). A frame is reliable when the ridge is at least min_db bright and at
        least min_unique_db brighter than the best path through the window that stays
        RIDGE_EXCLUDE_BINS bins away from it; harmonic h additionally needs h * ridge to be
        at least min_db bright."""
        w = self.sp.crop(self.fit.t_start, self.fit.t_end, max(self.fit.f_post * 0.8, self.sp.f[0]),
                         min(self.fit.f_pre * 1.25, self.sp.f[-1]))
        fr = ridge(w)
        j = np.rint((fr - w.f[0]) / w.df).astype(int)
        S2 = np.where(np.abs(np.arange(len(w.f))[:, None] - j[None, :]) <= RIDGE_EXCLUDE_BINS, 0.0, w.S)
        fr2 = ridge(Spectrogram(w.t, w.f, S2.astype(np.float32)))
        unique = (w.sample(w.t, fr) - w.sample(w.t, fr2)) >= min_unique_db
        fs = np.array([h * fr for h in range(1, self.harmonics + 1)])
        ws = np.array([(unique & (self.sp.sample(w.t, h * fr) >= min_db)).astype(float)
                       for h in range(1, self.harmonics + 1)])
        self.ridge_data = (w.t, fs, ws)

    def _sample(self, h: int, T, F):
        """Per-sample evidence for harmonic h at (time, frequency) pairs."""
        if self.mode == "brightness":
            return self.sp.sample(T, F)
        ts, fs, ws = self.ridge_data      # modes "ridge" and "line"
        i = np.rint((T - ts[0]) / (ts[1] - ts[0])).astype(int)
        inside = (i >= 0) & (i < len(ts))
        ic = np.clip(i, 0, len(ts) - 1)
        fr = np.interp(T, ts, fs[h - 1], left=np.nan, right=np.nan)
        wr = np.where(inside, ws[h - 1][ic], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (F - fr) / (self.ridge_sigma * fr)
            v = wr * np.exp(-0.5 * z * z)
        return np.where(np.isfinite(v), v, 0.0)

    def curve_score(self, t_audio, ratio, f0_scale=1.0) -> float:
        """Brightness under the curve averaged over the event's fit window (uncovered parts
        count as zero); same convention as sweep_scores, for a single placement."""
        return float(self.sweep_scores(t_audio, ratio, [0.0], [f0_scale])[0, 0])

    def sweep_scores(self, t_audio, ratio, shifts, f0_scales, n=160) -> np.ndarray:
        """Scores for every (shift, f0_scale) combination in one vectorised pass.
        The curve is resampled onto n points of the fit window for each shift.

        The score is the per-sample evidence (brightness, or the ridge kernel in mode="ridge")
        summed along the curve divided by n, i.e. the mean over the *whole* fit window, with
        the parts of the window the modelled curve does not cover counting as zero. A flight
        whose trajectory only explains half the measured sweep (it took off, landed or left
        the data mid-pass) therefore earns half the score, instead of being judged only where
        it happens to fit."""
        tq = np.linspace(self.fit.t_start, self.fit.t_end, n)
        rel = tq[None, :] - np.asarray(shifts)[:, None]                       # (S, n) un-shifted audio time
        ok = np.isfinite(ratio)
        rat = np.interp(rel, t_audio[ok], ratio[ok], left=np.nan, right=np.nan)  # (S, n)
        valid = np.isfinite(rat).sum(1) >= 10                                  # (S,)
        total = np.zeros((len(shifts), len(f0_scales)))
        for h in range(1, self.harmonics + 1):
            F = h * self.fit.f0 * np.asarray(f0_scales)[None, :, None] * rat[:, None, :]   # (S, K, n)
            T = np.broadcast_to(tq[None, None, :], F.shape)
            v = self._sample(h, T.ravel(), np.nan_to_num(F.ravel(), nan=-1.0)).reshape(F.shape)
            v = np.where(np.isfinite(F), v, 0.0)
            total += v.sum(axis=2) / n
        total[~valid] = 0.0
        return total / self.harmonics


def modelled_curve(seg: Segment, obs_enu, c, wind=None):
    """(t_obs unix, ratio f/f0) the observer at obs_enu hears from this segment."""
    t_obs, ratio, r = curve_from_trajectory(seg.t, seg.enu, obs_enu, f0=1.0, c=c, wind=wind)
    return t_obs, ratio


def locked_score(ev1: EventWindow, ev2: EventWindow, seg1: Segment, seg2: Segment, obs_enu,
                 t_heard1: float, c: float | None = None,
                 shifts=np.arange(-8, 8.01, 0.5), f0_scales=np.arange(0.95, 1.0501, 0.005), wind=None):
    """Two-event locked scoring.

    Event 1's placement is swept over small time shifts and f0 scales (both are
    measurements). Event 2 is placed with the *same* time shift, so its timing
    relative to event 1 is fixed purely by flight geometry.
    Returns (total, s1, s2, best_shift).
    """
    c = speed_of_sound() if c is None else c
    # map unix time -> audio time so that the modelled heard-CPA of event 1 lands on the measured one
    t0_map = ev1.fit.t_cpa - t_heard1
    to1, r1 = modelled_curve(seg1, obs_enu, c, wind)
    to2, r2 = modelled_curve(seg2, obs_enu, c, wind)
    sc1 = ev1.sweep_scores(to1 + t0_map, r1, shifts, f0_scales)
    i, j = np.unravel_index(np.argmax(sc1), sc1.shape)
    s1, sh = float(sc1[i, j]), float(shifts[i])
    s2 = float(ev2.sweep_scores(to2 + t0_map, r2, [sh], f0_scales).max())
    return s1 + s2, s1, s2, sh


def single_score(ev: EventWindow, seg: Segment, obs_enu, t_heard: float, c=None,
                 shifts=np.arange(-8, 8.01, 0.5), f0_scales=np.arange(0.95, 1.0501, 0.005), wind=None):
    c = speed_of_sound() if c is None else c
    to, r = modelled_curve(seg, obs_enu, c, wind)
    t0_map = ev.fit.t_cpa - t_heard
    return float(ev.sweep_scores(to + t0_map, r, shifts, f0_scales).max())


def auto_line(sp: Spectrogram, fit: EventFit, min_db: float = RIDGE_MIN_DB,
              min_unique_db: float = RIDGE_MIN_UNIQUE_DB) -> list[list[float]]:
    """The reliable frames of the automatic ridge of an event's fundamental, as a line
    [[t, f], ...] for EventFit.line (see EventWindow.extract_ridge for "reliable")."""
    ts, fs, ws = EventWindow.from_fit(sp, fit, 1, mode="ridge", ridge_min_db=min_db,
                                      ridge_min_unique_db=min_unique_db).ridge_data
    keep = ws[0] > 0
    return [[round(float(t), 3), round(float(f), 3)] for t, f in zip(ts[keep], fs[0][keep])]
