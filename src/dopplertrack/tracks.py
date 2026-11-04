"""Extract a Doppler event from a spectrogram window and fit the ideal model."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.optimize import minimize

from .audio import Spectrogram
from .doppler import ideal_curve, speed_of_sound


@dataclass
class EventFit:
    t_cpa: float        # heard CPA time in the recording, s (observation time of the f0 crossing)
    t_cpa_emit: float   # emission-time CPA (t_cpa - d/c)
    f0: float           # rest fundamental, Hz
    v: float            # speed, m/s
    d: float            # CPA slant distance, m
    t_start: float
    t_end: float
    score: float        # mean whitened dB along the fitted curve
    f_pre: float
    f_post: float
    # optional traced line of the fundamental, [[t, f], ...] (recording s, Hz): written by
    # `dopplertrack trace-line` from the automatic ridge or a hand-traced CSV, scored by
    # score mode "line" (the video scores its faint curve by distance from such a line)
    line: list | None = field(default=None, repr=False)

    def to_dict(self):
        d = asdict(self)
        if d["line"] is None:
            del d["line"]
        return d


def local_whiten(sp: Spectrogram, floor: float = 0.0) -> Spectrogram:
    """Remove tones that are stationary over this window (per-bin median), clip at floor."""
    S = sp.S - np.median(sp.S, axis=1, keepdims=True)
    return Spectrogram(sp.t, sp.f, np.maximum(S, floor).astype(np.float32))


def ridge(sp: Spectrogram, max_step_bins: int = 3, smooth: float = 1.5) -> np.ndarray:
    """Viterbi ridge through S (F,T): maximise sum(S) - smooth*|Δbin|. Returns freq per frame."""
    S = sp.S
    F, T = S.shape
    cost = np.full((T, F), -np.inf)
    back = np.zeros((T, F), np.int16)
    cost[0] = S[:, 0]
    offs = np.arange(-max_step_bins, max_step_bins + 1)
    pen = smooth * np.abs(offs)
    for t in range(1, T):
        prev = cost[t - 1]
        cand = np.full((len(offs), F), -np.inf)
        for k, o in enumerate(offs):
            if o >= 0:
                cand[k, o:] = prev[: F - o] - pen[k]
            else:
                cand[k, :o] = prev[-o:] - pen[k]
        kbest = np.argmax(cand, axis=0)
        cost[t] = cand[kbest, np.arange(F)] + S[:, t]
        back[t] = offs[kbest]
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(cost[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = path[t] - back[t, path[t]]
    return sp.f[path]


CONTRAST_DF = 0.04     # sideband offset as a fraction of frequency
SWEEP_FLOOR = 0.15     # weight given to the flat tails of a curve, relative to its steepest part


def sweep_weights(ratio, t, axis=-1):
    """Per-sample weights proportional to |d ratio / dt| (the sweeping part of a Doppler
    curve carries the information; flat asymptotes also fit any steady tone), normalised to
    a peak of 1 per curve, plus SWEEP_FLOOR."""
    g = np.abs(np.gradient(ratio, t, axis=axis))
    g = np.nan_to_num(g, nan=0.0)
    peak = np.max(g, axis=axis, keepdims=True)
    return g / np.maximum(peak, 1e-12) + SWEEP_FLOOR


def curve_contrast(sp: Spectrogram, t, f):
    """Per-sample ridge contrast: brightness on the curve minus the mean of two sidebands.
    A broadband burst is as bright off the curve as on it and so scores ~0. Off-grid -> 0."""
    return sp.sample(t, f) - 0.5 * (sp.sample(t, f * (1 + CONTRAST_DF)) + sp.sample(t, f * (1 - CONTRAST_DF)))


def _shape(v, d, c, span=400.0, n=161):
    """Observation-time offsets and frequency ratios for a straight pass, relative to
    emission-time CPA (tau=0)."""
    tau = np.linspace(-span, span, n)
    r = np.sqrt(d * d + (v * tau) ** 2)
    rr = v * v * tau / r
    return tau + r / c, c / (c + rr)


def grid_search(sp: Spectrogram, t_start, t_end, f_lo, f_hi, c, harmonics=1,
                v_grid=None, d_grid=None, f0_step=1.0, t0_step=1.0, n_curve=120,
                t0_range=None, f0_range=None, per_t0=False, fat_bins=5, objective="contrast"):
    """Coarse maximisation of brightness under ideal curves. Returns (score, (f0, v, d, t0_emit)).

    objective="contrast" scores the sweep-weighted ridge contrast (sweep_weights,
    curve_contrast) on a spectrogram max-filtered over `fat_bins` frequency bins, so that a
    grid curve a hertz or two off a narrow ridge still scores; the Nelder-Mead stage in
    fit_event refines on the unfiltered spectrogram. objective="brightness" scores the plain
    mean brightness under the curve, unweighted and unfiltered.

    The curve is scored on [t_start, t_end]; t0_range / f0_range (optional) restrict the
    searched emission-CPA times and fundamentals, which otherwise span the window / band.
    per_t0=True returns instead (t0s, best_score[T0], best_params[T0, 3]) so a caller can
    pick several local maxima in time."""
    from scipy.ndimage import maximum_filter1d
    if objective not in ("contrast", "brightness"):
        raise ValueError(f"unknown objective {objective!r}")
    contrast = objective == "contrast"
    if contrast and fat_bins and fat_bins > 1:
        sp = Spectrogram(sp.t, sp.f, maximum_filter1d(sp.S, fat_bins, axis=0))
    v_grid = np.arange(30, 170, 10.0) if v_grid is None else v_grid
    d_grid = np.geomspace(300, 9000, 14) if d_grid is None else d_grid
    f0s = np.arange(*(f0_range or (f_lo * 1.15, f_hi * 0.85)), f0_step)
    t0s = np.arange(*(t0_range or (t_start, t_end)), t0_step)
    tq = np.linspace(t_start, t_end, n_curve)
    bs = np.full(len(t0s), -np.inf); bp = np.zeros((len(t0s), 3))
    for v in v_grid:
        for d in d_grid:
            dto, ratio = _shape(v, d, c)
            rel = tq[None, :] - t0s[:, None]                         # (T0, N)
            rat = np.interp(rel, dto, ratio, left=np.nan, right=np.nan)  # (T0, N)
            if contrast:
                w = sweep_weights(rat, tq, axis=1)                    # (T0, N)
                w = np.where(np.isnan(rat), 0.0, w)
            else:                                                     # samples past the curve's span count as zero
                w = np.ones_like(rat)
            wsum = np.maximum(w.sum(axis=1), 1e-12)
            total = np.zeros((len(t0s), len(f0s)))
            for h in range(1, harmonics + 1):
                F = h * f0s[None, None, :] * rat[:, :, None]           # (T0, N, F0)
                T = np.broadcast_to(tq[None, :, None], F.shape)
                vals = curve_contrast(sp, T.ravel(), F.ravel()) if contrast else sp.sample(T.ravel(), F.ravel())
                total += np.einsum("tn,tnf->tf", w, np.nan_to_num(vals.reshape(F.shape))) / wsum[:, None]
            total /= harmonics
            j = np.argmax(total, axis=1); sc = total[np.arange(len(t0s)), j]
            upd = sc > bs
            bs[upd] = sc[upd]; bp[upd, 0] = f0s[j[upd]]; bp[upd, 1] = v; bp[upd, 2] = d
    if per_t0:
        return t0s, bs, bp
    i = int(np.argmax(bs))
    return float(bs[i]), (float(bp[i, 0]), float(bp[i, 1]), float(bp[i, 2]), float(t0s[i]))


def fit_event(sp: Spectrogram, t_start: float, t_end: float, f_lo: float, f_hi: float,
              c: float | None = None, harmonics: int = 1, whiten: bool = True,
              init: tuple[float, float, float, float] | None = None,
              v_bounds: tuple[float, float] = (10.0, 300.0), d_bounds: tuple[float, float] = (100.0, 30000.0),
              objective: str = "contrast") -> EventFit:
    """Fit (f0, v, d, t_cpa) to the brightest Doppler-shaped curve in a window.

    init=(f0, v, d, t0_emit) narrows the grid search to around that solution.

    objective="contrast" (what detect uses) maximises the sweep-weighted ridge contrast, which
    ignores broadband bursts and steady tones. objective="brightness" maximises the plain mean
    brightness under the curve, so the flat asymptotes count as much as the sweep; fits made
    before f890572 used it, including the committed docs/results event fits, and `fit` uses it
    by default for a window chosen by hand."""
    c = speed_of_sound() if c is None else c
    pad = 0.3 * (f_hi - f_lo)
    win = sp.crop(t_start - 30, t_end + 30, max(f_lo - pad, sp.f[0]), min(f_hi * harmonics + pad, sp.f[-1]))
    if whiten:
        win = local_whiten(win)
    if init is None:
        score0, (f0, v, d, t0) = grid_search(win, t_start, t_end, f_lo, f_hi, c, harmonics, objective=objective)
    else:   # full v/d grids, but only near the given fundamental and CPA time
        f0, v, d, t0 = init
        score0, (f0, v, d, t0) = grid_search(win, t_start, t_end, f_lo, f_hi, c, harmonics,
                                             t0_range=(t0 - 30, t0 + 30), f0_range=(max(f0 - 12, 1), f0 + 12),
                                             objective=objective)
    tt = np.linspace(t_start, t_end, 400)

    def neg(p):
        f0_, v_, d_, t0_ = p
        if not (v_bounds[0] < v_ < v_bounds[1] and d_bounds[0] < d_ < d_bounds[1]):
            return 1e3
        s = 0.0
        base = ideal_curve(tt, f0_, v_, d_, t0_, c=c)
        if objective == "brightness":
            return -sum(float(np.mean(win.sample(tt, h * base))) for h in range(1, harmonics + 1)) / harmonics
        w = sweep_weights(base / f0_, tt)
        for h in range(1, harmonics + 1):
            s += float(np.sum(w * curve_contrast(win, tt, h * base)) / np.sum(w))
        return -s / harmonics

    best = minimize(neg, [f0, v, d, t0], method="Nelder-Mead",
                    options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 4000,
                             "initial_simplex": np.array([[f0, v, d, t0], [f0 + 2, v, d, t0], [f0, v + 10, d, t0],
                                                          [f0, v, d * 1.3, t0], [f0, v, d, t0 + 2]])})
    f0, v, d, t0 = best.x
    return EventFit(t_cpa=float(t0 + d / c), t_cpa_emit=float(t0), f0=float(f0), v=float(v), d=float(d),
                    t_start=t_start, t_end=t_end, score=float(-best.fun),
                    f_pre=float(f0 * c / (c - v)), f_post=float(f0 * c / (c + v)))


def _scan_window(args):
    """Up to k local maxima (separated by min_sep seconds) of the coarse search in one window."""
    win, t, t1, f_lo, f_hi, c, harmonics, v_grid, d_grid, f0_step, t0_step, n_curve, k, min_sep = args
    t0s, bs, bp = grid_search(win, t, t1, f_lo, f_hi, c, harmonics, v_grid, d_grid,
                              f0_step=f0_step, t0_step=t0_step, n_curve=n_curve, per_t0=True)
    out = []
    order = np.argsort(-bs)
    for i in order:
        if all(abs(t0s[i] - o[4]) >= min_sep for o in out):
            out.append((float(bs[i]), float(bp[i, 0]), float(bp[i, 1]), float(bp[i, 2]), float(t0s[i])))
        if len(out) >= k:
            break
    return out


def detect_events(sp: Spectrogram, f_lo: float, f_hi: float, c: float | None = None, harmonics: int = 1,
                  window_s: float = 240.0, step_s: float = 120.0, v_grid=None, d_grid=None,
                  f0_step: float = 2.0, t0_step: float = 2.0, n_curve: int = 80, top: int = 20,
                  per_window: int = 3, min_sep_s: float = 60.0, refine: bool = True,
                  workers: int | None = None, progress=None) -> list[EventFit]:
    """Scan a whole recording for Doppler-shaped curves whose fundamental lies in [f_lo, f_hi].

    Each window gets the coarse grid search used by fit_event (windows run in parallel
    processes) and contributes up to `per_window` local maxima in time. Those are merged
    (one event per min_sep_s), the best 2*top are refined with a local grid search plus
    Nelder-Mead, and the best `top` after refinement are returned, sorted by score
    (whitened brightness under the curve, dB)."""
    from concurrent.futures import ProcessPoolExecutor
    c = speed_of_sound() if c is None else c
    v_grid = np.arange(40, 170, 13.0) if v_grid is None else v_grid
    d_grid = np.geomspace(400, 9000, 10) if d_grid is None else d_grid
    pad = 0.3 * (f_hi - f_lo)
    starts = np.arange(sp.t[0], max(sp.t[-1] - window_s, sp.t[0]) + 1e-9, step_s)

    def jobs():
        for t in starts:
            t1 = min(t + window_s, sp.t[-1])
            win = local_whiten(sp.crop(t - 30, t1 + 30, max(f_lo - pad, sp.f[0]), min(f_hi * harmonics + pad, sp.f[-1])))
            yield (win, t, t1, f_lo, f_hi, c, harmonics, v_grid, d_grid, f0_step, t0_step, n_curve, per_window, min_sep_s)

    cands = []
    if workers == 1:
        it = map(_scan_window, jobs())
    else:
        ex = ProcessPoolExecutor(workers)
        it = ex.map(_scan_window, jobs(), chunksize=2)
    for k, r in enumerate(it):
        cands.extend(r)
        if progress:
            progress(k + 1, len(starts))
    if workers != 1:
        ex.shutdown()
    cands.sort(reverse=True)
    keep = []
    for cnd in cands:
        if all(abs(cnd[4] - k[4]) >= min_sep_s for k in keep):
            keep.append(cnd)
        if len(keep) >= (2 * top if refine else top):
            break
    out = []
    for score, f0, v, d, t0 in keep:
        if refine:
            ef = fit_event(sp, max(t0 - 120, sp.t[0]), min(t0 + 120, sp.t[-1]), f_lo, f_hi, c=c,
                           harmonics=harmonics, init=(f0, v, d, t0), v_bounds=(35.0, 200.0), d_bounds=(200.0, 15000.0))
        else:
            ef = EventFit(t_cpa=t0 + d / c, t_cpa_emit=t0, f0=f0, v=v, d=d, t_start=t0 - 120, t_end=t0 + 120,
                          score=score, f_pre=f0 * c / (c - v), f_post=f0 * c / (c + v))
        out.append(ef)
    out.sort(key=lambda e: -e.score)
    final = []
    for e in out:          # refined fits can converge onto the same event; keep the best
        if all(abs(e.t_cpa - f.t_cpa) >= min_sep_s for f in final):
            final.append(e)
        if len(final) >= top:
            break
    return final
