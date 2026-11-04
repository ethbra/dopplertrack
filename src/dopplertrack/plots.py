"""Diagnostic figures: event fits, modelled curves over the spectrogram, score maps."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .audio import Spectrogram
from .doppler import ideal_curve, speed_of_sound
from .score import modelled_curve
from .tracks import EventFit, local_whiten


def _show(ax, sp: Spectrogram, vmax=10):
    ax.imshow(sp.S, origin="lower", aspect="auto", extent=[sp.t[0], sp.t[-1], sp.f[0], sp.f[-1]],
              vmin=0, vmax=vmax, cmap="magma")
    ax.set_ylabel("Hz")


def plot_event_fit(sp: Spectrogram, fit: EventFit, harmonics=1, pad=30, fmax=None, ax=None):
    ax = ax or plt.gca()
    w = local_whiten(sp.crop(fit.t_start - pad, fit.t_end + pad, fit.f_post * 0.7, fmax or fit.f_pre * 1.3 * harmonics))
    _show(ax, w)
    tt = np.linspace(w.t[0], w.t[-1], 600)
    for h in range(1, harmonics + 1):
        ax.plot(tt, h * ideal_curve(tt, fit.f0, fit.v, fit.d, fit.t_cpa_emit), "c-", lw=0.8)
    ax.set_title(f"ideal fit: f0={fit.f0:.1f} Hz  v={fit.v:.0f} m/s  d={fit.d:.0f} m  t_cpa={fit.t_cpa:.1f} s")
    return ax


def plot_modelled(sp: Spectrogram, fit: EventFit, seg, obs_enu, t_audio_offset, f0_scale=1.0, harmonics=1,
                  pad=30, ax=None, c=None, label=None):
    """Overlay the curve a candidate observer would hear from `seg`. t_audio_offset maps
    unix time to recording time (t_audio = t_unix + offset)."""
    c = speed_of_sound() if c is None else c
    ax = ax or plt.gca()
    w = local_whiten(sp.crop(fit.t_start - pad, fit.t_end + pad, fit.f_post * 0.7, fit.f_pre * 1.3 * harmonics))
    _show(ax, w)
    to, ratio = modelled_curve(seg, obs_enu, c)
    ta = to + t_audio_offset
    for h in range(1, harmonics + 1):
        ax.plot(ta, h * fit.f0 * f0_scale * ratio, "c-", lw=0.9)
    ax.set_xlim(w.t[0], w.t[-1]); ax.set_ylim(w.f[0], w.f[-1])
    if label:
        ax.set_title(label)
    return ax


def plot_score_map(res, truth=None, ax=None, top=None, s=6):
    """Scatter of candidate scores (best per grid point)."""
    ax = ax or plt.gca()
    best = res.sort_values("score", ascending=False).drop_duplicates("gidx")
    if top:
        best = best.head(top)
    sc = ax.scatter(best.lon, best.lat, c=best.score, s=s, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="combined score")
    if truth is not None:
        ax.plot(truth[1], truth[0], "r+", ms=14, mew=2, label="truth"); ax.legend()
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect(1 / np.cos(np.radians(best.lat.mean())))
    return ax


def plot_pair_scores(res, top=30, ax=None):
    """Best score of each flight pair (or flight), highest first: the video's view of the
    survivors grouped by pair, where the right pair stands clear of the rest."""
    ax = ax or plt.gca()
    key = ["icao1", "icao2"] if "icao1" in res.columns else ["icao"]
    best = res.sort_values("score", ascending=False).drop_duplicates(key).head(top)
    labels = ["/".join(str(best.iloc[i][k]) for k in key) for i in range(len(best))]
    ax.bar(range(len(best)), best.score.to_numpy(), color=["#d62728"] + ["#4477aa"] * (len(best) - 1))
    ax.set_xticks(range(len(best)), labels, rotation=90, fontsize=7)
    ax.set_ylabel("best score"); ax.set_title(f"best score per flight {'pair' if len(key) == 2 else ''} (top {len(best)})")
    if len(best) > 1:
        m = (best.score.iloc[0] - best.score.iloc[1]) / abs(best.score.iloc[1]) * 100
        ax.text(0.99, 0.95, f"margin {m:.1f} %", transform=ax.transAxes, ha="right", va="top")
    return ax
