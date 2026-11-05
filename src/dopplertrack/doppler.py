"""Doppler curve physics.

All functions use *emission-time* kinematics: the frequency heard at
observation time t_obs was produced at emission time t_e where
t_obs = t_e + r(t_e)/c, and the shift is set by the range rate at t_e.
"""
from __future__ import annotations

import numpy as np


def speed_of_sound(temp_c: float = 15.0) -> float:
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


def observed_frequency(f0, range_rate, c):
    """Frequency heard for a source of rest frequency f0 with range rate dr/dt
    (positive = receding)."""
    return f0 * c / (c + range_rate)


def ideal_curve(t_obs, f0, v, d_cpa, t_cpa, c=None, span=600.0, n=4001):
    """Frequency heard at times t_obs for a straight, constant-speed pass.

    v: speed (m/s); d_cpa: slant distance at closest approach (m);
    t_cpa: *emission* time of closest approach.
    """
    c = speed_of_sound() if c is None else c
    te = t_cpa + np.linspace(-span, span, n)
    r = np.sqrt(d_cpa**2 + (v * (te - t_cpa)) ** 2)
    rr = v * v * (te - t_cpa) / r
    f = observed_frequency(f0, rr, c)
    tobs = te + r / c
    return np.interp(np.asarray(t_obs, float), tobs, f, left=np.nan, right=np.nan)


def ideal_curve_params_from_endpoints(f_pre, f_post, c=None):
    """Recover (f0, v) from the approach and recede asymptote frequencies."""
    c = speed_of_sound() if c is None else c
    # f_pre = f0 c/(c-v), f_post = f0 c/(c+v)
    v = c * (f_pre - f_post) / (f_pre + f_post)
    f0 = f_pre * (c - v) / c
    return f0, v


def curve_from_trajectory(t_e, pos_enu, obs_enu, f0=1.0, c=None, model="emission"):
    """Doppler curve a stationary observer hears from a sampled trajectory.

    t_e: (N,) emission times (s); pos_enu: (N,3) source positions (m);
    obs_enu: (3,) observer position (m). Returns (t_obs, f, r) with f for
    rest frequency f0 (use f0=1 to get the ratio).

    model="reception" is the mistake the video warns about, kept for comparison: the heard
    time is still t_e + r/c, but the shift uses the range rate at that heard time instead of
    at t_e. Plotted against the heard time, that is the emission-time curve with no delay.
    """
    if model not in ("emission", "reception"):
        raise ValueError(f"unknown model {model!r}")
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float)
    d = np.asarray(pos_enu, float) - np.asarray(obs_enu, float)
    r = np.linalg.norm(d, axis=-1)
    rr = np.gradient(r, t_e)
    t_obs = t_e + r / c
    if model == "reception":
        rr = np.interp(t_obs, t_e, rr)
    f = observed_frequency(f0, rr, c)
    return t_obs, f, r


def emission_time(t_r, t_e, pos_enu, obs_enu, c=None) -> float:
    """Solve t_r = t_e + r(t_e)/c for the emission time of the sound heard at t_r. The heard
    time increases with t_e for a subsonic source, so the root is unique: bracket it on the
    samples, refine with Brent on the linearly interpolated trajectory."""
    from scipy.optimize import brentq
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float); pos = np.asarray(pos_enu, float); obs = np.asarray(obs_enu, float)
    t_obs = t_e + np.linalg.norm(pos - obs, axis=1) / c
    if not (t_obs[0] <= t_r <= t_obs[-1]):
        raise ValueError(f"t_r={t_r} is outside the heard span [{t_obs[0]:.1f}, {t_obs[-1]:.1f}]")

    def miss(te):
        p = np.array([np.interp(te, t_e, pos[:, k]) for k in range(3)])
        return te + float(np.linalg.norm(p - obs)) / c - t_r
    i = int(np.clip(np.searchsorted(t_obs, t_r), 1, len(t_e) - 1))
    return float(brentq(miss, t_e[i - 1], t_e[i], xtol=1e-9))


def explain(t_r, t_e, pos_enu, obs_enu, f0, c=None) -> dict:
    """The video's 'Building the Doppler Curve, Step by Step' panel for one heard time: the
    emission time, the aircraft's position and velocity then, slant range, radial speed and
    heard frequency, and what the reception-time model gives instead."""
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float); pos = np.asarray(pos_enu, float); obs = np.asarray(obs_enu, float)
    te = emission_time(t_r, t_e, pos, obs, c)
    vel = np.gradient(pos, t_e, axis=0)
    at = lambda arr, t: np.array([np.interp(t, t_e, arr[:, k]) for k in range(3)])
    p, v = at(pos, te), at(vel, te)
    d = p - obs; r = float(np.linalg.norm(d)); vr = float(v @ d / r)
    dw = at(pos, t_r) - obs                      # the reception-time model: where the aircraft is when the sound arrives
    vr_wrong = float(at(vel, t_r) @ dw / np.linalg.norm(dw))
    return {"t_r": float(t_r), "t_e": te, "delay_s": float(t_r - te), "pos": p.tolist(), "vel": v.tolist(), "r": r,
            "c": float(c), "v_r": vr, "f": float(observed_frequency(f0, vr, c)),
            "v_r_reception_model": vr_wrong, "f_reception_model": float(observed_frequency(f0, vr_wrong, c))}


def cpa_from_trajectory(t_e, pos_enu, obs_enu, c=None):
    """Closest point of approach: (t_cpa_emission, t_cpa_heard, slant_m, speed_mps)."""
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float)
    d = np.asarray(pos_enu, float) - np.asarray(obs_enu, float)
    r = np.linalg.norm(d, axis=-1)
    i = int(np.argmin(r))
    vel = np.gradient(np.asarray(pos_enu, float), t_e, axis=0)
    speed = float(np.linalg.norm(vel[i]))
    return float(t_e[i]), float(t_e[i] + r[i] / c), float(r[i]), speed
