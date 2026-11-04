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


DOPPLER_MODELS = ("emission", "reception")


def _wind3(wind):
    return np.array([float(wind[0]), float(wind[1]), 0.0])


def propagation_delay(d_src_minus_obs, c, wind=None):
    """Time sound takes from the source to the observer, d = source - observer (..., 3).

    In a uniform horizontal wind w (east, north m/s, the direction the air moves towards) sound
    moves at c relative to the air. In the air's frame the observer drifts by -w tau while the
    sound is in flight, so the delay solves |d + w tau| = c tau; that quadratic has the
    closed-form positive root used here. Without wind it is |d| / c."""
    d = np.asarray(d_src_minus_obs, float)
    r2 = np.sum(d * d, axis=-1)
    if wind is None:
        return np.sqrt(r2) / c
    w = _wind3(wind)
    dw = d @ w
    a = c * c - float(w @ w)
    return (dw + np.sqrt(dw * dw + a * r2)) / a


def curve_from_trajectory(t_e, pos_enu, obs_enu, f0=1.0, c=None, wind=None, model="emission"):
    """Doppler curve a stationary observer hears from a sampled trajectory.

    t_e: (N,) emission times (s); pos_enu: (N,3) source positions (m);
    obs_enu: (3,) observer position (m). Returns (t_obs, f, r) with f for
    rest frequency f0 (use f0=1 to get the ratio).

    wind: optional uniform horizontal wind (east, north) m/s, the direction the air moves
    towards. Exact for a uniform wind: with D = source - observer the sound heard at
    t_e + tau left when |D + w tau| = c tau, and differentiating that gives
    f = f0 (c - m.w) / (c - m.w + m.v), m = (D + w tau) / (c tau) the unit vector from the
    observer to where the air carrying the wavefront started. Both the sound speed along the
    ray (c - m.w) and the tilt of the ray (m against D/|D|, an angle of about |w|/c) matter;
    the tilt, multiplied by the aircraft's speed, is the larger effect.

    model="emission" (correct): the tone heard at t_obs = t_e + r(t_e)/c was shifted by the
    range rate at t_e. model="reception" is the mistake the video warns about: the heard time
    is still delayed, but the shift is taken from the range rate at the *reception* time
    (where the aircraft is when the sound arrives), which bends the curve, not just shifts it.
    Kept for comparison (`dopplertrack explain`), never used for scoring.
    """
    if model not in DOPPLER_MODELS:
        raise ValueError(f"unknown model {model!r}; use one of {DOPPLER_MODELS}")
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float)
    pos = np.asarray(pos_enu, float)
    d = pos - np.asarray(obs_enu, float)
    r = np.linalg.norm(d, axis=-1)
    if wind is None:
        ce = np.full(r.shape, float(c))
        rr = np.gradient(r, t_e)
        t_obs = t_e + r / c
    else:
        w = _wind3(wind)
        tau = propagation_delay(d, c, wind)
        m = (d + tau[:, None] * w) / (c * tau)[:, None]
        ce = c - m @ w
        rr = np.sum(m * np.gradient(pos, t_e, axis=0), axis=-1)
        t_obs = t_e + tau
    if model == "reception":
        rr = np.interp(t_obs, t_e, rr)
    f = f0 * ce / (ce + rr)
    return t_obs, f, r


def emission_time(t_r, t_e, pos_enu, obs_enu, c=None, wind=None) -> float:
    """Solve t_r = t_e + r(t_e)/c for the emission time of the sound heard at t_r.

    t_obs(t_e) is strictly increasing for a subsonic source, so the root is unique: bracket it
    on the samples, then refine with Brent on the linearly interpolated trajectory."""
    from scipy.optimize import brentq
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float); pos = np.asarray(pos_enu, float); obs = np.asarray(obs_enu, float)

    def at(te):
        p = np.array([np.interp(te, t_e, pos[:, k]) for k in range(3)])
        return te + float(propagation_delay(p - obs, c, wind)) - t_r

    t_obs, _, _ = curve_from_trajectory(t_e, pos, obs, 1.0, c, wind)
    if not (t_obs[0] <= t_r <= t_obs[-1]):
        raise ValueError(f"t_r={t_r} is outside the heard span [{t_obs[0]:.1f}, {t_obs[-1]:.1f}]")
    i = int(np.clip(np.searchsorted(t_obs, t_r), 1, len(t_e) - 1))
    return float(brentq(at, t_e[i - 1], t_e[i], xtol=1e-9))


def explain(t_r, t_e, pos_enu, obs_enu, f0, c=None, wind=None) -> dict:
    """The video's 'Building the Doppler Curve, Step by Step' panel for one reception time:
    emission time, position and velocity there, slant range, radial speed, heard frequency,
    and what the reception-time (wrong) model would say instead."""
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float); pos = np.asarray(pos_enu, float); obs = np.asarray(obs_enu, float)
    te = emission_time(t_r, t_e, pos, obs, c, wind)
    vel = np.gradient(pos, t_e, axis=0)
    p = np.array([np.interp(te, t_e, pos[:, k]) for k in range(3)])
    v = np.array([np.interp(te, t_e, vel[:, k]) for k in range(3)])
    d = p - obs
    r = float(np.linalg.norm(d))
    if wind is None:
        m, ce = d / r, float(c)
    else:
        w = _wind3(wind)
        tau = float(propagation_delay(d, c, wind))
        m = (d + tau * w) / (c * tau)
        ce = float(c - m @ w)
    vr = float(v @ m)          # radial speed along the ray (dr/dt in still air)
    # wrong model: radial speed of the aircraft where it is at the reception time
    pr = np.array([np.interp(t_r, t_e, pos[:, k]) for k in range(3)])
    vrr = np.array([np.interp(t_r, t_e, vel[:, k]) for k in range(3)])
    dr = pr - obs
    vr_wrong = float(vrr @ dr / np.linalg.norm(dr))
    return {"t_r": float(t_r), "t_e": te, "delay_s": float(t_r - te), "pos": p.tolist(), "vel": v.tolist(),
            "r": r, "c_eff": ce, "v_r": vr, "f": float(f0 * ce / (ce + vr)),
            "v_r_reception_model": vr_wrong, "f_reception_model": float(f0 * ce / (ce + vr_wrong))}


def cpa_from_trajectory(t_e, pos_enu, obs_enu, c=None, wind=None):
    """Closest point of approach: (t_cpa_emission, t_cpa_heard, slant_m, speed_mps)."""
    c = speed_of_sound() if c is None else c
    t_e = np.asarray(t_e, float)
    d = np.asarray(pos_enu, float) - np.asarray(obs_enu, float)
    r = np.linalg.norm(d, axis=-1)
    i = int(np.argmin(r))
    vel = np.gradient(np.asarray(pos_enu, float), t_e, axis=0)
    speed = float(np.linalg.norm(vel[i]))
    return float(t_e[i]), float(t_e[i] + propagation_delay(d[i], c, wind)), float(r[i]), speed
