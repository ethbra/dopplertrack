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


class Atmosphere:
    """Sound speed that varies with height: c(z) from a temperature profile.

    Wherever this module takes a sound speed `c`, it also takes an Atmosphere. A ray from the
    observer at height z_o to a source at z_s then travels at the harmonic mean of c over the
    heights it crosses (straight-ray approximation: the delay is r times the mean slowness),
    so a high aircraft is heard through the colder air above it. Heights are in the search
    frame's "up" coordinate, which over a few tens of km is height above the WGS-84 ellipsoid.

    The Oyster Bay recording is why this exists: scored with one sound speed for both
    flyovers, the PC-12 at 5.5 km fits best at 337 m/s and the King Air at 1.1 km at 342 m/s,
    on a 26 C afternoon (surface c 346 m/s)."""

    def __init__(self, z, temp_c, label: str = "profile", dz: float = 10.0, z_range=(-500.0, 20000.0)):
        z = np.asarray(z, float); temp_c = np.asarray(temp_c, float)
        o = np.argsort(z)
        self.z_levels, self.t_levels = z[o], temp_c[o]
        self.label = label
        self._zt = np.arange(z_range[0], z_range[1] + dz, dz)
        self._c = speed_of_sound(np.interp(self._zt, self.z_levels, self.t_levels))
        # cumulative slowness S(z) = integral of dz / c(z), trapezoidal on the table
        self._S = np.concatenate([[0.0], np.cumsum(0.5 * (1 / self._c[1:] + 1 / self._c[:-1]) * dz)])

    @classmethod
    def standard(cls, surface_temp_c: float, surface_z: float = 0.0, lapse_k_per_m: float = 0.0065,
                 tropopause_m: float = 11000.0):
        """A standard-lapse column from a surface temperature: linear up to the tropopause,
        isothermal above, constant below the surface."""
        top = surface_temp_c - lapse_k_per_m * (tropopause_m - surface_z)
        return cls([surface_z - 500.0, surface_z, tropopause_m, 20000.0], [surface_temp_c, surface_temp_c, top, top],
                   label=f"standard lapse from {surface_temp_c:.1f} C at {surface_z:.0f} m")

    def speed(self, z) -> np.ndarray:
        return np.interp(z, self._zt, self._c)

    def mean_speed(self, z1, z2) -> np.ndarray:
        """Harmonic-mean sound speed between heights z1 and z2 (elementwise)."""
        z1 = np.asarray(z1, float); z2 = np.asarray(z2, float)
        dz = z2 - z1
        dS = np.interp(z2, self._zt, self._S) - np.interp(z1, self._zt, self._S)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = dz / dS
        return np.where(np.abs(dz) < 1.0, self.speed(0.5 * (z1 + z2)), c)

    def __repr__(self):
        lv = ", ".join(f"{z:.0f} m {t:.1f} C" for z, t in zip(self.z_levels, self.t_levels))
        return f"Atmosphere({self.label}: {lv})"


def is_profile(c) -> bool:
    return isinstance(c, Atmosphere)


def describe_c(c) -> str:
    return repr(c) if is_profile(c) else f"{float(c):.1f} m/s"


def _wind3(wind):
    return np.array([float(wind[0]), float(wind[1]), 0.0])


def propagation_delay(d_src_minus_obs, c, wind=None, z_obs=None):
    """Time sound takes from the source to the observer, d = source - observer (..., 3).

    c: a sound speed, or an Atmosphere (then z_obs, the observer's height, is needed and each
    ray travels at its own mean sound speed).

    In a uniform horizontal wind w (east, north m/s, the direction the air moves towards) sound
    moves at c relative to the air. In the air's frame the observer drifts by -w tau while the
    sound is in flight, so the delay solves |d + w tau| = c tau; that quadratic has the
    closed-form positive root used here. Without wind it is |d| / c."""
    d = np.asarray(d_src_minus_obs, float)
    r2 = np.sum(d * d, axis=-1)
    if is_profile(c):
        if z_obs is None:
            raise ValueError("an Atmosphere needs the observer height z_obs")
        c = c.mean_speed(z_obs, np.asarray(z_obs, float) + d[..., 2])
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
    obs = np.asarray(obs_enu, float)
    d = pos - obs
    r = np.linalg.norm(d, axis=-1)
    if is_profile(c):
        # general form: the heard frequency is f0 dt_e/dt_obs with t_obs = t_e + tau(t_e),
        # exact for any delay model; with a constant c and no wind it is the formula below
        tau = propagation_delay(d, c, wind, z_obs=obs[2])
        t_obs = t_e + tau
        dtau = np.gradient(tau, t_e)
        if model == "reception":
            dtau = np.interp(t_obs, t_e, dtau)
        return t_obs, f0 / (1.0 + dtau), r
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
        return te + float(propagation_delay(p - obs, c, wind, z_obs=obs[2])) - t_r

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
    if is_profile(c):                    # the rest of the panel uses this ray's mean sound speed
        c = float(c.mean_speed(obs[2], p[2]))
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
    return float(t_e[i]), float(t_e[i] + propagation_delay(d[i], c, wind, z_obs=float(np.asarray(obs_enu, float)[2]))), float(r[i]), speed
