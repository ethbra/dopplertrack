"""Atmosphere for the propagation model: sound speed from temperature, and wind.

The video models still air at 343 m/s. Both matter at the level the search resolves: c moves
by 0.6 m/s per degree C, which rescales the fitted speed and distance and shifts every heard
time by r/c^2 per m/s (0.026 s at 3 km); a 7 m/s wind changes the effective sound speed along
a ray by up to 2 % and so bends the modelled curve and moves the heard CPA by up to 0.2 s at
3 km. A day's weather at the search region gives both.

`fetch_weather` reads hourly ERA5 reanalysis from Open-Meteo's free archive API (no key).
Values are surface ones; `Medium.from_surface` carries them to the middle of the sound path
with a standard lapse rate and a 1/7 power-law wind profile. Both are approximations, so
the user can always give --temp and --wind directly instead."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import requests

from .doppler import speed_of_sound

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAPSE_K_PER_M = 0.0065


@dataclass
class Medium:
    c: float                 # speed of sound, m/s
    wind: tuple | None       # (east, north) m/s the air moves towards, or None for still air

    @classmethod
    def from_surface(cls, temp_c: float, wind_speed: float = 0.0, wind_from_deg: float = 0.0,
                     path_mid_height_m: float = 500.0, profile_exponent: float = 1 / 7):
        """Surface temperature (2 m) and wind (10 m, meteorological 'from' direction) to the
        sound speed and wind at the middle of a typical source-observer path."""
        t_mid = temp_c - LAPSE_K_PER_M * path_mid_height_m
        scale = (max(path_mid_height_m, 10.0) / 10.0) ** profile_exponent
        return cls(float(speed_of_sound(t_mid)), wind_vector(wind_speed * scale, wind_from_deg) if wind_speed else None)

    @classmethod
    def still(cls, temp_c: float = 15.0):
        return cls(float(speed_of_sound(temp_c)), None)


def wind_vector(speed: float, from_deg: float) -> tuple[float, float]:
    """Meteorological wind (speed, direction it blows FROM, deg clockwise from north) to the
    (east, north) velocity of the air."""
    a = np.radians(from_deg)
    return (float(-speed * np.sin(a)), float(-speed * np.cos(a)))


def fetch_weather(lat: float, lon: float, unix_t: float, session=None, timeout: float = 60.0) -> dict:
    """Hourly ERA5 surface weather nearest unix_t at (lat, lon): temperature (C), wind speed
    (m/s) and direction (deg from). Raises on network/API errors."""
    day = datetime.fromtimestamp(unix_t, timezone.utc).strftime("%Y-%m-%d")
    params = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": day, "end_date": day,
              "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m", "wind_speed_unit": "ms",
              "timezone": "GMT"}
    r = (session or requests).get(ARCHIVE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    h = r.json()["hourly"]
    ts = np.array([datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp() for x in h["time"]])
    temp = np.asarray(h["temperature_2m"], float)
    ok = np.isfinite(temp)
    if not ok.any():
        raise ValueError(f"no weather for {day} at {lat:.3f}, {lon:.3f}")
    i = int(np.argmin(np.where(ok, np.abs(ts - unix_t), np.inf)))
    return {"time": h["time"][i], "temp_c": float(temp[i]),
            "wind_speed": float(h["wind_speed_10m"][i] or 0.0), "wind_from_deg": float(h["wind_direction_10m"][i] or 0.0)}
