"""Atmosphere for the propagation model: sound speed from temperature, and wind.

The speed of sound that matters is the mean along each ray, so it depends on the aircraft's
height. `fetch_profile` gives the hour's temperature against height (pressure levels from
Open-Meteo's historical forecast archive, no key) as a doppler.Atmosphere, plus the mean wind
below 3 km. On the Oyster Bay recording the per-flyover sound speeds the audio prefers (337
m/s for the PC-12 at 5.5 km, 342 m/s for the King Air at 1.1 km) match that profile's column
means (336 and 344 m/s), while the afternoon's surface value used for both (346 m/s) cut the
score at the true point by 16 % and moved the peak from 66 m to 166 m: one sound speed taken
from the ground is worse than the old 15 C default, which happens to sit between the two.

`fetch_weather` (surface ERA5) and `Medium.from_surface` remain for a single value; prefer
the profile."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import requests

from .doppler import speed_of_sound

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAPSE_K_PER_M = 0.0065


def _get(session, url, params, timeout, attempts: int = 4):
    """GET with exponential backoff: proxies and free APIs drop the odd connection."""
    import time
    for k in range(attempts):
        try:
            r = (session or requests).get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == attempts - 1:
                raise
            time.sleep(2 ** (k + 1))


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
    h = _get(session, ARCHIVE_URL, params, timeout).json()["hourly"]
    ts = np.array([datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp() for x in h["time"]])
    temp = np.asarray(h["temperature_2m"], float)
    ok = np.isfinite(temp)
    if not ok.any():
        raise ValueError(f"no weather for {day} at {lat:.3f}, {lon:.3f}")
    i = int(np.argmin(np.where(ok, np.abs(ts - unix_t), np.inf)))
    return {"time": h["time"][i], "temp_c": float(temp[i]),
            "wind_speed": float(h["wind_speed_10m"][i] or 0.0), "wind_from_deg": float(h["wind_direction_10m"][i] or 0.0)}


PROFILE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PRESSURE_LEVELS = (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300)


def fetch_profile(lat: float, lon: float, unix_t: float, session=None, timeout: float = 60.0,
                  wind_top_m: float = 3000.0) -> dict:
    """Temperature against height (pressure levels with their geopotential heights, plus the
    2 m temperature at the model's ground height) and the mean wind below wind_top_m, for the
    hour nearest unix_t, from Open-Meteo's historical forecast archive (no key). Returns
    {"time", "atmosphere" (doppler.Atmosphere), "wind" (east, north) or None, "levels"}."""
    from .doppler import Atmosphere
    day = datetime.fromtimestamp(unix_t, timezone.utc).strftime("%Y-%m-%d")
    var = ["temperature_2m", "wind_speed_10m", "wind_direction_10m"]
    for p in PRESSURE_LEVELS:
        var += [f"temperature_{p}hPa", f"geopotential_height_{p}hPa", f"wind_speed_{p}hPa", f"wind_direction_{p}hPa"]
    params = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": day, "end_date": day,
              "hourly": ",".join(var), "wind_speed_unit": "ms", "timezone": "GMT"}
    return profile_from_response(_get(session, PROFILE_URL, params, timeout).json(), unix_t, wind_top_m)


def profile_from_response(js: dict, unix_t: float, wind_top_m: float = 3000.0) -> dict:
    from .doppler import Atmosphere
    h = js["hourly"]
    ts = np.array([datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp() for x in h["time"]])
    i = int(np.argmin(np.abs(ts - unix_t)))
    ground = float(js.get("elevation", 0.0) or 0.0)
    z, t, wz, we, wn = [ground + 2.0], [float(h["temperature_2m"][i])], [10.0 + ground], [], []
    e0, n0 = wind_vector(float(h["wind_speed_10m"][i] or 0.0), float(h["wind_direction_10m"][i] or 0.0))
    we.append(e0); wn.append(n0)
    for p in PRESSURE_LEVELS:
        zz, tt = h.get(f"geopotential_height_{p}hPa", [None])[i], h.get(f"temperature_{p}hPa", [None])[i]
        if zz is None or tt is None or zz <= ground + 2.0:
            continue                      # level below ground (or missing)
        z.append(float(zz)); t.append(float(tt))
        ws, wd = h.get(f"wind_speed_{p}hPa", [None])[i], h.get(f"wind_direction_{p}hPa", [None])[i]
        if ws is not None and wd is not None and zz <= wind_top_m:
            e, n = wind_vector(float(ws), float(wd)); wz.append(float(zz)); we.append(e); wn.append(n)
    wind = (float(np.mean(we)), float(np.mean(wn))) if we else None
    lv = list(zip(z, t))
    return {"time": h["time"][i], "atmosphere": Atmosphere(z, t, label=f"Open-Meteo {h['time'][i]} UTC"),
            "wind": wind, "levels": lv}
