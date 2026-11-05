"""Geoid undulation (EGM96): the height of mean sea level above the WGS-84 ellipsoid.

ADS-B geometric altitude is GNSS height above the WGS-84 ellipsoid (readsb's `alt_geom`),
while terrain models give height above mean sea level. The two differ by the geoid
undulation N (h_ellipsoid = h_msl + N), which is about -33 m around New York and -28 m in
south-west Florida. Mixing them puts every observer N metres off in height relative to the
aircraft. The sample trace in tests/data shows it directly: its GNSS altitude minus its
QNH-corrected pressure altitude is -32 m where EGM96 gives -33.6 m.

The grid is NGA's EGM96 at 15', public domain, as PROJ distributes it (2.7 MB GeoTIFF),
downloaded once into the cache directory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

GEOID_URL = "https://raw.githubusercontent.com/OSGeo/PROJ-data/master/us_nga/us_nga_egm96_15.tif"
GEOID_FILE = "us_nga_egm96_15.tif"


class Geoid:
    """Bilinear lookup in a point-registered lat/lon grid of undulations (m).

    path: a GeoTIFF like PROJ's (tiepoint and pixel scale tags), or an .npz with arrays
    n (rows from north to south), lat_top, lon_left and step (a crop, as the tests use)."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            from .adsb import _cache_dir
            path = _cache_dir() / GEOID_FILE
            if not path.exists():
                r = requests.get(GEOID_URL, timeout=120)
                r.raise_for_status()
                path.write_bytes(r.content)
        path = Path(path)
        if path.suffix == ".npz":
            z = np.load(path)
            self.n = np.asarray(z["n"], np.float64)
            self.lat_top, self.lon_left, self.step = float(z["lat_top"]), float(z["lon_left"]), float(z["step"])
        else:
            from PIL import Image
            im = Image.open(path)
            tie, scale = im.tag_v2[33922], im.tag_v2[33550]
            self.n = np.asarray(im, np.float64)
            self.lat_top, self.lon_left, self.step = float(tie[4]), float(tie[3]), float(scale[0])
        self.wraps = self.n.shape[1] * self.step >= 360.0 - 1e-9

    def undulation(self, lat, lon) -> np.ndarray:
        lat = np.atleast_1d(np.asarray(lat, float)); lon = np.atleast_1d(np.asarray(lon, float))
        y = (self.lat_top - lat) / self.step
        x = (lon - self.lon_left) / self.step
        if self.wraps:
            x = np.mod(x, self.n.shape[1])
        ny, nx = self.n.shape
        if np.any((y < 0) | (y > ny - 1)) or (not self.wraps and np.any((x < 0) | (x > nx - 1))):
            raise ValueError("point outside the geoid grid")
        i = np.minimum(np.floor(y).astype(int), ny - 2); fy = y - i
        j = np.floor(x).astype(int); fx = x - j
        j1 = (j + 1) % nx if self.wraps else np.minimum(j + 1, nx - 1)
        j = np.minimum(j, nx - 1)
        g = self.n
        return (g[i, j] * (1 - fx) * (1 - fy) + g[i, j1] * fx * (1 - fy)
                + g[i + 1, j] * (1 - fx) * fy + g[i + 1, j1] * fx * fy)
