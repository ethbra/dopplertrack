"""Ground elevation from AWS Terrain Tiles (Terrarium encoding), cached on disk."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


class Terrain:
    def __init__(self, zoom: int = 12, cache_dir: str | Path | None = None):
        self.z = zoom
        self.cache = Path(cache_dir or os.environ.get("DOPPLERTRACK_CACHE", Path.home() / ".cache" / "dopplertrack")) / "terrarium" / str(zoom)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._mem: dict[tuple[int, int], np.ndarray] = {}

    def _path(self, x: int, y: int) -> Path:
        return self.cache / f"{x}_{y}.png"

    def _download(self, x: int, y: int) -> None:
        p = self._path(x, y)
        if p.exists():
            return
        for attempt in range(4):
            try:
                r = requests.get(_URL.format(z=self.z, x=x, y=y), timeout=60)
                r.raise_for_status()
                # write-then-rename so a concurrent reader (another worker process) never sees a partial file
                tmp = p.with_name(f"{p.name}.{os.getpid()}.part")
                tmp.write_bytes(r.content)
                os.replace(tmp, p)
                return
            except requests.RequestException:
                if attempt == 3:
                    raise

    def prefetch(self, tiles, workers: int = 16) -> None:
        missing = [t for t in tiles if not self._path(*t).exists()]
        if missing:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda t: self._download(*t), missing))

    def _tile(self, x: int, y: int) -> np.ndarray:
        k = (x, y)
        if k in self._mem:
            return self._mem[k]
        self._download(x, y)
        try:
            img = np.asarray(Image.open(self._path(x, y)).convert("RGB"), dtype=np.float64)
        except (OSError, Image.UnidentifiedImageError):      # a damaged cached tile: fetch it again once
            self._path(x, y).unlink(missing_ok=True)
            self._download(x, y)
            img = np.asarray(Image.open(self._path(x, y)).convert("RGB"), dtype=np.float64)
        elev = img[..., 0] * 256 + img[..., 1] + img[..., 2] / 256 - 32768
        self._mem[k] = elev
        return elev

    def elevation(self, lat, lon) -> np.ndarray:
        """Elevation in metres (MSL) for arrays of lat/lon. Nearest pixel."""
        lat = np.atleast_1d(np.asarray(lat, float)); lon = np.atleast_1d(np.asarray(lon, float))
        n = 2 ** self.z
        xf = (lon + 180.0) / 360.0 * n
        latr = np.radians(lat)
        yf = (1 - np.log(np.tan(latr) + 1 / np.cos(latr)) / np.pi) / 2 * n
        xt = np.floor(xf).astype(int); yt = np.floor(yf).astype(int)
        px = np.clip(((xf - xt) * 256).astype(int), 0, 255); py = np.clip(((yf - yt) * 256).astype(int), 0, 255)
        out = np.zeros(lat.shape)
        tiles = set(zip(xt.tolist(), yt.tolist()))
        self.prefetch(tiles)
        for (tx, ty) in tiles:
            m = (xt == tx) & (yt == ty)
            out[m] = self._tile(tx, ty)[py[m], px[m]]
        return out
