import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from dopplertrack import adsb

HERE = Path(__file__).parent


def test_parse_readsb_trace():
    tr = adsb.parse_trace_file(HERE / "data" / "trace_full_a2f38b.json")
    assert tr.icao == "a2f38b" and tr.typ == "SR20" and tr.reg == "N29EF"
    assert len(tr) > 500
    assert np.all(np.diff(tr.t) >= 0)
    assert tr.lat.min() > 39 and tr.lat.max() < 42 and tr.lon.min() > -76
    air = ~tr.on_ground
    assert np.nanmax(tr.alt[air]) > 300          # metres
    assert np.nanmax(tr.gs[air]) > 40            # m/s
    segs = tr.segments()
    assert segs and all(not tr.on_ground[s.start] for s in segs)
    t, enu = tr.enu(segs[0], 40.1, -74.1)
    assert enu.shape == (len(t), 3) and np.all(np.isfinite(enu))


def test_save_load_round_trip(tmp_path):
    tr = adsb.parse_trace_file(HERE / "data" / "trace_full_a2f38b.json")
    adsb.save_traces([tr], tmp_path / "t.npz")
    back = adsb.load_traces(tmp_path / "t.npz")[0]
    assert back.icao == tr.icao and back.typ == tr.typ
    assert np.allclose(back.t, tr.t) and np.array_equal(back.on_ground, tr.on_ground)


def test_parse_keeps_what_each_report_says():
    tr = adsb.parse_trace_file(HERE / "data" / "trace_full_a2f38b.json")
    geo = tr.alt_src == adsb.ALT_SOURCES.index("geometric")
    assert geo.sum() == 1032 and (tr.alt_src == adsb.ALT_SOURCES.index("barometric")).sum() == 6
    # alt is unchanged: geometric where present, else barometric
    both = geo & np.isfinite(tr.alt_baro)
    assert np.median(tr.alt[both] - tr.alt_baro[both]) == pytest.approx(76, abs=15)    # 250 ft, QNH 1026 hPa
    assert (tr.pos_src == adsb.POS_SOURCES.index("adsb")).sum() == 1035 and (tr.pos_src == adsb.POS_SOURCES.index("tisb")).sum() == 3
    assert np.isfinite(tr.track).mean() > 0.99 and np.isfinite(tr.vrate).mean() > 0.99


def test_flag_8_marks_a_geometric_altitude_in_slot_3():
    rows = [[0.0, 40.0, -74.0, 3000, 120.0, 90.0, 0, 0, None, "adsb_icao", None],
            [5.0, 40.0, -73.99, 3100, 120.0, 90.0, 8, 0, None, "mlat", None]]
    tr = adsb.parse_trace_bytes(json.dumps({"icao": "abc123", "timestamp": 1.7e9, "trace": rows}).encode())
    assert list(tr.alt_src) == [adsb.ALT_SOURCES.index("barometric"), adsb.ALT_SOURCES.index("geometric")]
    assert tr.alt_baro[0] == pytest.approx(3000 * adsb.FT) and np.isnan(tr.alt_baro[1])
    assert np.allclose(tr.alt, np.array([3000, 3100]) * adsb.FT)
    assert list(tr.pos_src) == [adsb.POS_SOURCES.index("adsb"), adsb.POS_SOURCES.index("mlat")] and list(tr.flags) == [0, 8]


def test_save_load_keeps_the_sources_and_reads_older_files(tmp_path):
    tr = adsb.parse_trace_file(HERE / "data" / "trace_full_a2f38b.json")
    adsb.save_traces([tr], tmp_path / "t.npz")
    back = adsb.load_traces(tmp_path / "t.npz")[0]
    for k in ("alt_src", "pos_src", "flags"):
        assert np.array_equal(getattr(back, k), getattr(tr, k))
    for k in ("alt_baro", "track", "vrate"):
        assert np.allclose(getattr(back, k), getattr(tr, k), equal_nan=True)
    np.savez_compressed(tmp_path / "old.npz", n=1, meta=np.array([json.dumps([tr.icao, tr.reg, tr.typ, tr.desc])]),
                        a0=np.stack([tr.t, tr.lat, tr.lon, tr.alt, tr.gs, tr.on_ground.astype(float)]))
    old = adsb.load_traces(tmp_path / "old.npz")[0]
    assert old.alt_src is None and np.allclose(old.alt, tr.alt)
    adsb.save_traces([old], tmp_path / "old2.npz")                       # and saves back in the old layout
    assert np.load(tmp_path / "old2.npz")["a0"].shape[0] == 6


def test_geometric_altitude_is_height_above_the_ellipsoid():
    """Why the source matters: the trace's QNH-corrected pressure altitude is height above sea
    level (ISA), and geometric altitude minus it is the geoid undulation there (EGM96, NGA,
    public domain; a 41 x 41 crop at 15'), not zero. Terrain tiles are heights above sea level."""
    d = json.loads(gzip.open(HERE / "data" / "trace_full_a2f38b.json").read())
    z = np.load(HERE / "data" / "egm96_15_crop_ne_us.npz")
    qnh, diff, und = None, [], []
    for r in d["trace"]:
        if isinstance(r[8], dict) and r[8].get("nav_qnh"):
            qnh = r[8]["nav_qnh"]
        if qnh and isinstance(r[3], (int, float)) and isinstance(r[10], (int, float)):
            p = 1013.25 * (1 - 2.25577e-5 * r[3] * adsb.FT) ** 5.25588          # ISA pressure at that pressure altitude
            diff.append(r[10] * adsb.FT - (1 - (p / qnh) ** 0.190263) * 44330.77)
            y, x = (float(z["lat_top"]) - r[1]) / float(z["step"]), (r[2] - float(z["lon_left"])) / float(z["step"])
            i, j = int(y), int(x); fy, fx = y - i, x - j; n = z["n"]
            und.append(n[i, j] * (1 - fx) * (1 - fy) + n[i, j + 1] * fx * (1 - fy) + n[i + 1, j] * (1 - fx) * fy + n[i + 1, j + 1] * fx * fy)
    assert len(diff) > 500
    assert np.median(diff) == pytest.approx(np.median(und), abs=3.0)      # measured -32.2 m, EGM96 -33.6 m
