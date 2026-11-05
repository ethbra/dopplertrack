"""Candidate generation and scoring on synthetic flights: the pair join must keep the true
pair and drop a same-geometry pair at the wrong lag, the lens lattice must put a point within
one cell of the true observer with exact gates satisfied, the silence filter must remove it
when a third flight passes between the events, and scoring synthetic audio must rank it first."""
import numpy as np
import pytest

from dopplertrack import audio, doppler, tracks, search
from dopplertrack.adsb import Trace
from dopplertrack.geo import enu_to_lla, haversine_m
from dopplertrack.tracks import EventFit

C = 340.0
LAT0, LON0 = 40.8765, -73.525
T0 = 1_700_000_000.0


def _trace(icao, t, enu, gs):
    lat, lon, alt = enu_to_lla(enu[:, 0], enu[:, 1], enu[:, 2], LAT0, LON0)
    return Trace(icao, None, "C172", "synthetic", t, lat, lon, alt, np.full_like(t, gs), np.zeros(len(t), bool))


def _straight(icao, t_cpa, p_cpa, vel, span=450.0, bend=0.0):
    """A straight pass through p_cpa at t_cpa, optionally with a gentle turn: bend * tau^2
    sideways, which breaks the mirror symmetry two perfectly straight passes always have."""
    t = T0 + np.arange(t_cpa - span, t_cpa + span, 5.0)
    tau = (t - (T0 + t_cpa))[:, None]
    side = np.array([vel[1], -vel[0], 0.0]) / np.linalg.norm(vel[:2])
    enu = np.asarray(p_cpa, float)[None, :] + np.asarray(vel, float)[None, :] * tau + bend * tau ** 2 * side[None, :]
    return _trace(icao, t, enu, np.linalg.norm(vel) / 0.5144)


def _flights(extra=()):
    fl = [_straight("aaa001", 300, (0, 2500, 1500), (110, 0, 0)),          # event 1: east-bound, 2.5 km north
          _straight("aaa002", 1220, (3000, 0, 1200), (0, 95, 0), bend=0.1),   # event 2: north-bound, 3 km east, gentle turn
          _straight("bbb001", 1500, (3000, 0, 1200), (0, 95, 0)),         # same geometry as event 2, 280 s late (after the silence window)
          _straight("ccc001", 300, (60000, 2500, 1500), (110, 0, 0))]      # event-1 twin 60 km east
    return fl + list(extra)


D1, D2 = np.hypot(2500, 1500), np.hypot(3000, 1200)
EV1 = EventFit(t_cpa=300 + D1 / C, t_cpa_emit=300, f0=120, v=110, d=D1, t_start=230, t_end=400, score=1, f_pre=130, f_post=110)
EV2 = EventFit(t_cpa=1220 + D2 / C, t_cpa_emit=1220, f0=130, v=95, d=D2, t_start=1150, t_end=1310, score=1, f_pre=140, f_post=120)
GATES = dict(v1=(EV1.v * 0.65, EV1.v * 1.35), d1=(EV1.d * 0.4, EV1.d * 2.5), v2=(EV2.v * 0.65, EV2.v * 1.35), d2=(EV2.d * 0.4, EV2.d * 2.5))


def _index(extra=()):
    return search.DayIndex.build(_flights(extra), LAT0, LON0)


def _seg(idx, icao):
    return [i for i, s in enumerate(idx.segs) if s.icao == icao][0]


def test_index_roundtrip(tmp_path):
    idx = _index()
    assert len(idx) == 4 and np.all(np.diff(idx.sb) >= 0)
    assert np.allclose(idx.spos, idx.enu[idx.off[idx.ss] + idx.sj])
    idx.save(tmp_path / "day.npz")
    back = search.DayIndex.load(tmp_path / "day.npz")
    assert back.fingerprint() == idx.fingerprint() and np.allclose(back.spos, idx.spos)
    assert [s.icao for s in back.segs] == [s.icao for s in idx.segs]


def test_pair_join_keeps_true_pair_and_drops_wrong_lag():
    idx = _index()
    pairs = search.pair_join(idx, EV2.t_cpa - EV1.t_cpa, c=C, **GATES)
    got = set(zip(pairs.seg1, pairs.seg2))
    assert (_seg(idx, "aaa001"), _seg(idx, "aaa002")) in got
    assert (_seg(idx, "aaa001"), _seg(idx, "bbb001")) not in got        # right place, 280 s late
    assert (_seg(idx, "ccc001"), _seg(idx, "aaa002")) not in got        # 60 km away
    a, b = _seg(idx, "aaa001"), _seg(idx, "aaa002")
    row = pairs[(pairs.seg1 == a) & (pairs.seg2 == b)].iloc[0]
    assert row.b_min * idx.bucket_s <= T0 + 300 <= (row.b_max + 1) * idx.bucket_s


def _params(**kw):
    return search.LensParams(dt_heard=EV2.t_cpa - EV1.t_cpa, c=C, **GATES, **kw)


def test_lens_puts_a_point_within_one_cell_of_the_observer():
    idx = _index()
    pairs = search.pair_join(idx, EV2.t_cpa - EV1.t_cpa, c=C, **GATES)
    cands = search.lens_candidates(idx, pairs, _params(silence_d_m=None), workers=1)
    assert len(cands) > 0
    dist = haversine_m(cands.lat, cands.lon, LAT0, LON0)
    assert dist.min() < 150.0                                            # 200 m lattice: within half a diagonal
    # every candidate satisfies the exact gates
    p = _params()
    assert (cands.d1.between(*p.d1) & cands.d2.between(*p.d2) & cands.v1.between(*p.v1) & cands.v2.between(*p.v2)).all()
    assert ((cands.t_heard2 - cands.t_heard1 - p.dt_heard).abs() <= p.tol_s).all()
    near = cands.iloc[int(np.argmin(dist))]
    assert abs(near.d1 - D1) < 200 and abs(near.d2 - D2) < 200 and abs(near.t_heard1 - (T0 + EV1.t_cpa)) < 2


def test_silence_filter_removes_the_observer_when_a_third_flight_passes():
    breaker = _straight("ddd001", 800, (0, 2000, 1000), (100, 0, 0))        # 2.2 km, heard between the events
    idx = _index([breaker])
    pairs = search.pair_join(idx, EV2.t_cpa - EV1.t_cpa, c=C, **GATES)
    loud = search.lens_candidates(idx, pairs, _params(silence_d_m=None), workers=1)
    quiet = search.lens_candidates(idx, pairs, _params(silence_d_m=6000.0), workers=1)
    assert haversine_m(loud.lat, loud.lon, LAT0, LON0).min() < 150.0
    assert len(quiet) == 0 or haversine_m(quiet.lat, quiet.lon, LAT0, LON0).min() > 3000.0


def test_search_ranks_the_observer_first_on_synthetic_audio(tmp_path):
    idx = _index()
    sr, dur = 4000, 1800
    tt = np.arange(0, dur, 1 / sr)
    x = 0.4 * np.random.default_rng(1).standard_normal(len(tt)).astype(np.float32)
    for icao, f0 in (("aaa001", 120.0), ("aaa002", 130.0)):
        seg = idx.segs[_seg(idx, icao)]
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, np.zeros(3), f0=1.0, c=C)
        finst = np.interp(tt + T0, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        x[m] += 0.05 * np.sin(ph[m])
    sp = audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)
    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")
    ev2 = tracks.fit_event(sp, 1150, 1310, 100, 180, c=C, objective="brightness")
    cache = tmp_path / "cands.npz"
    res = search.run_search(sp, ev1, ev2, idx, c=C, workers=1, cand_cache=str(cache), log=lambda m: None)
    assert len(res) > 0 and cache.exists()
    best = res.iloc[0]
    assert best.icao1 == "aaa001" and best.icao2 == "aaa002"
    assert haversine_m(best.lat, best.lon, LAT0, LON0) < 400.0
    again = search.run_search(sp, ev1, ev2, idx, c=C, workers=1, cand_cache=str(cache), log=lambda m: None)
    assert len(again) == len(res)


def test_extra_flyover_rescoring_keeps_the_observer_first_and_picks_the_right_flight():
    third = _straight("eee001", 1500, (0, -1800, 900), (-100, 0, 0))     # low lateral pass after both events
    idx = _index([third])
    sr, dur = 4000, 1800
    tt = np.arange(0, dur, 1 / sr)
    x = 0.4 * np.random.default_rng(1).standard_normal(len(tt)).astype(np.float32)
    for icao, f0 in (("aaa001", 120.0), ("aaa002", 130.0), ("eee001", 150.0)):
        seg = idx.segs[_seg(idx, icao)]
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, np.zeros(3), f0=1.0, c=C)
        finst = np.interp(tt + T0, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        x[m] += 0.05 * np.sin(ph[m])
    sp = audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)
    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")
    ev2 = tracks.fit_event(sp, 1150, 1310, 100, 180, c=C, objective="brightness")
    ev3 = tracks.fit_event(sp, 1430, 1570, 110, 200, c=C, objective="brightness")
    res = search.run_search(sp, ev1, ev2, idx, c=C, workers=1, log=lambda m: None)
    out = search.rescore_with_extra(sp, ev1, [ev3], idx, res, top_pairs=5, c=C, workers=1, log=lambda m: None)
    assert set(out.columns) >= {"score12", "s3", "x_icao1"}
    best = out.iloc[0]
    assert best.icao1 == "aaa001" and best.icao2 == "aaa002" and best.x_icao1 == "eee001"
    assert haversine_m(best.lat, best.lon, LAT0, LON0) < 400.0
    assert best.s3 > 0.5 * best.s1
    assert np.isclose(best.score, best.score12 + best.s3)


def _audio(idx, flights):
    sr, dur = 4000, 1800
    tt = np.arange(0, dur, 1 / sr)
    x = 0.4 * np.random.default_rng(1).standard_normal(len(tt)).astype(np.float32)
    for icao, f0 in flights:
        seg = idx.segs[_seg(idx, icao)]
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, np.zeros(3), f0=1.0, c=C)
        finst = np.interp(tt + T0, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        x[m] += 0.05 * np.sin(ph[m])
    return audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)


def test_bbox_restricts_the_join_to_flights_in_the_box():
    idx = _index()
    east = idx.sample_mask_in_bbox((LAT0 - 0.2, LAT0 + 0.2, LON0 + 0.5, LON0 + 0.9))     # around the twin 60 km east
    pairs = search.pair_join(idx, EV2.t_cpa - EV1.t_cpa, c=C, a_mask=east, **GATES)
    assert _seg(idx, "aaa001") not in set(pairs.seg1)
    home = idx.sample_mask_in_bbox((LAT0 - 0.05, LAT0 + 0.05, LON0 - 0.05, LON0 + 0.05), pad_m=5000.0)
    pairs = search.pair_join(idx, EV2.t_cpa - EV1.t_cpa, c=C, a_mask=home, **GATES)
    assert (_seg(idx, "aaa001"), _seg(idx, "aaa002")) in set(zip(pairs.seg1, pairs.seg2))


def test_known_point_scores_only_that_point():
    idx = _index()
    sp = _audio(idx, (("aaa001", 120.0), ("aaa002", 130.0)))
    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")
    ev2 = tracks.fit_event(sp, 1150, 1310, 100, 180, c=C, objective="brightness")
    res = search.run_search(sp, ev1, ev2, idx, c=C, workers=1, points=([LAT0], [LON0], [0.0]), log=lambda m: None)
    assert len(res) >= 1 and np.allclose(res.lat, LAT0) and np.allclose(res.lon, LON0)
    assert res.icao1[0] == "aaa001" and res.icao2[0] == "aaa002" and abs(res.d1[0] - D1) < 5


def test_one_event_at_a_known_time_locates_up_to_the_mirror():
    idx = _index()
    sp = _audio(idx, (("aaa001", 120.0),))
    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=C, objective="brightness")
    # the clock pins the position along the track only to tol_s * v, so with the default 30 s a
    # point 3 km along the track scores the same; 5 s here
    res = search.run_search(sp, ev1, None, idx, c=C, workers=1, t_heard_unix=T0 + EV1.t_cpa, tol_s=5.0, log=lambda m: None)
    assert len(res) > 0 and (res.seg2 == -1).all() and (res.icao2 == "").all()
    best = res.iloc[0]
    assert best.icao1 in ("aaa001", "ccc001")                               # the twin 60 km east is the same pass
    e0 = 0.0 if best.icao1 == "aaa001" else 60000.0
    d = [haversine_m(best.lat, best.lon, *enu_to_lla(e0, n, 0.0, LAT0, LON0)[:2]) for n in (0.0, 5000.0)]   # either side of the track
    assert min(d) < 800.0
