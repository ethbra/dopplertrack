"""Sound speed that varies with height (doppler.Atmosphere) and the weather profiles that
feed it. tests/data/weather_*.json are Open-Meteo responses (CC BY 4.0) for the two
validation recordings' afternoons."""
import json
from pathlib import Path

import numpy as np
import pytest

from dopplertrack import doppler
from dopplertrack.doppler import Atmosphere, speed_of_sound
from dopplertrack.weather import profile_from_response

DATA = Path(__file__).parent / "data"


def _pass(alt=5500.0):
    te = np.arange(0, 800, 1.0)
    pos = np.stack([130.0 * (te - 400), np.full_like(te, 2500.0), np.full_like(te, alt)], -1)
    return te, pos


def test_isothermal_atmosphere_is_the_constant_speed_model():
    te, pos = _pass()
    obs = np.array([0.0, 0.0, 5.0])
    b = doppler.curve_from_trajectory(te, pos, obs, 100.0, Atmosphere([0, 10000], [15.0, 15.0]))
    c = speed_of_sound(15.0)
    a = doppler.curve_from_trajectory(te, pos, obs, 100.0, c)
    assert np.allclose(a[0], b[0], atol=1e-6) and np.allclose(a[1][1:-1], b[1][1:-1], atol=1e-6)


def test_mean_speed_is_the_harmonic_mean_of_a_linear_profile():
    a = Atmosphere.standard(20.0)
    z = np.linspace(0, 5000, 20001)
    exact = 5000 / np.trapezoid(1 / speed_of_sound(20.0 - 0.0065 * z), z)
    assert float(a.mean_speed(0.0, 5000.0)) == pytest.approx(exact, abs=0.01)
    assert float(a.mean_speed(5000.0, 0.0)) == pytest.approx(exact, abs=0.01)     # either direction
    assert float(a.mean_speed(100.0, 100.2)) == pytest.approx(float(speed_of_sound(20.0 - 0.65)), abs=0.01)
    assert float(a.speed(20000.0)) == pytest.approx(float(speed_of_sound(20.0 - 0.0065 * 11000)))   # isothermal above 11 km


def test_a_high_pass_is_heard_through_colder_air():
    """With a lapse, the high aircraft's curve matches a constant-speed curve at the
    column's mean speed, not the surface speed: the effect the Oyster Bay PC-12 showed."""
    te, pos = _pass(5500.0)
    obs = np.zeros(3)
    atm = Atmosphere.standard(26.0)
    to, f, _ = doppler.curve_from_trajectory(te, pos, obs, 100.0, atm)
    tq = np.arange(300, 520, 0.5)
    ref = np.interp(tq, to, f)

    def err(c):
        tc, fc, _ = doppler.curve_from_trajectory(te, pos, obs, 100.0, c)
        return np.max(np.abs(np.interp(tq, tc, fc) - ref))
    surface, column = float(speed_of_sound(26.0)), float(atm.mean_speed(0, 5500))
    assert column < surface - 8
    assert err(column) < 0.25 * err(surface)


def test_cpa_table_and_curve_use_the_same_delay():
    from dopplertrack.search import Grid, Segment, cpa_table
    te, pos = _pass(3000.0)
    seg = Segment(0, "x", te + 1e9, pos, np.full(len(te), 130.0))
    g = Grid.points([40.0], [-73.0], [0.0])
    atm = Atmosphere.standard(25.0)
    row = cpa_table(g, [seg], max_range_m=10000, c=atm).iloc[0]
    tcpa, theard, _, _ = doppler.cpa_from_trajectory(seg.t, seg.enu, g.enu[0], c=atm)
    assert row.t_heard == pytest.approx(theard, abs=1e-6)
    assert theard - tcpa == pytest.approx(np.linalg.norm(pos[400] - g.enu[0]) / float(atm.mean_speed(g.enu[0][2], 3000.0)), rel=1e-3)


@pytest.mark.parametrize("name,t,surface", [("oyster_bay", 1788900805, 26.0), ("swfl", 1735415920, 26.3)])
def test_profiles_from_the_validation_afternoons(name, t, surface):
    w = profile_from_response(json.load(open(DATA / f"weather_{name}.json")), t)
    z, temp = np.array(w["levels"]).T
    assert temp[0] == pytest.approx(surface, abs=0.5) and np.all(np.diff(z) > 0)
    a = w["atmosphere"]
    # the column the Oyster Bay PC-12 was heard through is ~11 m/s slower than the surface air
    assert 333 < float(a.mean_speed(0, 5500)) < 338 and float(a.speed(0)) > 346
    assert w["wind"] is not None and np.hypot(*w["wind"]) < 15


def test_cli_lapse_gives_a_profile():
    import argparse
    from dopplertrack import cli
    a = argparse.Namespace(temp=20.0, wind=None, lapse=True, weather=False)
    c, wind = cli._medium(a)
    assert isinstance(c, Atmosphere) and wind is None
    a.lapse = False
    assert cli._medium(a)[0] == pytest.approx(float(speed_of_sound(20.0)))
