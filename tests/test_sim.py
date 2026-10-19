import json

import pytest

from f1_mlops._sim import Race

CONFIG = {
    "total_laps": 5, "base_lap_time": 80.0, "fuel_effect": -0.1, "pit_loss": 20.0,
    "lap_noise_sd": 0.0,
    "compounds": [{"name": "SOFT", "pace_offset": 0.0, "degradation": 0.2},
                  {"name": "HARD", "pace_offset": 1.0, "degradation": 0.05}],
    "push": {"pace_delta": -0.3, "wear": 1.5}, "save": {"pace_delta": 0.3, "wear": 0.6},
    "overtake_delta": 1.0, "follow_gap": 0.5, "dirty_air_gap": 2.0, "dirty_air_loss": 0.4,
    "drivers": [{"id": "A", "pace_offset": 0.0, "start_compound": "SOFT"},
                {"id": "B", "pace_offset": 0.0, "start_compound": "SOFT", "start_gap": 30.0}],
}


def race(seed: int | None = 0, **overrides) -> Race:
    return Race(json.dumps({**CONFIG, **overrides}), seed)


def test_lap_time_follows_the_model() -> None:
    r = race()
    r.step({})
    a = r.driver("A")
    assert a.last_lap_time == pytest.approx(80.0 + 0.2 * 1)  # one lap of wear, no fuel burnt yet
    r.step({})
    assert r.driver("A").last_lap_time == pytest.approx(80.0 + 0.2 * 2 - 0.1)
    assert (a.tyre_age, a.wear) == (1, 1.0)


def test_race_ends_after_total_laps() -> None:
    r = race()
    for _ in range(5):
        assert not r.finished
        r.step({})
    assert r.finished and r.lap == 5
    with pytest.raises(ValueError, match="finished"):
        r.step({})


def test_pit_stop_costs_pit_loss_and_fits_new_tyres() -> None:
    r, ref = race(), race()
    r.step({"A": "PIT_HARD"})
    ref.step({})
    a = r.driver("A")
    assert a.last_lap_time == pytest.approx(ref.driver("A").last_lap_time + 20.0)
    assert (a.compound, a.tyre_age, a.wear, a.pit_stops, a.compounds_used) == ("HARD", 0, 0.0, 1, 2)
    r.step({})
    assert r.driver("A").last_lap_time == pytest.approx(80.0 + 1.0 + 0.05 - 0.1)


def test_push_is_faster_now_and_wears_tyres_more() -> None:
    push, stay = race(), race()
    push.step({"A": "PUSH"})
    stay.step({})
    assert push.driver("A").last_lap_time == pytest.approx(stay.driver("A").last_lap_time - 0.3 + 0.2 * 0.5)
    assert push.driver("A").wear == pytest.approx(1.5)


def test_positions_and_gaps_follow_total_time() -> None:
    r = race()
    r.step({"A": "PIT_HARD"})  # A loses 20 s but stays ahead of B (30 s back)
    a, b = r.driver("A"), r.driver("B")
    assert (a.position, b.position) == (1, 2)
    assert b.gap_to_leader == pytest.approx(10.0)
    assert [d.id for d in r.standings()] == ["A", "B"]


def test_slightly_faster_car_is_held_behind_and_loses_time_in_dirty_air() -> None:
    drivers = [{"id": "A", "pace_offset": 0.0, "start_compound": "SOFT"},
               {"id": "B", "pace_offset": -0.5, "start_compound": "SOFT", "start_gap": 0.6}]
    r = race(drivers=drivers)
    r.step({})
    a, b = r.driver("A"), r.driver("B")
    # B is 0.5 s faster (0.1 s after dirty air), not the 1.0 s needed to pass: held 0.5 s back.
    assert b.position == 2
    assert b.total_time == pytest.approx(a.total_time + 0.5)


def test_much_faster_car_passes() -> None:
    drivers = [{"id": "A", "pace_offset": 0.0, "start_compound": "SOFT"},
               {"id": "B", "pace_offset": -3.0, "start_compound": "SOFT", "start_gap": 0.6}]
    r = race(drivers=drivers)
    r.step({})
    assert r.driver("B").position == 1


def test_seed_makes_noise_reproducible() -> None:
    def times(seed: int) -> list[float]:
        r = race(seed, lap_noise_sd=0.5)
        out = []
        while not r.finished:
            r.step({})
            out.append(r.driver("A").last_lap_time)
        return out

    assert times(1) == times(1)
    assert times(1) != times(2)


@pytest.mark.parametrize("actions, error", [
    ({"A": "PIT_WET"}, ValueError),
    ({"A": "BOX"}, ValueError),
    ({"Z": "STAY"}, KeyError),
])
def test_bad_actions_are_errors(actions: dict, error: type) -> None:
    with pytest.raises(error):
        race().step(actions)


def test_bad_config_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown compound"):
        race(drivers=[{"id": "A", "pace_offset": 0.0, "start_compound": "WET"}])
    with pytest.raises(ValueError, match="invalid sim config"):
        Race("{}")
