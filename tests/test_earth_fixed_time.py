from astra_shared.earth_fixed_time import normalize_simulation_time, parse_simulation_time_instant


def test_equivalent_timestamp_spellings_share_millisecond_identity():
    values = [
        "2026-01-01T00:00:00Z",
        "2025-12-31T19:00:00-05:00",
        "2026-01-01T00:00:00.0009Z",
    ]
    assert [normalize_simulation_time(value)[0] for value in values] == [
        "2026-01-01T00:00:00.000Z",
        "2026-01-01T00:00:00.000Z",
        "2026-01-01T00:00:00.000Z",
    ]
    assert parse_simulation_time_instant(values[0])[0] == parse_simulation_time_instant(values[1])[0]


def test_invalid_time_returns_stable_error():
    assert normalize_simulation_time("not-a-time") == (None, "coverage_time_invalid")
