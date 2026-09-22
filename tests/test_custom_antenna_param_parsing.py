#!/usr/bin/env python3

from astra_shared.custom_antenna_schema import (
    default_custom_antenna,
    normalize_custom_antenna,
)
from astra_shared.param_parsing import _get_float_pair, _parse_custom_antenna_payload, parse_rf_params


def test_get_float_pair_preserves_requested_value_and_clamps_once():
    assert _get_float_pair({"step": "0.1"}, "step", 25.0, min_val=0.25, max_val=100.0) == (0.1, 0.25)
    assert _get_float_pair({"step": "200"}, "step", 25.0, min_val=0.25, max_val=100.0) == (200.0, 100.0)
    assert _get_float_pair({"step": "0.25"}, "step", 25.0, min_val=0.25, max_val=100.0) == (0.25, 0.25)


def test_get_float_pair_uses_default_for_non_finite_or_invalid_values():
    for value in (None, "", "bad", "NaN", "inf", "-inf"):
        assert _get_float_pair({"step": value}, "step", 25.0, min_val=0.25, max_val=100.0) == (25.0, 25.0)


def test_parse_rf_params_accepts_custom_antenna_json_string_payload():
    params = {
        "antenna_model": "custom",
        "frequency_ghz": "12.0",
        "custom_antenna": (
            '{"enabled": true, "source_format": "csv", "filename": "p.csv", '
            '"frequency_hz": 12000000000, "psi_deg": [0, 30], '
            '"phi_deg": [0, 90], "gain_dbi": [[10, 9], [8, 7]]}'
        ),
    }

    payload = parse_rf_params(params)["custom_antenna"]
    assert payload["enabled"] is True
    assert payload["source_format"] == "csv"
    assert payload["filename"] == "p.csv"
    assert payload["validation"]["is_valid"] is True


def test_parse_rf_params_invalid_custom_antenna_json_falls_back_to_default_payload():
    rf = parse_rf_params(
        {
            "antenna_model": "custom",
            "frequency_ghz": "12.0",
            "custom_antenna": "{not_json}",
        }
    )
    assert rf["custom_antenna"] == default_custom_antenna()


def test_parse_rf_params_is_idempotent_for_canonical_frequency_and_aperture():
    rf = parse_rf_params(
        {
            "freq_hz": 42_000_000_000.0,
            "aperture_radius_m": 0.42,
            "min_el_deg": 25.0,
        }
    )

    assert rf["freq_hz"] == 42_000_000_000.0
    assert rf["aperture_radius_m"] == 0.42
    assert rf["min_el_deg"] == 25.0


def test_parse_rf_params_accepts_frequency_hz_alias():
    rf = parse_rf_params({"frequency_hz": 28_000_000_000.0})

    assert rf["freq_hz"] == 28_000_000_000.0


def test_parse_custom_antenna_payload_accepts_json_string_and_normalizes():
    raw_json = (
        '{"enabled": true, "source_format": "csv", "frequency_hz": 12000000000, '
        '"psi_deg": [0, 1], "phi_deg": [0, 180], "gain_dbi": [[1, 0], [0, -1]]}'
    )
    assert _parse_custom_antenna_payload(
        {"custom_antenna": raw_json}
    ) == normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": 12_000_000_000,
            "psi_deg": [0, 1],
            "phi_deg": [0, 180],
            "gain_dbi": [[1, 0], [0, -1]],
        }
    )


def test_parse_custom_antenna_payload_bad_or_missing_values_return_default():
    assert _parse_custom_antenna_payload({}) == default_custom_antenna()
    assert (
        _parse_custom_antenna_payload({"custom_antenna": "{not-json}"})
        == default_custom_antenna()
    )
    assert (
        _parse_custom_antenna_payload({"custom_antenna": [1, 2, 3]})
        == default_custom_antenna()
    )
