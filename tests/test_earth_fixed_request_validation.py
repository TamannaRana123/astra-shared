from astra_shared.earth_fixed_validation import validate_earth_fixed_request


def test_earth_fixed_validation_rejects_phased_array_and_missing_targets():
    errors = validate_earth_fixed_request(
        {"pointing_mode": "earth_fixed", "antenna_model": "phased_array", "targets": []},
        analysis_mode="coverage",
    )
    assert "earth_fixed_phased_array_unsupported" in errors
    assert "earth_fixed_targets_required" in errors

