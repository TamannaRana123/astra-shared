from astra_shared.earth_fixed_validation import (
    EARTH_FIXED_VALIDATION_REGISTRY,
    EarthFixedValidationError,
    earth_fixed_validation_error_envelope,
    get_highest_cn_availability,
    validate_earth_fixed_request,
)


def earth_fixed(**overrides):
    value = {
        "pointing_mode": "earth_fixed",
        "assignment_rule": "highest_elevation",
        "targets": [{"target_id": "t1", "label": "T1", "target_lat": 0.0, "target_lon": 0.0, "target_alt_km": 0.0}],
    }
    value.update(overrides)
    return value


def test_registry_contains_metadata_for_each_known_code():
    assert EARTH_FIXED_VALIDATION_REGISTRY
    for code, meta in EARTH_FIXED_VALIDATION_REGISTRY.items():
        assert meta["message"]
        assert meta["field"] is not None
        assert meta["category"]


def test_validation_error_has_stable_codes_and_request_envelope():
    exc = EarthFixedValidationError(["invalid_pointing_mode", "invalid_pointing_mode"])
    assert exc.codes == ["invalid_pointing_mode"]
    envelope = earth_fixed_validation_error_envelope(exc.codes)
    assert envelope["error"] == "earth_fixed_validation_failed"
    assert envelope["errors"][0]["code"] == "invalid_pointing_mode"
    assert envelope["errors"][0]["category"] == "request"


def test_unknown_validation_code_uses_safe_metadata_fallback():
    detail = earth_fixed_validation_error_envelope(["unknown_code"])["errors"][0]
    assert detail["message"] == "Validation failed."
    assert detail["category"] == "internal"


def test_disabled_highest_cn_sources_are_available_without_adapters():
    result = get_highest_cn_availability("point", earth_fixed(assignment_rule="highest_cn"))
    assert result["clutter_loss"]["available"]
    assert result["atmospheric_loss"]["available"]


def test_static_point_highest_cn_atmosphere_is_rejected():
    errors = validate_earth_fixed_request(
        earth_fixed(assignment_rule="highest_cn", atmospheric_mode="enable"),
        entry_point="live_point_bulk",
    )
    assert errors == ["highest_cn_atmosphere_unavailable"]


def test_disabled_nadir_does_not_validate_stale_earth_fixed_fields():
    assert validate_earth_fixed_request({"pointing_mode": "nadir", "assignment_rule": "highest_cn", "atmospheric_mode": "enable"}) == []
