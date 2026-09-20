"""Validation primitives shared by earth-fixed request boundaries."""

from __future__ import annotations

import math


def _validation_meta(field: str | None, category: str, message: str) -> dict:
    return {"field": field, "category": category, "message": message}


_VALIDATION_MESSAGES = {
    "invalid_pointing_mode": ("pointing_mode", "Antenna pointing must be nadir or earth_fixed."),
    "pointing_mode_invalid": ("pointing_mode", "Antenna pointing must be nadir or earth_fixed."),
    "invalid_assignment_rule": ("assignment_rule", "Assignment rule must be highest_elevation or highest_cn."),
    "assignment_rule_invalid": ("assignment_rule", "Assignment rule must be highest_elevation or highest_cn."),
    "max_steer_deg_invalid": ("max_steer_deg", "Max steering angle must be a finite number from 0 to 90 degrees."),
    "max_steer_deg_out_of_range": ("max_steer_deg", "Max steering angle must be from 0 to 90 degrees."),
    "min_elevation_deg_invalid": ("min_elevation_deg", "Minimum elevation angle must be a finite number from 0 to 90 degrees."),
    "min_elevation_deg_out_of_range": ("min_elevation_deg", "Minimum elevation angle must be from 0 to 90 degrees."),
    "min_el_deg_invalid": ("min_el_deg", "Minimum elevation angle must be a finite number from 0 to 90 degrees."),
    "min_el_deg_out_of_range": ("min_el_deg", "Minimum elevation angle must be from 0 to 90 degrees."),
    "targets_json_invalid": ("targets_json", "Targets JSON is malformed."),
    "invalid_targets_json": ("targets_json", "Targets JSON is malformed."),
    "targets_invalid": ("targets", "Targets must be a list."),
    "targets_not_list": ("targets", "Targets must be a list."),
    "target_invalid": ("targets", "Each target must be an object."),
    "target_not_object": ("targets", "Each target must be an object."),
    "target_id_invalid": ("target_id", "Target ID must not be blank."),
    "target_id_blank": ("target_id", "Target ID must not be blank."),
    "target_id_duplicate": ("target_id", "Target IDs must be unique."),
    "target_coordinates_invalid": ("targets", "Target coordinates must be finite numbers."),
    "target_lat_invalid": ("target_lat", "Target latitude is invalid."),
    "target_lat_out_of_range": ("target_lat", "Target latitude must be from -90 to 90 degrees."),
    "target_lon_invalid": ("target_lon", "Target longitude is invalid."),
    "target_lon_out_of_range": ("target_lon", "Target longitude must be from -180 to 180 degrees."),
    "target_altitude_unsupported": ("target_alt_km", "Target altitude must be 0 km in Phase 1."),
    "conflicting_targets_sources": ("targets", "Targets JSON and targets list disagree."),
    "targets_required": ("targets", "Earth-fixed pointing requires at least one target."),
    "earth_fixed_targets_required": ("targets", "Earth-fixed pointing requires at least one target."),
    "earth_fixed_analysis_mode_unsupported": ("analysis_mode", "Earth-fixed pointing is not supported for this analysis mode."),
    "earth_fixed_phased_array_unsupported": ("antenna_model", "Earth-fixed pointing is not supported with phased_array."),
    "debug_target_reachability_not_public": ("debug_target_reachability", "Target reachability diagnostics are not public."),
    "coverage_rf_null": ("rf", "Coverage RF configuration must be an object."),
    "coverage_rf_not_object": ("rf", "Coverage RF configuration must be an object."),
    "coverage_top_level_pointing_alias": ("rf", "Coverage earth-fixed pointing fields must be nested under rf."),
    "coverage_result_id_invalid": ("coverage_result_id", "Coverage result ID is invalid."),
    "configured_satellite_manifest_invalid": ("configured_satellite_manifest", "Configured satellite manifest is invalid."),
    "client_sat_id_invalid": ("client_sat_id", "Client satellite identity is invalid."),
    "scalar_ground_lat_missing": ("ground_lat", "Scalar point ground latitude is required."),
    "scalar_ground_lon_missing": ("ground_lon", "Scalar point ground longitude is required."),
    "scalar_ground_lat_invalid": ("ground_lat", "Scalar point ground latitude is invalid."),
    "scalar_ground_lon_invalid": ("ground_lon", "Scalar point ground longitude is invalid."),
    "scalar_ground_lat_out_of_range": ("ground_lat", "Scalar point ground latitude is out of range."),
    "scalar_ground_lon_out_of_range": ("ground_lon", "Scalar point ground longitude is out of range."),
    "point_satellite_identity_invalid": ("satellites", "Point Analysis satellites need stable unique IDs."),
    "point_time_collision": ("timestamp", "Point Analysis snapshots conflict at the same millisecond."),
    "coverage_time_required": ("coverage_time", "Coverage time is required."),
    "coverage_time_invalid": ("coverage_time", "Coverage time is invalid."),
    "highest_cn_atmosphere_unavailable": ("atmospheric_mode", "Atmospheric scoring is unavailable for highest C/N."),
    "highest_cn_clutter_unavailable": ("clutter_mode", "Clutter scoring is unavailable for highest C/N."),
    "highest_cn_adapter_error": ("assignment_rule", "Highest C/N scoring adapter failed."),
    "highest_cn_probe_unavailable": ("assignment_rule", "Highest C/N scoring probe is unavailable."),
    "highest_cn_runtime_loss_unavailable": ("assignment_rule", "A runtime loss source failed during highest C/N scoring."),
    "conflicting_coverage_assignment_chunk": ("coverage.assignment_table", "Stored Coverage chunks disagree for the same time and satellite."),
    "coverage_propagation_failed": ("satellites", "A configured satellite could not be propagated for this run."),
    "trajectory_time_collision": ("timestamp", "Trajectory samples conflict at the same millisecond."),
    "trajectory_sample_invalid": ("sample", "A trajectory sample has an invalid position or timestamp."),
}
EARTH_FIXED_VALIDATION_REGISTRY = {
    code: _validation_meta(field, "runtime" if code in {"trajectory_time_collision", "trajectory_sample_invalid", "coverage_propagation_failed", "highest_cn_runtime_loss_unavailable"} else "request", message)
    for code, (field, message) in _VALIDATION_MESSAGES.items()
}
EARTH_FIXED_VALIDATION_REGISTRY["conflicting_coverage_assignment_chunk"]["category"] = "cache_replay"
EARTH_FIXED_VALIDATION_MESSAGES = {code: meta["message"] for code, meta in EARTH_FIXED_VALIDATION_REGISTRY.items()}
EARTH_FIXED_VALIDATION_FIELDS = {code: meta["field"] for code, meta in EARTH_FIXED_VALIDATION_REGISTRY.items()}


def dedupe_preserving_order(codes):
    return list(dict.fromkeys(code for code in codes if code))


def validation_error_detail(code: str) -> dict:
    meta = EARTH_FIXED_VALIDATION_REGISTRY.get(code, {})
    return {"code": code, "message": meta.get("message", "Validation failed."), "field": meta.get("field"), "category": meta.get("category", "internal")}


def earth_fixed_validation_error_envelope(codes: list[str]) -> dict:
    return {"error": "earth_fixed_validation_failed", "errors": [validation_error_detail(code) for code in dedupe_preserving_order(codes)]}


class EarthFixedValidationError(ValueError):
    """Stable validation error for direct/in-process earth-fixed callers."""

    def __init__(self, codes: list[str] | tuple[str, ...]):
        self.codes = list(dict.fromkeys(codes))
        super().__init__(", ".join(self.codes))


class EarthFixedRuntimeError(RuntimeError):
    """A typed runtime availability failure during earth-fixed scoring."""

    def __init__(self, code: str, detail: str | None = None, *, message: str | None = None, field: str | None = None):
        self.code = code
        self.field = field
        self.detail = detail or message or code
        super().__init__(self.detail)


def earth_fixed_runtime_error_envelope(exc: EarthFixedRuntimeError) -> dict:
    meta = EARTH_FIXED_VALIDATION_REGISTRY.get(exc.code, {})
    return {"error": "earth_fixed_runtime_failed", "errors": [{
        "code": exc.code,
        "message": str(exc) if str(exc) != exc.code else meta.get("message", "Run failed."),
        "field": exc.field if exc.field is not None else meta.get("field"),
        "category": meta.get("category", "runtime"),
    }]}


def validate_earth_fixed_request(
    rf_params: dict,
    *,
    analysis_mode: str = "point",
    entry_point: str = "radio_engine",
    antenna_model: str | None = None,
    fallback_target: dict | None = None,
    candidate_loss_adapters: dict | None = None,
    candidate_loss_probe: tuple[dict, dict] | None = None,
    candidate_loss_probe_status: str = "not_applicable",
    probe_dynamic_adapters: bool = False,
) -> list[str]:
    """Return stable request errors for the Section 1 contract."""
    errors = list(rf_params.get("rf_param_errors", []))
    pointing_mode = rf_params.get("pointing_mode", "nadir")
    if pointing_mode not in {"nadir", "earth_fixed"}:
        errors.append("invalid_pointing_mode")
    if pointing_mode != "earth_fixed":
        return list(dict.fromkeys(errors))
    if analysis_mode in {"footprint", "multibeam"}:
        errors.append("earth_fixed_analysis_mode_unsupported")
    # [rev3: REV2-03] Earth-fixed Coverage through the UI CLI has no configured
    # manifest / client_sat_id / coverage_result_id wiring (deferred to Phase 2,
    # see plan section 13 item 0) and must be rejected at this boundary. The web
    # UI and the HTTP batch CLI (entry_point="coverage"/"batch") are unaffected.
    if entry_point == "ui_cli" and analysis_mode == "coverage":
        errors.append("earth_fixed_analysis_mode_unsupported")
    if str(antenna_model or rf_params.get("antenna_model", "")).lower() == "phased_array":
        errors.append("earth_fixed_phased_array_unsupported")
    if not rf_params.get("targets") and fallback_target is None:
        errors.append("earth_fixed_targets_required")
    if rf_params.get("assignment_rule") == "highest_cn":
        availability = get_highest_cn_availability(
            entry_point,
            rf_params,
            candidate_loss_adapters=candidate_loss_adapters,
            candidate_loss_probe=candidate_loss_probe,
            candidate_loss_probe_status=candidate_loss_probe_status,
            probe_dynamic_adapters=probe_dynamic_adapters,
        )
        for key, source in availability.items():
            if not source["available"]:
                # A no-work snapshot (no satellites / no reachable target) means a
                # dynamic loss source simply had nothing to probe — skip only those
                # probe-dependent sources (clutter). Atmosphere is STATICALLY
                # unsupported for highest C/N and must be rejected regardless of
                # work availability, so never suppress it here.
                if (
                    key != "atmospheric_loss"
                    and candidate_loss_probe_status in {"empty_satellites", "visible_no_reachable_target"}
                    and entry_point in {"coverage", "live_point_bulk"}
                ):
                    continue
                if not probe_dynamic_adapters and source.get("dynamic_probe_required"):
                    continue
                errors.append(source["error_code"])
    return dedupe_preserving_order(errors)


def get_highest_cn_availability(
    entry_point: str,
    rf_params: dict,
    *,
    candidate_loss_adapters: dict | None = None,
    candidate_loss_probe: tuple[dict, dict] | None = None,
    candidate_loss_probe_status: str = "not_applicable",
    probe_dynamic_adapters: bool = False,
) -> dict:
    """Describe whether optional candidate loss sources can score a target."""
    result = {}
    adapters = candidate_loss_adapters or {}
    for key, enabled, code in (
        ("atmospheric_loss", str(rf_params.get("atmospheric_mode", "disable")).lower() == "enable", "highest_cn_atmosphere_unavailable"),
        ("clutter_loss", bool(rf_params.get("clutter_enable")), "highest_cn_clutter_unavailable"),
    ):
        if not enabled:
            result[key] = {"enabled": False, "available": True, "source": "0 dB; disabled", "error_code": None, "dynamic_probe_required": False}
            continue
        adapter = adapters.get(key)
        # Atmosphere is not a Phase-1 highest-C/N candidate scorer. Clutter is
        # engine-local and may be constructed only after the portable preflight.
        static_unavailable = (
            entry_point in {"scalar_point", "live_point_bulk", "coverage"}
            and key == "atmospheric_loss"
        ) or entry_point == "scalar_point"
        available = False if static_unavailable else (adapter is not None or not probe_dynamic_adapters)
        if probe_dynamic_adapters and available:
            if adapter is None or candidate_loss_probe is None:
                available = False
            else:
                try:
                    loss, adapter_error = adapter(*candidate_loss_probe, rf_params)
                    available = adapter_error is None and loss is not None and math.isfinite(float(loss))
                except Exception:
                    available = False
        result[key] = {
            "enabled": True,
            "available": available,
            "source": (
                "candidate_loss_adapter" if probe_dynamic_adapters and available
                else "dynamic_validation_required" if available
                else "unavailable"
            ),
            "error_code": None if available else code,
            "dynamic_probe_required": bool(available and not probe_dynamic_adapters),
        }
    return result
