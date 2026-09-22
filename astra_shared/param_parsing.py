"""Shared RF parameter parsing helpers used by all Astra services."""

import json
import logging
import math

from .custom_antenna_schema import normalize_custom_antenna
from .defaults import (
    ADDITIONAL_LOSSES_DB_MAX,
    ADDITIONAL_LOSSES_DB_MIN,
    CLUTTER_LOSS_DB_MAX,
    CLUTTER_LOSS_DB_MIN,
    DEFAULT_BANDWIDTH_HZ,
    DEFAULT_CODE_RATE,
    DEFAULT_COMPUTE_PFD,
    DEFAULT_EIRP_DBW,
    DEFAULT_MODULATION,
    DEFAULT_PFD_LIMIT_BAND,
    DEFAULT_PFD_REF_BW_HZ,
    DEFAULT_RX_GAIN_DBI,
    DEFAULT_SYSTEM_NOISE_TEMP_K,
    POLARIZATION_LOSS_DB_MAX,
    POLARIZATION_LOSS_DB_MIN,
    VALID_CLUTTER_CLASS_IDS,
)

logger = logging.getLogger(__name__)

MODULATIONS = {"BPSK", "QPSK", "OQPSK", "8PSK", "16QAM", "64QAM"}
VALID_CODE_RATES = (
    1.0 / 4.0,
    1.0 / 3.0,
    2.0 / 5.0,
    1.0 / 2.0,
    3.0 / 5.0,
    2.0 / 3.0,
    3.0 / 4.0,
    4.0 / 5.0,
    5.0 / 6.0,
    8.0 / 9.0,
    9.0 / 10.0,
    1.0,
)
VALID_CODE_RATE_LABELS = "1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10, 1"
CUSTOM_PFD_LIMIT_MIN_DBW_M2 = -200.0
CUSTOM_PFD_LIMIT_MAX_DBW_M2 = 0.0
PFD_LIMIT_PRESETS = {
    "S-2500-2690-FSS": {"l0": -136.0, "l25": -125.0, "ref_bw_hz": 1.0e6},
    "C-3400-4200-GSO": {"l0": -152.0, "l25": -142.0, "ref_bw_hz": 4.0e3},
    "C-4500-4800-FSS": {"l0": -152.0, "l25": -142.0, "ref_bw_hz": 4.0e3},
    "C-5150-5216-FSS": {"l0": -164.0, "l25": -164.0, "ref_bw_hz": 4.0e3},
    "C-6700-6825-FSS": {"l0": -137.0, "l25": -127.0, "ref_bw_hz": 1.0e6},
    "C-6825-7075-FSS": {
        "conjunctive": [
            {"l0": -154.0, "l25": -144.0, "ref_bw_hz": 4.0e3},
            {"l0": -134.0, "l25": -124.0, "ref_bw_hz": 1.0e6},
        ],
    },
    "X-7250-7900-FSS": {"l0": -152.0, "l25": -142.0, "ref_bw_hz": 4.0e3},
    "Ku-10700-11700-GSO": {"l0": -150.0, "l25": -140.0, "ref_bw_hz": 4.0e3},
    "Ku-10700-11700-NGSO-normal": {"l0": -126.0, "l25": -116.0, "ref_bw_hz": 1.0e6},
    "Ka-17700-19300-GSO-or-old-NGSO": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Ka-19300-19700-FSS": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Ka-27500-27501-FSS": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Q-37500-40000-NGSO": {
        "l0": -120.0,
        "l25": -105.0,
        "ref_bw_hz": 1.0e6,
        "slope": 0.75,
    },
    "Q-37500-40000-GSO": {"ref_bw_hz": 1.0e6, "shape": "q_gso_127"},
    "Q-40000-40500-FSS": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Q-40500-42000-NGSO": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Q-40500-42000-GSO": {"ref_bw_hz": 1.0e6, "shape": "q_gso_120"},
    "Q-42000-42500-NGSO": {
        "l0": -120.0,
        "l25": -105.0,
        "ref_bw_hz": 1.0e6,
        "slope": 0.75,
    },
    "Q-42000-42500-GSO": {"ref_bw_hz": 1.0e6, "shape": "q_gso_127"},
}


def _get_float(
    params: dict,
    key: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Parse a float parameter with optional clamping."""
    raw_value = params.get(key, default)
    if raw_value in (None, ""):
        raw_value = default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def _get_int(
    params: dict,
    key: str,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Parse an integer parameter with optional clamping."""
    raw_value = params.get(key, default)
    if raw_value in (None, ""):
        raw_value = default
    try:
        value = int(float(raw_value))
    except (TypeError, ValueError):
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def _get_str(params: dict, key: str, default: str) -> str:
    """Parse a string parameter."""
    raw_value = params.get(key, default)
    if raw_value is None:
        raw_value = default
    return str(raw_value).strip()


def _parse_clutter_values(params: dict) -> dict[int, float] | None:
    """Parse user-supplied clutter loss overrides per WorldCover class."""
    raw = params.get("clutter_values")
    if raw is None or raw == "" or raw == "null":
        return None

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    if not isinstance(raw, dict):
        return None

    if len(raw) > 15:
        return None

    result: dict[int, float] = {}
    for key, val in raw.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            continue
        if class_id not in VALID_CLUTTER_CLASS_IDS:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fval):
            continue
        result[class_id] = max(CLUTTER_LOSS_DB_MIN, min(CLUTTER_LOSS_DB_MAX, fval))

    return result if result else None


def _parse_clutter_fallback(params: dict) -> float | None:
    """Parse user-supplied clutter fallback value (dB), clamped to CLUTTER_LOSS_DB_MIN..MAX."""
    raw = params.get("clutter_fallback")
    if raw is None or raw == "" or raw == "null":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return max(CLUTTER_LOSS_DB_MIN, min(CLUTTER_LOSS_DB_MAX, val))


def _parse_clutter_enable(params: dict) -> bool:
    """Parse clutter enable from 'clutter_mode' (string) or 'clutter_enable' (bool)."""
    if "clutter_mode" in params:
        return _get_str(params, "clutter_mode", "disable").lower() == "enable"
    if "clutter_enable" in params:
        val = params["clutter_enable"]
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "enable")
    return False


def _parse_eirp(params: dict) -> float:
    """Parse EIRP, supporting both new (eirp_dbw) and legacy (tx_power + tx_gain) formats."""
    if "eirp" in params:
        return _get_float(params, "eirp", DEFAULT_EIRP_DBW)
    if "eirp_dbw" in params:
        return _get_float(params, "eirp_dbw", DEFAULT_EIRP_DBW)
    if "tx_power" in params and "tx_gain" in params:
        return _get_float(params, "tx_power", 40.0) + _get_float(
            params, "tx_gain", 30.0
        )
    if "tx_power_dbw" in params and "tx_gain_dbi" in params:
        return _get_float(params, "tx_power_dbw", 40.0) + _get_float(
            params, "tx_gain_dbi", 30.0
        )
    return DEFAULT_EIRP_DBW


def _parse_custom_antenna_payload(params: dict) -> dict:
    raw = params.get("custom_antenna")
    if raw is None or raw == "" or raw == "null":
        return normalize_custom_antenna(None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return normalize_custom_antenna(None)
    return normalize_custom_antenna(raw)


def _normalize_antenna_model(value: str | None) -> str:
    model = str(value or "gaussian").strip().lower()
    allowed = {"gaussian", "bessel", "itu_s672", "phased_array", "custom"}
    return model if model in allowed else "gaussian"


def _parse_modulation(params: dict) -> str:
    modulation = str(params.get("modulation") or DEFAULT_MODULATION).strip().upper()
    if modulation not in MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return modulation


def _parse_data_rate_bps(params: dict) -> float | None:
    raw = params.get("data_rate_bps")
    if raw in (None, "", "null"):
        return None
    value = float(raw)
    if value <= 0.0:
        raise ValueError("data_rate_bps must be greater than 0")
    return value


def _parse_code_rate(params: dict) -> float:
    raw = params.get("code_rate", DEFAULT_CODE_RATE)
    if raw in (None, "", "null"):
        return DEFAULT_CODE_RATE
    value = float(raw)
    if not any(
        math.isclose(value, allowed, rel_tol=0.0, abs_tol=1.0e-9)
        for allowed in VALID_CODE_RATES
    ):
        raise ValueError(f"code_rate must be one of: {VALID_CODE_RATE_LABELS}")
    return value


def _parse_bool(params: dict, key: str, default: bool) -> bool:
    raw = params.get(key, default)
    if raw in (None, ""):
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    return str(raw).strip().lower() in ("true", "1", "yes", "on", "enable", "enabled")


def _parse_optional_float(params: dict, key: str) -> float | None:
    raw = params.get(key)
    if raw in (None, "", "null"):
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _parse_pfd_limit_band(params: dict) -> str | None:
    raw = params.get("pfd_limit_band", DEFAULT_PFD_LIMIT_BAND)
    if raw in (None, "", "null", "none"):
        return None
    band = str(raw).strip()
    if band.lower() == "none":
        return None
    if band not in PFD_LIMIT_PRESETS and band != "custom":
        raise ValueError(f"Unsupported PFD limit band: {raw}")
    return band


def _validate_custom_pfd_limit(value: float, key: str) -> float:
    if value < CUSTOM_PFD_LIMIT_MIN_DBW_M2 or value > CUSTOM_PFD_LIMIT_MAX_DBW_M2:
        raise ValueError(
            f"{key} must be between {CUSTOM_PFD_LIMIT_MIN_DBW_M2:g} "
            f"and {CUSTOM_PFD_LIMIT_MAX_DBW_M2:g} dBW/m^2"
        )
    return value


def _parse_pfd_params(
    params: dict,
) -> tuple[bool, str | None, float | None, float | None, float]:
    compute_pfd = _parse_bool(params, "compute_pfd", DEFAULT_COMPUTE_PFD)
    if not compute_pfd:
        return compute_pfd, None, None, None, DEFAULT_PFD_REF_BW_HZ

    pfd_limit_band = _parse_pfd_limit_band(params)
    pfd_ref_bw_hz = _parse_optional_float(params, "pfd_ref_bw_hz")
    if pfd_ref_bw_hz is None:
        pfd_ref_bw_hz = DEFAULT_PFD_REF_BW_HZ
    if pfd_ref_bw_hz <= 0.0:
        raise ValueError("pfd_ref_bw_hz must be greater than 0")

    if pfd_limit_band in PFD_LIMIT_PRESETS:
        preset = PFD_LIMIT_PRESETS[pfd_limit_band]
        if "conjunctive" in preset:
            preset = preset["conjunctive"][0]
        if "shape" in preset:
            return compute_pfd, pfd_limit_band, None, None, preset["ref_bw_hz"]
        return (
            compute_pfd,
            pfd_limit_band,
            preset["l0"],
            preset["l25"],
            preset["ref_bw_hz"],
        )

    if pfd_limit_band == "custom":
        pfd_l0_dbw_m2 = _parse_optional_float(params, "pfd_l0_dbw_m2")
        pfd_l25_dbw_m2 = _parse_optional_float(params, "pfd_l25_dbw_m2")
        if pfd_l0_dbw_m2 is None or pfd_l25_dbw_m2 is None:
            raise ValueError(
                "custom PFD limit requires pfd_l0_dbw_m2 and pfd_l25_dbw_m2"
            )
        pfd_l0_dbw_m2 = _validate_custom_pfd_limit(pfd_l0_dbw_m2, "pfd_l0_dbw_m2")
        pfd_l25_dbw_m2 = _validate_custom_pfd_limit(pfd_l25_dbw_m2, "pfd_l25_dbw_m2")
        return compute_pfd, pfd_limit_band, pfd_l0_dbw_m2, pfd_l25_dbw_m2, pfd_ref_bw_hz

    return compute_pfd, None, None, None, pfd_ref_bw_hz


# =============================================================================
# Unified RF Parameter Parsing
# =============================================================================


def parse_rf_params(params: dict) -> dict:
    """Parse RF parameters from any source into a canonical dict.

    Accepts form args, config.json, project files, or HTTP request bodies.
    Callers use the subset they need � unused keys are harmless.
    """
    if params.get("frequency_ghz") not in (None, "") or params.get("frequency") not in (
        None,
        "",
    ):
        freq_ghz = _get_float(
            params,
            "frequency_ghz",
            _get_float(params, "frequency", 12.0),
            min_val=0.001,
        )
        freq_hz = freq_ghz * 1e9
    else:
        freq_hz = _get_float(
            params, "freq_hz", _get_float(params, "frequency_hz", 12.0e9), min_val=1.0e6
        )

    wavelength_m = 3.0e8 / freq_hz
    aperture_radius_m_raw = params.get("aperture_radius_m")
    if aperture_radius_m_raw in (None, ""):
        aperture_radius_wl = _get_float(params, "aperture_radius_wl", 10.0, min_val=1.0)
        aperture_radius_m = aperture_radius_wl * wavelength_m
    else:
        aperture_radius_m = _get_float(
            params, "aperture_radius_m", 10.0 * wavelength_m, min_val=0.0
        )
        aperture_radius_wl = aperture_radius_m / wavelength_m
    system_noise_temp_k = _get_float(
        params,
        "system_noise_temp_k",
        DEFAULT_SYSTEM_NOISE_TEMP_K,
        min_val=10.0,
        max_val=10000.0,
    )

    bw_mhz_raw = params.get("bandwidth_mhz")
    bw_hz_raw = params.get("bandwidth_hz")
    bandwidth_hz = None
    explicit_bw = False
    if bw_mhz_raw not in (None, "", "null"):
        explicit_bw = True
        try:
            bandwidth_hz = float(bw_mhz_raw) * 1e6
        except (TypeError, ValueError):
            bandwidth_hz = None
    elif bw_hz_raw not in (None, "", "null"):
        explicit_bw = True
        try:
            bandwidth_hz = float(bw_hz_raw)
        except (TypeError, ValueError):
            bandwidth_hz = None

    if bandwidth_hz is not None and bandwidth_hz <= 0:
        bandwidth_hz = None

    if bandwidth_hz is None and not explicit_bw:
        bandwidth_hz = DEFAULT_BANDWIDTH_HZ

    logger.debug(
        "[RF] noise_temp_k=%s bandwidth_hz=%s cn_enabled=%s",
        system_noise_temp_k,
        bandwidth_hz,
        bandwidth_hz is not None,
    )

    antenna_model = _normalize_antenna_model(
        _get_str(params, "antenna_model", "gaussian")
    )

    modulation = _parse_modulation(params)
    data_rate_bps = _parse_data_rate_bps(params)
    code_rate = _parse_code_rate(params)
    compute_pfd, pfd_limit_band, pfd_l0_dbw_m2, pfd_l25_dbw_m2, pfd_ref_bw_hz = (
        _parse_pfd_params(params)
    )

    return {
        "eirp_dbw": _parse_eirp(params),
        "rx_gain_dbi": _get_float(
            params, "rx_gain_dbi", _get_float(params, "rx_gain", DEFAULT_RX_GAIN_DBI)
        ),
        "freq_hz": freq_hz,
        "antenna_model": antenna_model,
        "custom_antenna": _parse_custom_antenna_payload(params),
        "beamwidth_deg": _get_float(
            params, "beamwidth_deg", _get_float(params, "beamwidth", 4.5), min_val=0.1
        ),
        "aperture_radius_wl": aperture_radius_wl,
        "aperture_radius_m": aperture_radius_m,
        "max_gain_dbi": _get_float(
            params, "max_gain_dbi", 30.0, min_val=10.0, max_val=60.0
        ),
        "ln_db": _get_float(params, "ln_db", -20.0),
        "ellipticity_ratio": _get_float(
            params, "ellipticity_ratio", 1.0, min_val=1.0, max_val=3.0
        ),
        "num_elements_x": _get_int(params, "num_elements_x", 8, min_val=1, max_val=64),
        "num_elements_y": _get_int(params, "num_elements_y", 8, min_val=1, max_val=64),
        "spacing_wl": _get_float(params, "spacing_wl", 0.5, min_val=0.1, max_val=2.0),
        "element_exponent": _get_float(
            params, "element_exponent", 1.3, min_val=0.0, max_val=3.0
        ),
        "clutter_enable": _parse_clutter_enable(params),
        "clutter_values": _parse_clutter_values(params),
        "clutter_fallback": _parse_clutter_fallback(params),
        "atmospheric_mode": _get_str(params, "atmospheric_mode", "disable").lower(),
        "availability_percent": _get_float(
            params, "availability_percent", 99.0, min_val=90.0, max_val=99.999
        ),
        "additional_losses_db": _get_float(
            params,
            "additional_losses_db",
            2.0,
            min_val=ADDITIONAL_LOSSES_DB_MIN,
            max_val=ADDITIONAL_LOSSES_DB_MAX,
        ),
        "polarization_loss_db": _get_float(
            params,
            "polarization_loss_db",
            0.0,
            min_val=POLARIZATION_LOSS_DB_MIN,
            max_val=POLARIZATION_LOSS_DB_MAX,
        ),
        "system_noise_temp_k": system_noise_temp_k,
        "bandwidth_hz": bandwidth_hz,
        "modulation": modulation,
        "data_rate_bps": data_rate_bps,
        "code_rate": code_rate,
        "compute_pfd": compute_pfd,
        "pfd_limit_band": pfd_limit_band,
        "pfd_ref_bw_hz": pfd_ref_bw_hz,
        "pfd_l0_dbw_m2": pfd_l0_dbw_m2,
        "pfd_l25_dbw_m2": pfd_l25_dbw_m2,
        "min_el_deg": _get_float(params, "min_el_deg", 5.0, min_val=0.0, max_val=90.0),
    }
