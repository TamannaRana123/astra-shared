"""
defaults.py

Shared configuration constants for all Astra services.
Covers RF defaults, WorldCover clutter configuration, and atmospheric model settings.
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# RF Link Budget Defaults
# =============================================================================

DEFAULT_EIRP_DBW: float = 70.0
DEFAULT_RX_GAIN_DBI: float = 0.0
DEFAULT_SYSTEM_NOISE_TEMP_K: float = 290.0
DEFAULT_BANDWIDTH_MHZ: float = 10.0
DEFAULT_BANDWIDTH_HZ: float = DEFAULT_BANDWIDTH_MHZ * 1e6
ADDITIONAL_LOSSES_DB_MIN: float = 0.0
ADDITIONAL_LOSSES_DB_MAX: float = 20.0
POLARIZATION_LOSS_DB_MIN: float = 0.0
POLARIZATION_LOSS_DB_MAX: float = 3.0
CLUTTER_LOSS_DB_MIN: float = 0.0
CLUTTER_LOSS_DB_MAX: float = 20.0
BOLTZMANN_DB: float = 228.6
K_BOLTZMANN_LINEAR: float = 1.380649e-23
DEFAULT_MODULATION = "QPSK"
DEFAULT_CODE_RATE = 1.0
DEFAULT_COMPUTE_PFD = True
DEFAULT_PFD_LIMIT_BAND = None
DEFAULT_PFD_REF_BW_HZ = 1.0e6

# =============================================================================
# ITU-R P.618 Atmospheric Model Defaults
# =============================================================================

DEFAULT_ATMOSPHERIC_LOSS_ENABLED: bool = False
DEFAULT_AVAILABILITY_PERCENT: float = 99.0

ATMOSPHERIC_IMPACT_NOTES: dict[str, str] = {
    "L-band": "Minimal atmospheric loss (<0.5 dB)",
    "S-band": "Minimal atmospheric loss (<0.5 dB)",
    "X-band": "Moderate rain attenuation (1-5 dB)",
    "Ku-band": "Significant rain attenuation (5-15 dB)",
    "Ka-band": "High rain attenuation (10-30 dB)",
    "Q/V-band": "Very high rain attenuation (>20 dB)",
}

# =============================================================================
# WorldCover Clutter Configuration
# =============================================================================

# Model-based clutter constants. P_MIN_PCT is the helper lower bound; the
# CLUTTER_PERCENTILE_INPUT_* values are for the later config normalizer.
P_MIN_PCT: float = 1.0e-300
CLUTTER_PERCENTILE_INPUT_MIN: float = 0.001
CLUTTER_PERCENTILE_INPUT_MAX: float = 99.999
DEFAULT_CLUTTER_PERCENTILE: float = 50.0
P2108_F_MIN_GHZ: float = 0.5
P2108_F_VALID_GHZ: float = 10.0
P2108_F_MAX_GHZ: float = 100.0
CLUTTER_ELEV_FLOOR_DEG: float = 5.0
P833_A: float = 1.87
P833_E: float = 0.01
P833_G: float = -0.12

# Tile directory — override via WORLDCOVER_DIR env var in container deployments
WORLDCOVER_DIR: Path = Path(
    os.environ.get(
        "WORLDCOVER_DIR", str(Path(__file__).parent.parent / "data/worldcover")
    )
)

# S3 base URL for downloading ESA WorldCover tiles on demand
WORLDCOVER_S3_BASE: str = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
)

CLUTTER_LOSS_DB: dict[int, float] = {
    10: 3.0,  # Tree cover
    20: 2.0,  # Shrubland
    30: 1.0,  # Grassland
    40: 1.5,  # Cropland
    50: 8.0,  # Built-up
    60: 0.5,  # Bare / sparse vegetation
    70: 0.5,  # Snow & ice
    80: 0.5,  # Permanent water bodies
    90: 2.0,  # Herbaceous wetland
    95: 0.0,  # Mangroves
    100: 0.0,  # Moss & lichen
}
CLUTTER_FALLBACK_DB: float = 0.0

# Human-readable labels for WorldCover land cover classes
CLUTTER_CLASS_LABELS: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow & ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss & lichen",
}

# Valid WorldCover class IDs for the model-routing layer.
VALID_CLUTTER_CLASS_IDS: frozenset[int] = frozenset(CLUTTER_CLASS_LABELS.keys())

# =============================================================================
# Flask Server Configuration
# =============================================================================

# Debug mode for Flask engines (radio, constellation).
# False by default — debug mode enables the Werkzeug file-system reloader which
# scans all Python files every ~2s, consuming 50%+ of engine CPU for nothing.
# Enable via environment variable: ASTRA_DEBUG=1
# Or change this constant directly for local development builds.
FLASK_DEBUG_MODE: bool = os.environ.get("ASTRA_DEBUG", "0") == "1"
