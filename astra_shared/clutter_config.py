"""Phase 2 interim clutter configuration helpers."""

from __future__ import annotations

from .clutter import ClutterConfig, ClutterModel
from .defaults import DEFAULT_CLUTTER_PERCENTILE

INTERIM_DISABLED_CLUTTER_CONFIG = ClutterConfig(ClutterModel.DISABLED, None)
INTERIM_WORLDCOVER_CLUTTER_CONFIG = ClutterConfig(
    ClutterModel.WORLDCOVER_P2108_P833,
    DEFAULT_CLUTTER_PERCENTILE,
)


def build_interim_clutter_config(rf_params: dict) -> ClutterConfig:
    """Build the Phase 2 ClutterConfig from the current RF payload fields.

    Phase 3 replaces this interim adapter with the versioned normalizer. Until
    then, this is the only place that interprets the current clutter_enable
    field for the new evaluator path.
    """
    if rf_params.get("clutter_enable") is not True:
        return INTERIM_DISABLED_CLUTTER_CONFIG
    return INTERIM_WORLDCOVER_CLUTTER_CONFIG
