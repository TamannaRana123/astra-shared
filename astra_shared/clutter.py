#!/usr/bin/env python3
"""
clutter.py

Clutter model helpers for spec sections 5.1, 5.3 and 5.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist
from typing import Callable, NamedTuple

import numpy as np

from .defaults import (
    CLUTTER_ELEV_FLOOR_DEG,
    DEFAULT_CLUTTER_PERCENTILE,
    P2108_F_MAX_GHZ,
    P2108_F_MIN_GHZ,
    P833_A,
    P833_E,
    P833_G,
    P_MIN_PCT,
    VALID_CLUTTER_CLASS_IDS,
)


class ClutterModel(str, Enum):
    """Spec section 5.4: clutter model selector for the new evaluator."""

    DISABLED = "disabled"
    WORLDCOVER_P2108_P833 = "worldcover_p2108_p833"


class ClutterBranch(str, Enum):
    """Spec section 5.4: branch selected by the WorldCover class table."""

    P2108 = "p2108"
    P833 = "p833"
    NONE = "none"


class LookupState(str, Enum):
    """Spec sections 5.4 and 5.5: WorldCover lookup state."""

    CLASS = "class"
    NO_DATA_PIXEL = "no_data_pixel"
    TILE_NOT_PUBLISHED = "tile_not_published"
    TILE_MISSING = "tile_missing"
    READ_FAILED = "read_failed"
    RASTERIO_UNAVAILABLE = "rasterio_unavailable"
    UNKNOWN_CLASS = "unknown_class"


class ClutterLookup(NamedTuple):
    """Spec section 5.5: lookup metadata passed to the evaluator."""

    lookup_state: LookupState
    class_id: int | None
    class_label: str


CACHEABLE_LOOKUP_STATES: frozenset[LookupState] = frozenset(
    {
        LookupState.CLASS,
        LookupState.NO_DATA_PIXEL,
        LookupState.TILE_NOT_PUBLISHED,
    }
)


CLUTTER_BRANCH_NONE: int = 0
CLUTTER_BRANCH_P2108: int = 1
CLUTTER_BRANCH_P833: int = 2
CLUTTER_BRANCH_TO_CODE: dict[ClutterBranch, int] = {
    ClutterBranch.NONE: CLUTTER_BRANCH_NONE,
    ClutterBranch.P2108: CLUTTER_BRANCH_P2108,
    ClutterBranch.P833: CLUTTER_BRANCH_P833,
}


@dataclass(frozen=True)
class ClutterConfig:
    """Spec section 5.4: model and percentile for the new evaluator."""

    model: ClutterModel = ClutterModel.WORLDCOVER_P2108_P833
    clutter_percentile: float = DEFAULT_CLUTTER_PERCENTILE

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", self.normalized_model())
        try:
            percentile = float(self.clutter_percentile)
        except (TypeError, ValueError) as exc:
            raise ValueError("p must satisfy 1e-300 <= p < 100") from exc
        if not (P_MIN_PCT <= percentile < 100.0):
            raise ValueError("p must satisfy 1e-300 <= p < 100")
        object.__setattr__(self, "clutter_percentile", percentile)

    def normalized_model(self) -> ClutterModel:
        if isinstance(self.model, ClutterModel):
            return self.model
        try:
            return ClutterModel(str(self.model))
        except ValueError as exc:
            raise ValueError(f"unsupported clutter model: {self.model}") from exc


class ClutterResult(NamedTuple):
    """Spec section 5.4: scalar clutter evaluation result."""

    loss_db: float
    class_id: int | None
    class_label: str
    branch: ClutterBranch
    lookup_state: LookupState


CLASS_TO_BRANCH: dict[int, ClutterBranch] = {
    10: ClutterBranch.P833,
    20: ClutterBranch.NONE,
    30: ClutterBranch.NONE,
    40: ClutterBranch.NONE,
    50: ClutterBranch.P2108,
    60: ClutterBranch.NONE,
    70: ClutterBranch.NONE,
    80: ClutterBranch.NONE,
    90: ClutterBranch.NONE,
    95: ClutterBranch.P833,
    100: ClutterBranch.NONE,
}

CLUTTER_CLASS_NONE: int = -1
CLUTTER_STATE_CLASS: int = 1
CLUTTER_STATE_NO_DATA_PIXEL: int = 2
CLUTTER_STATE_TILE_NOT_PUBLISHED: int = 3
CLUTTER_STATE_TILE_MISSING: int = 4
CLUTTER_STATE_READ_FAILED: int = 5
CLUTTER_STATE_RASTERIO_UNAVAILABLE: int = 6
CLUTTER_STATE_UNKNOWN_CLASS: int = 7
LOOKUP_STATE_TO_CODE: dict[LookupState, int] = {
    LookupState.CLASS: CLUTTER_STATE_CLASS,
    LookupState.NO_DATA_PIXEL: CLUTTER_STATE_NO_DATA_PIXEL,
    LookupState.TILE_NOT_PUBLISHED: CLUTTER_STATE_TILE_NOT_PUBLISHED,
    LookupState.TILE_MISSING: CLUTTER_STATE_TILE_MISSING,
    LookupState.READ_FAILED: CLUTTER_STATE_READ_FAILED,
    LookupState.RASTERIO_UNAVAILABLE: CLUTTER_STATE_RASTERIO_UNAVAILABLE,
    LookupState.UNKNOWN_CLASS: CLUTTER_STATE_UNKNOWN_CLASS,
}
CODE_TO_LOOKUP_STATE: dict[int, LookupState] = {
    code: state for state, code in LOOKUP_STATE_TO_CODE.items()
}


def _check_percentile_domain(p_pct: float) -> None:
    if not (P_MIN_PCT <= p_pct < 100.0):
        raise ValueError("p must satisfy 1e-300 <= p < 100")


def _check_clutter_domain(f_ghz: float, elev_deg: float, p_pct: float) -> None:
    _check_percentile_domain(p_pct)
    if not (P2108_F_MIN_GHZ <= f_ghz <= P2108_F_MAX_GHZ):
        raise ValueError("frequency outside 0.5-100 GHz")
    if not (0.0 <= elev_deg <= 90.0):
        raise ValueError("elevation outside 0-90 degrees")


def _check_array_clutter_domain(f_ghz: float, p_pct: float) -> None:
    _check_percentile_domain(p_pct)
    if not (P2108_F_MIN_GHZ <= f_ghz <= P2108_F_MAX_GHZ):
        raise ValueError("frequency outside 0.5-100 GHz")


def _p2108_core(f_ghz: float, elev_deg: float, p_pct: float) -> float:
    u = p_pct / 100.0
    k1 = 93.0 * (f_ghz ** 0.175)
    ang = 0.05 * (1.0 - elev_deg / 90.0) + math.pi * elev_deg / 180.0
    base = -k1 * math.log1p(-u) / math.tan(ang)
    expo = 0.5 * (90.0 - elev_deg) / 90.0
    return base ** expo - 1.0 + 0.6 * NormalDist().inv_cdf(u)


def _p833_core(f_ghz: float, elev_deg: float, p_pct: float) -> float:
    f_mhz = f_ghz * 1000.0
    kh = 5.5 - 5.0 * p_pct / 100.0
    b_exp = (0.30281 - 0.003624 * kh) * (
        f_ghz ** (0.0013118 - 0.026236 * kh)
    )
    depth_m = _p833_depth_m(elev_deg, p_pct)
    return (
        P833_A
        * (f_mhz ** b_exp)
        * math.log10(depth_m)
        * ((elev_deg + P833_E) ** P833_G)
        - 4.0 * (p_pct / 100.0)
        + 0.4
    )


def _p833_depth_m(elev_deg: float, p_pct: float) -> float:
    return 243.0 * (p_pct / 100.0) * (elev_deg + 1.0) ** -0.93047 + 1.0


def clutter_loss_p2108(f_ghz: float, elev_deg: float, p_pct: float) -> float:
    """Spec section 5.1: guarded scalar P.2108 clutter loss."""

    _check_clutter_domain(f_ghz, elev_deg, p_pct)
    return _p2108_core(f_ghz, elev_deg, p_pct)


def clutter_loss_p833(f_ghz: float, elev_deg: float, p_pct: float) -> float:
    """Spec section 5.1: guarded scalar P.833 clutter loss."""

    _check_clutter_domain(f_ghz, elev_deg, p_pct)
    return _p833_core(f_ghz, elev_deg, p_pct)


def clutter_loss_p2108_arr(f_ghz, elev_deg, visible, p_pct) -> tuple[np.ndarray, np.ndarray]:
    """Spec section 5.3: array P.2108 helper with visibility mask."""

    _check_array_clutter_domain(f_ghz, p_pct)
    elev = np.asarray(elev_deg, dtype=np.float64)
    visible_arr = np.asarray(visible, dtype=bool)
    elev, visible_arr = np.broadcast_arrays(elev, visible_arr)
    mask = visible_arr & np.isfinite(elev)
    loss = np.zeros(elev.shape, dtype=np.float64)
    if np.any(mask):
        eval_elev = np.clip(elev[mask], CLUTTER_ELEV_FLOOR_DEG, 90.0)
        u = p_pct / 100.0
        k1 = 93.0 * (f_ghz ** 0.175)
        q_inv = NormalDist().inv_cdf(u)
        ang = 0.05 * (1.0 - eval_elev / 90.0) + np.pi * eval_elev / 180.0
        with np.errstate(under="ignore"):
            base = -k1 * math.log1p(-u) / np.tan(ang)
            expo = 0.5 * (90.0 - eval_elev) / 90.0
            loss[mask] = np.power(base, expo) - 1.0 + 0.6 * q_inv
    return loss, mask


def clutter_loss_p833_arr(f_ghz, elev_deg, visible, p_pct) -> tuple[np.ndarray, np.ndarray]:
    """Spec section 5.3: array P.833 helper with visibility mask."""

    _check_array_clutter_domain(f_ghz, p_pct)
    elev = np.asarray(elev_deg, dtype=np.float64)
    visible_arr = np.asarray(visible, dtype=bool)
    elev, visible_arr = np.broadcast_arrays(elev, visible_arr)
    mask = visible_arr & np.isfinite(elev)
    loss = np.zeros(elev.shape, dtype=np.float64)
    if np.any(mask):
        eval_elev = np.clip(elev[mask], CLUTTER_ELEV_FLOOR_DEG, 90.0)
        f_mhz = f_ghz * 1000.0
        kh = 5.5 - 5.0 * p_pct / 100.0
        b_exp = (0.30281 - 0.003624 * kh) * (
            f_ghz ** (0.0013118 - 0.026236 * kh)
        )
        depth_m = 243.0 * (p_pct / 100.0) * np.power(eval_elev + 1.0, -0.93047) + 1.0
        loss[mask] = (
            P833_A
            * (f_mhz ** b_exp)
            * np.log10(depth_m)
            * np.power(eval_elev + P833_E, P833_G)
            - 4.0 * (p_pct / 100.0)
            + 0.4
        )
    return loss, mask


def _branch_for_class(class_id: int | None) -> ClutterBranch:
    if class_id is None:
        return ClutterBranch.NONE
    return CLASS_TO_BRANCH.get(class_id, ClutterBranch.NONE)


def _branch_codes_for_class_arr(class_arr) -> np.ndarray:
    classes = np.asarray(class_arr)
    branches = np.full(classes.shape, CLUTTER_BRANCH_NONE, dtype=np.int8)
    for class_id, branch in CLASS_TO_BRANCH.items():
        branches[classes == class_id] = CLUTTER_BRANCH_TO_CODE[branch]
    return branches


def _lookup_state_value(state) -> str:
    if isinstance(state, LookupState):
        return state.value
    if isinstance(state, str):
        try:
            return LookupState(state).value
        except ValueError as exc:
            raise ValueError(f"unsupported lookup state: {state}") from exc
    raise ValueError(f"unsupported lookup state: {state}")


def _lookup_state_like(reference_state, value: str):
    del reference_state
    return LookupState(value)


def _coerce_lookup_for_eval(lookup):
    if all(hasattr(lookup, attr) for attr in ("class_id", "lookup_state", "class_label")):
        state = LookupState(_lookup_state_value(lookup.lookup_state))
        class_id = lookup.class_id
        if class_id is not None:
            class_id = int(class_id)
        return ClutterLookup(state, class_id, str(lookup.class_label))
    if isinstance(lookup, tuple) and len(lookup) == 3:
        state, class_id, label = lookup
        state = LookupState(_lookup_state_value(state))
        if class_id is not None:
            class_id = int(class_id)
        return ClutterLookup(state, class_id, str(label))
    raise TypeError("lookup must provide class_id, lookup_state, and class_label")


def _lookup_state_codes_arr(state_arr) -> np.ndarray:
    raw_states = np.asarray(state_arr)
    if np.issubdtype(raw_states.dtype, np.integer):
        raw_codes = raw_states.astype(np.int64, copy=False)
        valid = np.isin(raw_codes, np.fromiter(CODE_TO_LOOKUP_STATE, dtype=np.int64))
        if not np.all(valid):
            bad = raw_codes[~valid][0]
            raise ValueError(f"unsupported lookup state code: {int(bad)}")
        return raw_codes.astype(np.int16, copy=False)

    states = np.asarray(state_arr, dtype=object)
    if np.issubdtype(states.dtype, np.integer):
        return states.astype(np.int16, copy=False)
    encode = np.frompyfunc(
        lambda s: LOOKUP_STATE_TO_CODE[LookupState(_lookup_state_value(s))],
        1,
        1,
    )
    return encode(states).astype(np.int16)


def evaluate_clutter_loss(lookup, config: ClutterConfig, freq_hz: float, elevation_deg: float) -> ClutterResult:
    """Spec section 5.4: scalar WorldCover clutter evaluator."""

    lookup_result = _coerce_lookup_for_eval(lookup)
    model = config.normalized_model()

    lookup_state = lookup_result.lookup_state
    class_id = lookup_result.class_id
    if lookup_state == LookupState.CLASS and class_id not in VALID_CLUTTER_CLASS_IDS:
        lookup_state = _lookup_state_like(lookup_state, LookupState.UNKNOWN_CLASS.value)

    if model == ClutterModel.DISABLED:
        return ClutterResult(0.0, class_id, lookup_result.class_label, ClutterBranch.NONE, lookup_state)

    f_ghz = float(freq_hz) / 1.0e9
    if not (P2108_F_MIN_GHZ <= f_ghz <= P2108_F_MAX_GHZ):
        return ClutterResult(0.0, class_id, lookup_result.class_label, ClutterBranch.NONE, lookup_state)

    raw_elev = float(elevation_deg)
    if not np.isfinite(raw_elev) or raw_elev > 90.0:
        raise ValueError("elevation outside 0-90 degrees")

    branch = _branch_for_class(class_id) if lookup_state == LookupState.CLASS else ClutterBranch.NONE
    eval_elev = max(raw_elev, CLUTTER_ELEV_FLOOR_DEG)
    if branch == ClutterBranch.P2108:
        loss_db = clutter_loss_p2108(f_ghz, eval_elev, config.clutter_percentile)
    elif branch == ClutterBranch.P833:
        loss_db = clutter_loss_p833(f_ghz, eval_elev, config.clutter_percentile)
    else:
        loss_db = 0.0

    return ClutterResult(float(loss_db), class_id, lookup_result.class_label, branch, lookup_state)


def evaluate_clutter_arr(class_arr, state_arr, elev_arr, visible_arr, config: ClutterConfig, freq_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Spec sections 5.3 and 5.4: vector clutter evaluator."""

    elev = np.asarray(elev_arr, dtype=np.float64)
    visible = np.asarray(visible_arr, dtype=bool)
    classes = np.asarray(class_arr)
    raw_states = np.asarray(state_arr)
    states = raw_states if np.issubdtype(raw_states.dtype, np.integer) else np.asarray(state_arr, dtype=object)
    elev, visible, classes, states = np.broadcast_arrays(elev, visible, classes, states)
    mask = visible & np.isfinite(elev)
    loss = np.zeros(elev.shape, dtype=np.float64)

    _check_percentile_domain(config.clutter_percentile)
    if config.normalized_model() == ClutterModel.DISABLED:
        return loss, mask

    f_ghz = float(freq_hz) / 1.0e9
    if not (P2108_F_MIN_GHZ <= f_ghz <= P2108_F_MAX_GHZ):
        return loss, mask
    _check_array_clutter_domain(f_ghz, config.clutter_percentile)

    state_codes = _lookup_state_codes_arr(states)
    class_mask = state_codes == CLUTTER_STATE_CLASS
    branch_codes = _branch_codes_for_class_arr(classes)
    p2108_idx = mask & class_mask & (branch_codes == CLUTTER_BRANCH_P2108)
    p833_idx = mask & class_mask & (branch_codes == CLUTTER_BRANCH_P833)

    if np.any(p2108_idx):
        vals, _ = clutter_loss_p2108_arr(
            f_ghz,
            elev[p2108_idx],
            np.ones(np.count_nonzero(p2108_idx), dtype=bool),
            config.clutter_percentile,
        )
        loss[p2108_idx] = vals
    if np.any(p833_idx):
        vals, _ = clutter_loss_p833_arr(
            f_ghz,
            elev[p833_idx],
            np.ones(np.count_nonzero(p833_idx), dtype=bool),
            config.clutter_percentile,
        )
        loss[p833_idx] = vals
    return loss, mask


def make_clutter_arr_evaluator(
    config: ClutterConfig,
    freq_hz: float,
) -> Callable[[object, object, object, object], tuple[np.ndarray, np.ndarray]]:
    """Spec section 8 item 1: bind run-level config and frequency."""
    model = config.normalized_model()
    bound_config = ClutterConfig(model=model, clutter_percentile=config.clutter_percentile)
    bound_freq_hz = float(freq_hz)

    def _evaluate(class_arr, state_arr, elev_arr, visible_arr) -> tuple[np.ndarray, np.ndarray]:
        return evaluate_clutter_arr(
            class_arr,
            state_arr,
            elev_arr,
            visible_arr,
            bound_config,
            bound_freq_hz,
        )

    return _evaluate
