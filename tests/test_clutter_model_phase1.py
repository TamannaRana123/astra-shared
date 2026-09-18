#!/usr/bin/env python3

from unittest.mock import patch

import numpy as np
import pytest

from astra_shared.defaults import (
    CLUTTER_LOSS_DB,
    P2108_F_MAX_GHZ,
    P2108_F_MIN_GHZ,
    P2108_F_VALID_GHZ,
    P_MIN_PCT,
)
from astra_shared.clutter import (
    _branch_codes_for_class_arr,
    _p2108_core,
    _p833_core,
    _p833_depth_m,
    CLASS_TO_BRANCH,
    CLUTTER_BRANCH_NONE,
    CLUTTER_BRANCH_P2108,
    CLUTTER_BRANCH_P833,
    CLUTTER_CLASS_NONE,
    ClutterBranch,
    ClutterConfig,
    ClutterModel,
    clutter_loss_p2108,
    clutter_loss_p2108_arr,
    clutter_loss_p833,
    clutter_loss_p833_arr,
    evaluate_clutter_arr,
    evaluate_clutter_loss,
    make_clutter_arr_evaluator,
)
from astra_shared.geo_math import compute_elevation, compute_elevation_vec
from astra_shared.worldcover import (
    _CLUTTER_CACHE,
    _coerce_lookup,
    ClutterLookup,
    LOOKUP_STATE_TO_CODE,
    LookupState,
    clear_clutter_cache,
    clutter_loss_and_class,
    clutter_loss_db,
    lookup_clutter_arr,
    lookup_worldcover_class,
)


P2108_REFERENCE_VECTORS = [
    # Spec section 9 P.2108-1 Annex 1 section 3 reference-vector fixture:
    # (f GHz, elevation deg, p %, full-precision oracle, NTIA published).
    # The final column is retained as the NTIA one-decimal cross-check at 0.06 dB.
    (30.0, 2.0, 5.0, 7.652178444829, 7.7),
    (30.0, 2.0, 1.0, 1.948907219274, 1.9),
    (30.0, 2.0, 99.0, 87.277157073233, 87.3),
    (10.0, 10.5, 45.0, 12.375262758749, 12.4),
    (15.0, 90.0, 50.0, 0.0, 0.0),
    (20.0, 0.0, 50.0, 45.647488422852, 45.6),
    (11.1, 15.5, 80.5, 14.729641323502, 14.7),
]


@pytest.mark.parametrize(
    ("f_ghz", "elev_deg", "p_pct", "expected_full_precision", "published"),
    P2108_REFERENCE_VECTORS,
)
def test_p2108_reference_vectors(
    f_ghz, elev_deg, p_pct, expected_full_precision, published
):
    actual = clutter_loss_p2108(f_ghz, elev_deg, p_pct)
    assert actual == pytest.approx(expected_full_precision, abs=1e-6)
    assert actual == pytest.approx(published, abs=0.06)


@pytest.mark.parametrize(
    ("f_ghz", "elev_deg", "p_pct", "expected"),
    [
        (12.0, 10.0, 50.0, 14.04),
        (20.0, 20.0, 50.0, 10.18),
        (30.0, 45.0, 50.0, 6.16),
        (20.0, 90.0, 50.0, 3.27),
    ],
)
def test_p833_reference_vectors(f_ghz, elev_deg, p_pct, expected):
    assert clutter_loss_p833(f_ghz, elev_deg, p_pct) == pytest.approx(
        expected, abs=0.01
    )


def test_p833_generated_depth_reference():
    # Spec section 9: generated canopy depth d(10 deg, 50%) = 14.05 m.
    assert _p833_depth_m(10.0, 50.0) == pytest.approx(14.05, abs=0.01)


@pytest.mark.parametrize("helper", [clutter_loss_p2108, clutter_loss_p833])
def test_scalar_helpers_reject_outside_helper_domain(helper):
    valid = helper(20.0, 20.0, 50.0)
    assert np.isfinite(valid)

    for args in [
        (P2108_F_MIN_GHZ - 0.001, 20.0, 50.0),
        (P2108_F_MAX_GHZ + 0.001, 20.0, 50.0),
        (20.0, -0.001, 50.0),
        (20.0, 90.001, 50.0),
        (20.0, 20.0, 100.0),
    ]:
        with pytest.raises(ValueError):
            helper(*args)


@pytest.mark.parametrize("helper", [clutter_loss_p2108, clutter_loss_p833])
def test_scalar_helpers_pass_at_helper_boundaries(helper):
    for args in [
        (P2108_F_MIN_GHZ, 20.0, 50.0),
        (P2108_F_MAX_GHZ, 20.0, 50.0),
        (20.0, 0.0, 50.0),
        (20.0, 90.0, 50.0),
        (20.0, 20.0, 0.001),
        (20.0, 20.0, 99.999),
        (20.0, 20.0, P_MIN_PCT),
    ]:
        assert np.isfinite(helper(*args))


@pytest.mark.parametrize("helper", [clutter_loss_p2108, clutter_loss_p833])
def test_scalar_helpers_reject_subfloor_percentile_from_guard(helper):
    with pytest.raises(ValueError, match="1e-300"):
        helper(20.0, 20.0, 5e-324)


def test_tiny_percentile_uses_stable_p2108_arithmetic():
    assert clutter_loss_p2108(20.0, 20.0, 1e-15) == pytest.approx(
        -6.0962735, abs=1e-6
    )
    assert np.isfinite(clutter_loss_p2108(20.0, 20.0, 1e-12))
    assert np.isfinite(clutter_loss_p2108(20.0, 20.0, P_MIN_PCT))
    assert np.isfinite(clutter_loss_p833(20.0, 20.0, 1e-15))
    assert np.isfinite(clutter_loss_p833(20.0, 20.0, 1e-12))
    assert np.isfinite(clutter_loss_p833(20.0, 20.0, P_MIN_PCT))
    for helper in (clutter_loss_p2108_arr, clutter_loss_p833_arr):
        for p_pct in (1e-15, 1e-12, P_MIN_PCT):
            p_loss, p_mask = helper(
                20.0,
                np.array([20.0, 20.0, 20.0]),
                np.array([True, True, True]),
                p_pct,
            )
            assert np.array_equal(p_mask, np.array([True, True, True]))
            assert np.all(np.isfinite(p_loss))


def test_frequency_contract_includes_valid_notice_threshold():
    assert P2108_F_MIN_GHZ == 0.5
    assert P2108_F_VALID_GHZ == 10.0
    assert P2108_F_MAX_GHZ == 100.0


def test_p2108_no_clamp_helper_reference():
    assert clutter_loss_p2108(20.0, 20.0, 1.0) == pytest.approx(
        -0.7043,
        abs=0.001,
    )


def test_p2108_zenith_is_percentile_term_not_always_zero():
    assert clutter_loss_p2108(20.0, 90.0, 50.0) == pytest.approx(0.0, abs=1e-12)
    assert clutter_loss_p2108(20.0, 90.0, 80.0) == pytest.approx(0.50497, abs=1e-5)


def test_monotonicity_and_documented_canopy_zenith_reversal():
    p_values = np.array([0.001, 1.0, 5.0, 20.0, 50.0, 80.0, 99.0, 99.999])
    for freq in (0.5, 10.0, 20.0, 30.0, 100.0):
        for elev in (0.0, 5.0, 20.0, 45.0, 80.0, 90.0):
            values = [clutter_loss_p2108(freq, elev, float(p)) for p in p_values]
            assert values == sorted(values)

    for freq in (1.2, 2.0, 10.0):
        for elev in (0.0, 5.0, 20.0, 45.0, 80.0, 90.0):
            values = [clutter_loss_p833(freq, elev, float(p)) for p in p_values]
            assert values == sorted(values)

    reversal_values = [
        clutter_loss_p833(0.5, 90.0, float(p))
        for p in (20.0, 49.4, 80.0, 99.9)
    ]
    assert reversal_values[1] > reversal_values[0]
    assert reversal_values[1] > reversal_values[2]
    assert reversal_values[1] - reversal_values[3] == pytest.approx(0.536, abs=0.01)


def test_guarded_helpers_block_invalid_elevation_core_failures():
    assert _p2108_core(20.0, -1.0, 50.0) == pytest.approx(58.98, abs=0.01)
    assert isinstance(_p2108_core(20.0, -30.0, 50.0), complex)
    assert isinstance(_p833_core(20.0, -0.5, 50.0), complex)
    with pytest.raises(TypeError):
        _p833_core(20.0, -999.0, 50.0)
    with pytest.raises(ValueError):
        clutter_loss_p2108(20.0, -1.0, 50.0)
    with pytest.raises(ValueError):
        clutter_loss_p2108(20.0, -30.0, 50.0)
    with pytest.raises(ValueError):
        clutter_loss_p833(20.0, -0.5, 50.0)
    with pytest.raises(ValueError):
        clutter_loss_p833(20.0, -999.0, 50.0)


def test_array_helpers_mask_before_flooring_and_match_scalar():
    elev = np.array([20.0, 1.0, -30.0, -90.0, np.nan, np.inf, 90.0])
    visible = np.array([True, True, False, False, True, True, True])

    p2108_loss, p2108_mask = clutter_loss_p2108_arr(20.0, elev, visible, 50.0)
    p833_loss, p833_mask = clutter_loss_p833_arr(20.0, elev, visible, 50.0)

    expected_mask = visible & np.isfinite(elev)
    assert np.array_equal(p2108_mask, expected_mask)
    assert np.array_equal(p833_mask, expected_mask)
    assert np.all(np.isfinite(p2108_loss))
    assert np.all(np.isfinite(p833_loss))
    assert not np.iscomplexobj(p2108_loss)
    assert not np.iscomplexobj(p833_loss)
    assert p2108_loss[2] == 0.0
    assert p2108_loss[3] == 0.0
    assert p2108_loss[4] == 0.0
    assert p2108_loss[5] == 0.0
    assert p833_loss[2] == 0.0
    assert p833_loss[3] == 0.0
    assert p833_loss[4] == 0.0
    assert p833_loss[5] == 0.0
    assert p2108_loss[1] == pytest.approx(clutter_loss_p2108(20.0, 5.0, 50.0), abs=1e-9)
    assert p833_loss[1] == pytest.approx(clutter_loss_p833(20.0, 5.0, 50.0), abs=1e-9)
    assert p2108_loss[6] == pytest.approx(clutter_loss_p2108(20.0, 90.0, 50.0), abs=1e-9)
    assert p833_loss[6] == pytest.approx(clutter_loss_p833(20.0, 90.0, 50.0), abs=1e-9)


def test_mixed_class_array_routes_each_point_once():
    classes = np.array([50, 10, 95, 80, 999, 50, 50, 50, 50, 50, 50], dtype=object)
    states = np.array(
        [
            LookupState.CLASS,
            LookupState.CLASS,
            LookupState.CLASS,
            LookupState.CLASS,
            LookupState.CLASS,
            LookupState.NO_DATA_PIXEL,
            LookupState.TILE_NOT_PUBLISHED,
            LookupState.TILE_MISSING,
            LookupState.READ_FAILED,
            LookupState.RASTERIO_UNAVAILABLE,
            LookupState.UNKNOWN_CLASS,
        ],
        dtype=object,
    )
    elev = np.full(classes.shape, 20.0)
    visible = np.ones(classes.shape, dtype=bool)

    loss, mask = evaluate_clutter_arr(
        classes,
        states,
        elev,
        visible,
        ClutterConfig(),
        20_000_000_000.0,
    )

    assert np.array_equal(mask, visible)
    assert loss[0] == pytest.approx(clutter_loss_p2108(20.0, 20.0, 50.0))
    assert loss[1] == pytest.approx(clutter_loss_p833(20.0, 20.0, 50.0))
    assert loss[2] == pytest.approx(clutter_loss_p833(20.0, 20.0, 50.0))
    assert loss[3] == 0.0
    assert loss[4] == 0.0
    assert np.all(loss[5:] == 0.0)
    branch_codes = _branch_codes_for_class_arr(classes)
    assert branch_codes.dtype == np.int8
    assert branch_codes.tolist() == [
        CLUTTER_BRANCH_P2108,
        CLUTTER_BRANCH_P833,
        CLUTTER_BRANCH_P833,
        CLUTTER_BRANCH_NONE,
        CLUTTER_BRANCH_NONE,
        CLUTTER_BRANCH_P2108,
        CLUTTER_BRANCH_P2108,
        CLUTTER_BRANCH_P2108,
        CLUTTER_BRANCH_P2108,
        CLUTTER_BRANCH_P2108,
        CLUTTER_BRANCH_P2108,
    ]


def test_mixed_class_array_accepts_empty_grid():
    loss, mask = evaluate_clutter_arr(
        np.array([], dtype=object),
        np.array([], dtype=object),
        np.array([], dtype=float),
        np.array([], dtype=bool),
        ClutterConfig(),
        20_000_000_000.0,
    )
    assert loss.shape == (0,)
    assert mask.shape == (0,)


def test_array_evaluator_accepts_plain_lookup_state_list():
    loss, mask = evaluate_clutter_arr(
        [50, 10],
        [LookupState.CLASS, LookupState.TILE_MISSING],
        [30.0, 30.0],
        [True, True],
        ClutterConfig(),
        20_000_000_000.0,
    )

    assert mask.tolist() == [True, True]
    assert loss[0] == pytest.approx(clutter_loss_p2108(20.0, 30.0, 50.0))
    assert loss[1] == 0.0


def test_array_evaluator_rejects_unknown_state_values_and_codes():
    with pytest.raises(ValueError, match="unsupported lookup state"):
        evaluate_clutter_arr([50], ["bogus"], [20.0], [True], ClutterConfig(), 20e9)
    with pytest.raises(ValueError, match="unsupported lookup state code"):
        evaluate_clutter_arr([50], [40000], [20.0], [True], ClutterConfig(), 20e9)


def test_bound_array_evaluator_matches_direct_evaluator():
    classes = np.array([50, 10, 80], dtype=np.int16)
    states = np.array([LOOKUP_STATE_TO_CODE[LookupState.CLASS]] * 3, dtype=np.int16)
    elev = np.array([20.0, 20.0, 20.0])
    visible = np.array([True, True, True])
    cfg = ClutterConfig()

    direct_loss, direct_mask = evaluate_clutter_arr(
        classes,
        states,
        elev,
        visible,
        cfg,
        20_000_000_000.0,
    )
    bound_eval = make_clutter_arr_evaluator(cfg, 20_000_000_000.0)
    bound_loss, bound_mask = bound_eval(classes, states, elev, visible)

    assert np.array_equal(bound_mask, direct_mask)
    assert np.allclose(bound_loss, direct_loss)


def test_evaluator_routes_disabled_frequency_and_classes():
    cfg = ClutterConfig()
    built = ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up")
    tree = ClutterLookup(lookup_state=LookupState.CLASS, class_id=10, class_label="Tree cover")
    water = ClutterLookup(lookup_state=LookupState.CLASS, class_id=80, class_label="Permanent water bodies")
    missing = ClutterLookup(lookup_state=LookupState.TILE_MISSING, class_id=None, class_label="Unknown")

    assert evaluate_clutter_loss(built, ClutterConfig(ClutterModel.DISABLED), 20e9, 20.0).loss_db == 0.0
    assert evaluate_clutter_loss(built, cfg, 120e9, 20.0).loss_db == 0.0
    assert evaluate_clutter_loss(built, cfg, 20e9, 20.0).branch == ClutterBranch.P2108
    assert evaluate_clutter_loss(tree, cfg, 20e9, 20.0).branch == ClutterBranch.P833
    assert evaluate_clutter_loss(water, cfg, 20e9, 20.0).branch == ClutterBranch.NONE
    assert evaluate_clutter_loss(missing, cfg, 20e9, 20.0).loss_db == 0.0
    assert (
        evaluate_clutter_loss(
            ClutterLookup(lookup_state=LookupState.CLASS, class_id=999, class_label="Unknown (999)"),
            cfg,
            120e9,
            20.0,
        ).lookup_state
        == LookupState.UNKNOWN_CLASS
    )


def test_scalar_evaluator_routes_all_classes_and_non_class_states():
    cfg = ClutterConfig()
    expected_branches = {
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
    for class_id, branch in expected_branches.items():
        result = evaluate_clutter_loss(
            ClutterLookup(
                lookup_state=LookupState.CLASS,
                class_id=class_id,
                class_label=f"class {class_id}",
            ),
            cfg,
            20e9,
            20.0,
        )
        assert result.lookup_state == LookupState.CLASS
        assert result.branch == branch
        if branch == ClutterBranch.NONE:
            assert result.loss_db == 0.0
        else:
            assert result.loss_db > 0.0

    for state in (
        LookupState.NO_DATA_PIXEL,
        LookupState.TILE_NOT_PUBLISHED,
        LookupState.TILE_MISSING,
        LookupState.READ_FAILED,
        LookupState.RASTERIO_UNAVAILABLE,
        LookupState.UNKNOWN_CLASS,
    ):
        result = evaluate_clutter_loss(
            ClutterLookup(lookup_state=state, class_id=None, class_label="Unknown"),
            cfg,
            20e9,
            20.0,
        )
        assert result.lookup_state == state
        assert result.branch == ClutterBranch.NONE
        assert result.loss_db == 0.0


def test_scalar_evaluator_accepts_spec_order_tuple_lookup():
    result = evaluate_clutter_loss(
        (LookupState.CLASS, 50, "Built-up"),
        ClutterConfig(),
        20e9,
        20.0,
    )

    assert result.branch == ClutterBranch.P2108
    assert result.class_id == 50
    assert result.class_label == "Built-up"


def test_scalar_evaluator_rejects_swapped_tuple_and_unknown_state():
    with pytest.raises(ValueError, match="unsupported lookup state"):
        evaluate_clutter_loss((50, LookupState.CLASS, "Built-up"), ClutterConfig(), 20e9, 20.0)
    with pytest.raises(ValueError, match="unsupported lookup state"):
        evaluate_clutter_loss(ClutterLookup("bogus", 50, "Built-up"), ClutterConfig(), 20e9, 20.0)


def test_evaluator_does_not_hide_invalid_elevation_sentinels():
    built = ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up")
    cfg = ClutterConfig()
    with pytest.raises(ValueError):
        evaluate_clutter_loss(built, cfg, 20e9, np.nan)
    with pytest.raises(ValueError):
        evaluate_clutter_loss(built, cfg, 20e9, np.inf)
    with pytest.raises(ValueError):
        evaluate_clutter_loss(built, cfg, 20e9, 120.0)
    with pytest.raises(ValueError):
        evaluate_clutter_loss(built, cfg, 20e9, -np.inf)
    assert np.isfinite(evaluate_clutter_loss(built, cfg, 20e9, -30.0).loss_db)


def test_scalar_rejects_above_zenith_while_array_clips_grid_points():
    built = ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up")
    cfg = ClutterConfig(clutter_percentile=80.0)

    with pytest.raises(ValueError, match="elevation outside 0-90 degrees"):
        evaluate_clutter_loss(built, cfg, 20e9, 90.000001)

    loss, mask = evaluate_clutter_arr(
        [50, 50, 50],
        [LookupState.CLASS, LookupState.CLASS, LookupState.CLASS],
        [90.000001, np.nan, np.inf],
        [True, True, True],
        cfg,
        20e9,
    )
    assert mask.tolist() == [True, False, False]
    assert loss[0] == pytest.approx(clutter_loss_p2108(20.0, 90.0, 80.0), abs=1e-12)
    assert loss[1:].tolist() == [0.0, 0.0]


def test_array_evaluator_checks_percentile_outside_frequency_window():
    with pytest.raises(ValueError, match="1e-300"):
        evaluate_clutter_arr(
            np.array([50], dtype=object),
            np.array([LookupState.CLASS], dtype=object),
            np.array([20.0]),
            np.array([True]),
            ClutterConfig(clutter_percentile=150.0),
            120_000_000_000.0,
        )


def test_clutter_config_validates_percentile_at_construction():
    with pytest.raises(ValueError, match="1e-300"):
        ClutterConfig(clutter_percentile=100.0)
    with pytest.raises(ValueError, match="1e-300"):
        ClutterConfig(clutter_percentile="abc")
    assert ClutterConfig(clutter_percentile="50").clutter_percentile == 50.0


def test_unknown_class_is_derived_above_lookup():
    result = evaluate_clutter_loss(
        ClutterLookup(lookup_state=LookupState.CLASS, class_id=999, class_label="Unknown (999)"),
        ClutterConfig(),
        20e9,
        20.0,
    )
    assert result.lookup_state == LookupState.UNKNOWN_CLASS
    assert result.branch == ClutterBranch.NONE
    assert result.loss_db == 0.0


def test_lookup_clutter_arr_uses_metadata_cache_for_repeated_coordinates():
    clear_clutter_cache()
    with patch(
        "astra_shared.worldcover.fetch_worldcover_class",
        return_value=ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up"),
    ) as fetch:
        lat = np.array([1.0, 1.0, 1.0])
        lon = np.array([2.0, 2.0, 2.0])
        classes, states = lookup_clutter_arr(lat, lon)

    assert fetch.call_count == 1
    assert classes.tolist() == [50, 50, 50]
    assert classes.dtype == np.int16
    assert states.dtype == np.int16
    assert states.tolist() == [LOOKUP_STATE_TO_CODE[LookupState.CLASS]] * 3
    cached = next(iter(_CLUTTER_CACHE.values()))
    assert isinstance(cached, ClutterLookup)
    assert cached.class_id == 50


@pytest.mark.parametrize(
    ("state", "expected_cached"),
    [
        (LookupState.CLASS, True),
        (LookupState.NO_DATA_PIXEL, True),
        (LookupState.TILE_NOT_PUBLISHED, True),
        (LookupState.TILE_MISSING, False),
        (LookupState.READ_FAILED, False),
        (LookupState.RASTERIO_UNAVAILABLE, False),
    ],
)
def test_lookup_worldcover_class_cacheability_by_state(state, expected_cached):
    clear_clutter_cache()
    class_id = 50 if state == LookupState.CLASS else None
    lookup = ClutterLookup(
        lookup_state=state,
        class_id=class_id,
        class_label="Built-up" if class_id else "Unknown",
    )
    with patch("astra_shared.worldcover.fetch_worldcover_class", return_value=lookup):
        assert lookup_worldcover_class(1.0, 2.0) == lookup
    assert bool(_CLUTTER_CACHE) is expected_cached


def test_lookup_worldcover_class_recovers_after_uncached_failure():
    clear_clutter_cache()
    with patch(
        "astra_shared.worldcover.fetch_worldcover_class",
        side_effect=[
            ClutterLookup(lookup_state=LookupState.TILE_MISSING, class_id=None, class_label="Unknown"),
            ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up"),
        ],
    ) as fetch:
        first = lookup_worldcover_class(1.0, 2.0)
        second = lookup_worldcover_class(1.0, 2.0)

    assert first.lookup_state == LookupState.TILE_MISSING
    assert second.class_id == 50
    assert fetch.call_count == 2


def test_lookup_worldcover_class_recovers_after_uncached_read_failed():
    clear_clutter_cache()
    with patch(
        "astra_shared.worldcover.fetch_worldcover_class",
        side_effect=[
            ClutterLookup(lookup_state=LookupState.READ_FAILED, class_id=None, class_label="Unknown"),
            ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up"),
        ],
    ) as fetch:
        first = lookup_worldcover_class(1.0, 2.0)
        second = lookup_worldcover_class(1.0, 2.0)

    assert first.lookup_state == LookupState.READ_FAILED
    assert second.class_id == 50
    assert fetch.call_count == 2


def test_lookup_clutter_arr_uses_numeric_none_sentinel():
    clear_clutter_cache()
    with patch(
        "astra_shared.worldcover.fetch_worldcover_class",
        return_value=ClutterLookup(lookup_state=LookupState.TILE_MISSING, class_id=None, class_label="Unknown"),
    ):
        classes, states = lookup_clutter_arr(np.array([1.0]), np.array([2.0]))

    assert classes.tolist() == [CLUTTER_CLASS_NONE]
    assert states.tolist() == [LOOKUP_STATE_TO_CODE[LookupState.TILE_MISSING]]


def test_lookup_coercion_rejects_evidence_free_shapes():
    with pytest.raises(TypeError):
        _coerce_lookup(None)
    with pytest.raises(TypeError):
        _coerce_lookup(50)


def test_lookup_coercion_accepts_spec_order_tuple():
    lookup = _coerce_lookup((LookupState.CLASS, 50, "Built-up"))
    assert lookup.lookup_state == LookupState.CLASS
    assert lookup.class_id == 50
    assert lookup.class_label == "Built-up"


def test_table_wrappers_accept_custom_table_keywords_until_consumers_move():
    clear_clutter_cache()
    with patch(
        "astra_shared.worldcover.fetch_worldcover_class",
        return_value=ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up"),
    ):
        assert clutter_loss_and_class(
            1.0,
            2.0,
            loss_table={50: 12.0},
            fallback_db=7.0,
        ) == (12.0, "Built-up")
        assert clutter_loss_db(
            1.0,
            2.0,
            loss_table={50: 12.0},
            fallback_db=7.0,
        ) == 12.0


def test_table_wrappers_use_default_table_values_until_consumers_move():
    clear_clutter_cache()
    with patch(
        "astra_shared.worldcover.fetch_worldcover_class",
        return_value=ClutterLookup(lookup_state=LookupState.CLASS, class_id=10, class_label="Tree cover"),
    ):
        assert clutter_loss_db(1.0, 2.0) == CLUTTER_LOSS_DB[10]


def test_class_to_branch_table_covers_worldcover_classes():
    assert CLASS_TO_BRANCH[50] == ClutterBranch.P2108
    assert CLASS_TO_BRANCH[10] == ClutterBranch.P833
    assert CLASS_TO_BRANCH[95] == ClutterBranch.P833
    for class_id in (20, 30, 40, 60, 70, 80, 90, 100):
        assert CLASS_TO_BRANCH[class_id] == ClutterBranch.NONE


def test_shared_vector_elevation_matches_scalar_helper():
    obs_lat = np.array([12.0, 35.0, -10.0])
    obs_lon = np.array([77.0, 139.0, 20.0])
    sat_lat = np.array([13.0, 36.0, -8.0])
    sat_lon = np.array([78.0, 140.0, 22.0])
    sat_alt = np.array([600.0, 1200.0, 800.0])

    vector = compute_elevation_vec(obs_lat, obs_lon, sat_lat, sat_lon, sat_alt)
    scalar = np.array(
        [
            compute_elevation(
                float(obs_lat[i]),
                float(obs_lon[i]),
                float(sat_lat[i]),
                float(sat_lon[i]),
                float(sat_alt[i]),
            )
            for i in range(len(obs_lat))
        ]
    )
    assert vector == pytest.approx(scalar, abs=1e-12)


def test_shared_vector_elevation_covers_below_horizon_and_observer_altitude():
    obs_lat = np.array([0.0, 12.0])
    obs_lon = np.array([0.0, 77.0])
    sat_lat = np.array([0.0, 12.5])
    sat_lon = np.array([180.0, 77.5])
    sat_alt = np.array([600.0, 1200.0])
    obs_alt = np.array([0.0, 1.2])

    vector = compute_elevation_vec(obs_lat, obs_lon, sat_lat, sat_lon, sat_alt, obs_alt)
    scalar = np.array(
        [
            compute_elevation(
                float(obs_lat[i]),
                float(obs_lon[i]),
                float(sat_lat[i]),
                float(sat_lon[i]),
                float(sat_alt[i]),
                float(obs_alt[i]),
            )
            for i in range(len(obs_lat))
        ]
    )
    assert vector == pytest.approx(scalar, abs=1e-12)
    assert vector[0] < 0.0


def test_shared_vector_elevation_accepts_list_altitudes():
    vector = compute_elevation_vec([12.0], [77.0], [13.0], [78.0], [600.0], [0.0])
    scalar = compute_elevation(12.0, 77.0, 13.0, 78.0, 600.0, 0.0)
    assert vector == pytest.approx([scalar], abs=1e-12)


def test_shared_vector_elevation_degenerate_geometry_is_nan():
    vector = compute_elevation_vec(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    scalar = compute_elevation(0.0, 0.0, 0.0, 0.0, 0.0)
    assert scalar == 0.0
    assert np.isnan(vector[0])
