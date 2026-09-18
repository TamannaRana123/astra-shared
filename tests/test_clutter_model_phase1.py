#!/usr/bin/env python3

import math
from statistics import NormalDist
from unittest.mock import patch

import numpy as np
import pytest

from astra_shared.defaults import (
    CLUTTER_ELEV_FLOOR_DEG,
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


# NTIA/ITS reference vectors for P.2108-1 Annex 1 section 3.3: (f GHz, elevation
# deg, p %, published L_ces dB).  Source: github.com/NTIA/p2108-test-data at
# e46db673, AeronauticalStatisticalModelTestData.csv.  NTIA publishes these to
# one decimal place and tests its own C++ against them at 0.1 dB.
P2108_REFERENCE_VECTORS = [
    (30.0, 2.0, 5.0, 7.7),
    (30.0, 2.0, 1.0, 1.9),
    (30.0, 2.0, 99.0, 87.3),
    (10.0, 10.5, 45.0, 12.4),
    (15.0, 90.0, 50.0, 0.0),
    (20.0, 0.0, 50.0, 45.6),
    (11.1, 15.5, 80.5, 14.7),
]


def _ntia_q_inverse_abramowitz_stegun(q):
    """NTIA's InverseComplementaryCumulativeDistribution, transcribed verbatim.

    github.com/NTIA/p2108 v1.1 (eadb58f),
    src/InverseComplementaryCumulativeDistribution.cpp: Abramowitz & Stegun
    26.2.23, |error| < 4.5e-4.
    """
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = 1.0 - q if q > 0.5 else q
    t = math.sqrt(-2.0 * math.log(x))
    zeta = ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)
    return -(t - zeta) if q > 0.5 else (t - zeta)


def _ntia_q_inverse_exact(q):
    """Q^-1(q) = Phi^-1(1 - q), computed exactly."""
    return NormalDist().inv_cdf(1.0 - q)


def _ntia_aeronautical_model(f_ghz, theta_deg, p, q_inverse):
    """NTIA's AeronauticalStatisticalModel, transcribed verbatim.

    github.com/NTIA/p2108 v1.1 (eadb58f), src/AeronauticalStatisticalModel.cpp.
    Written the way NTIA writes it, not the way section 5.1 does: log(1 - p),
    cot, and a subtracted Q^-1(p) rather than an added Phi^-1(p).  That is the
    point.  Agreement shows section 5.1's stable rearrangement is the same
    equation, which comparing Astra with a snapshot of its own output cannot.
    """
    k_1 = 93 * f_ghz**0.175
    part1 = math.log(1 - p / 100.0)
    part2 = 0.05 * (1 - theta_deg / 90.0) + math.pi * theta_deg / 180.0
    part3 = 0.5 * (90.0 - theta_deg) / 90.0
    part4 = 0.6 * q_inverse(p / 100)
    return (-k_1 * part1 * (1 / math.tan(part2))) ** part3 - 1 - part4


@pytest.mark.parametrize(("f_ghz", "elev_deg", "p_pct", "published"), P2108_REFERENCE_VECTORS)
def test_p2108_matches_ntia_equation_exactly(f_ghz, elev_deg, p_pct, published):
    """Same equation as NTIA's code, with the inverse normal made exact: 1e-9."""
    reference = _ntia_aeronautical_model(f_ghz, elev_deg, p_pct, _ntia_q_inverse_exact)
    assert clutter_loss_p2108(f_ghz, elev_deg, p_pct) == pytest.approx(reference, abs=1e-9)


@pytest.mark.parametrize(("f_ghz", "elev_deg", "p_pct", "published"), P2108_REFERENCE_VECTORS)
def test_p2108_matches_ntia_code_within_its_approximation(f_ghz, elev_deg, p_pct, published):
    """NTIA's code verbatim, approximation included.

    The two can never agree to 1e-6: NTIA's inverse normal is accurate to
    4.5e-4, which reaches L_ces through the 0.6 factor as up to 2.7e-4 dB, and
    the measured gap on these vectors is 2.6e-4 dB.  3e-4 is that bound, not a
    fitted tolerance.
    """
    ntia = _ntia_aeronautical_model(f_ghz, elev_deg, p_pct, _ntia_q_inverse_abramowitz_stegun)
    assert clutter_loss_p2108(f_ghz, elev_deg, p_pct) == pytest.approx(ntia, abs=3e-4)


@pytest.mark.parametrize(("f_ghz", "elev_deg", "p_pct", "published"), P2108_REFERENCE_VECTORS)
def test_p2108_matches_ntia_published_values(f_ghz, elev_deg, p_pct, published):
    assert clutter_loss_p2108(f_ghz, elev_deg, p_pct) == pytest.approx(published, abs=0.06)


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

    # Spec section 9: canopy is monotone in p for every f >= 1.2 GHz, across the
    # whole 0.5-100 GHz window -- not only near the 1.2 GHz edge.
    for freq in (1.2, 2.0, 10.0, 20.0, 30.0, 60.0, 100.0):
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


@pytest.mark.parametrize("below_horizon", [-999.0, -30.0, -0.5, -1e-9])
def test_scalar_evaluator_refuses_rather_than_clamps_below_the_horizon(below_horizon):
    """Spec section 4.5: mask first, then clamp -- never clamp in place of masking.

    The scalar evaluator has no mask, so a negative angle means its caller
    skipped the visibility check.  Clamping it to the 5 deg floor would report
    22.56 dB of built-up clutter at 20 GHz for a satellite that is not in view,
    which is the same number a real 5 deg link gets, so nothing downstream
    could tell them apart.  An earlier version of these tests asserted that
    -30 deg returned a finite value: it pinned the defect.
    """
    built = ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up")
    with pytest.raises(ValueError, match="elevation outside 0-90 degrees"):
        evaluate_clutter_loss(built, ClutterConfig(), 20e9, below_horizon)


def test_scalar_evaluator_admits_the_whole_visible_range_and_floors_it():
    """The must-pass half: 0 and 90 are admitted, and 0 takes the 5 deg floor."""
    built = ClutterLookup(lookup_state=LookupState.CLASS, class_id=50, class_label="Built-up")
    cfg = ClutterConfig()
    at_zero = evaluate_clutter_loss(built, cfg, 20e9, 0.0).loss_db
    assert at_zero == pytest.approx(clutter_loss_p2108(20.0, CLUTTER_ELEV_FLOOR_DEG, 50.0), abs=1e-12)
    assert at_zero == pytest.approx(22.557, abs=1e-3)
    assert evaluate_clutter_loss(built, cfg, 20e9, 90.0).loss_db == pytest.approx(
        clutter_loss_p2108(20.0, 90.0, 50.0), abs=1e-12
    )


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
    """The evaluator's own percentile check, not ClutterConfig's.

    ClutterConfig rejects 150 at construction, so building one with 150 would
    raise before the evaluator ran and this test would pass without reaching
    the check it is named for.  The config is made valid and then corrupted,
    so the only thing that can raise is evaluate_clutter_arr.
    """
    cfg = ClutterConfig(clutter_percentile=50.0)
    object.__setattr__(cfg, "clutter_percentile", 150.0)
    with pytest.raises(ValueError, match="1e-300"):
        evaluate_clutter_arr(
            np.array([50], dtype=object),
            np.array([LookupState.CLASS], dtype=object),
            np.array([20.0]),
            np.array([True]),
            cfg,
            120_000_000_000.0,
        )


# ----------------------------------------------------------------------------
# The configured percentile has to reach the arithmetic.  Every check below uses
# p away from 50 and elevations away from the zenith, because those are the two
# places a wrong percentile is invisible: at p = 50, p and 100 - p give the same
# answer, and at 90 deg the base term is raised to the power zero.
# ----------------------------------------------------------------------------

_OFF_MEDIAN = (1.0, 20.0, 80.0, 99.0)
_OFF_ZENITH = np.array([5.0, 12.5, 30.0, 60.0, 85.0])


@pytest.mark.parametrize("p_pct", _OFF_MEDIAN)
@pytest.mark.parametrize("f_ghz", (2.0, 20.0, 60.0))
@pytest.mark.parametrize(
    ("array_helper", "scalar_helper"),
    [
        (clutter_loss_p2108_arr, clutter_loss_p2108),
        (clutter_loss_p833_arr, clutter_loss_p833),
    ],
)
def test_array_helpers_match_scalar_away_from_the_median(array_helper, scalar_helper, f_ghz, p_pct):
    loss, mask = array_helper(f_ghz, _OFF_ZENITH, np.ones(_OFF_ZENITH.size, dtype=bool), p_pct)
    assert mask.all()
    expected = [scalar_helper(f_ghz, float(e), p_pct) for e in _OFF_ZENITH]
    assert loss.tolist() == pytest.approx(expected, abs=1e-9)


def test_the_percentile_changes_the_answer_at_every_checked_point():
    """Guard the guard: the points above must actually be sensitive to p."""
    for helper in (clutter_loss_p2108, clutter_loss_p833):
        for e in _OFF_ZENITH:
            at_median = helper(20.0, float(e), 50.0)
            for p in _OFF_MEDIAN:
                assert abs(helper(20.0, float(e), p) - at_median) > 0.05, (helper.__name__, e, p)


@pytest.mark.parametrize("p_pct", _OFF_MEDIAN)
@pytest.mark.parametrize(
    ("class_id", "helper"), [(50, clutter_loss_p2108), (10, clutter_loss_p833), (95, clutter_loss_p833)]
)
def test_scalar_evaluator_uses_the_configured_percentile(class_id, helper, p_pct):
    lookup = ClutterLookup(LookupState.CLASS, class_id, "x")
    result = evaluate_clutter_loss(lookup, ClutterConfig(clutter_percentile=p_pct), 20e9, 20.0)
    assert result.loss_db == pytest.approx(helper(20.0, 20.0, p_pct), abs=1e-12)


@pytest.mark.parametrize("p_pct", _OFF_MEDIAN)
def test_array_evaluator_uses_the_configured_percentile(p_pct):
    classes = [50, 10, 95, 30]
    states = [LookupState.CLASS] * 4
    elev = [20.0, 20.0, 45.0, 20.0]
    visible = [True] * 4
    expected = [
        clutter_loss_p2108(20.0, 20.0, p_pct),
        clutter_loss_p833(20.0, 20.0, p_pct),
        clutter_loss_p833(20.0, 45.0, p_pct),
        0.0,
    ]
    cfg = ClutterConfig(clutter_percentile=p_pct)
    direct, _ = evaluate_clutter_arr(classes, states, elev, visible, cfg, 20e9)
    bound, _ = make_clutter_arr_evaluator(cfg, 20e9)(classes, states, elev, visible)
    assert direct.tolist() == pytest.approx(expected, abs=1e-12)
    assert bound.tolist() == pytest.approx(expected, abs=1e-12)


def test_two_percentiles_at_one_coordinate_each_get_their_own_value():
    """Spec section 9 cache tests: the percentile is per run, not per coordinate."""
    lookup = ClutterLookup(LookupState.CLASS, 50, "Built-up")
    low = evaluate_clutter_loss(lookup, ClutterConfig(clutter_percentile=20.0), 20e9, 20.0).loss_db
    high = evaluate_clutter_loss(lookup, ClutterConfig(clutter_percentile=80.0), 20e9, 20.0).loss_db
    again = evaluate_clutter_loss(lookup, ClutterConfig(clutter_percentile=20.0), 20e9, 20.0).loss_db
    assert high > low
    assert again == low


# ----------------------------------------------------------------------------
# Spec section 5.4: the frequency window short-circuits at BOTH ends, in both
# forms.  Outside it the evaluator returns 0 dB without calling a helper, which
# would otherwise raise ValueError from inside the RF path.
# ----------------------------------------------------------------------------

_OUTSIDE_WINDOW_HZ = (0.3e9, 0.4999e9, 100.0001e9, 120e9)


@pytest.mark.parametrize("freq_hz", _OUTSIDE_WINDOW_HZ)
@pytest.mark.parametrize("class_id", [50, 10, 95])
def test_scalar_evaluator_short_circuits_outside_the_window(class_id, freq_hz):
    result = evaluate_clutter_loss(ClutterLookup(LookupState.CLASS, class_id, "x"), ClutterConfig(), freq_hz, 20.0)
    assert result.loss_db == 0.0
    assert result.branch == ClutterBranch.NONE
    assert result.class_id == class_id
    assert result.lookup_state == LookupState.CLASS


@pytest.mark.parametrize("freq_hz", _OUTSIDE_WINDOW_HZ)
def test_array_evaluator_short_circuits_outside_the_window(freq_hz):
    loss, mask = evaluate_clutter_arr(
        [50, 10, 95], [LookupState.CLASS] * 3, [20.0, 20.0, 20.0], [True] * 3, ClutterConfig(), freq_hz
    )
    assert loss.tolist() == [0.0, 0.0, 0.0]
    assert mask.tolist() == [True, True, True]


@pytest.mark.parametrize("freq_hz", (0.5e9, 100.0e9))
def test_both_window_edges_are_inside(freq_hz):
    """The must-pass half: 0.5 and 100 GHz are evaluated, not short-circuited."""
    f_ghz = freq_hz / 1e9
    result = evaluate_clutter_loss(ClutterLookup(LookupState.CLASS, 50, "x"), ClutterConfig(), freq_hz, 20.0)
    assert result.branch == ClutterBranch.P2108
    assert result.loss_db == pytest.approx(clutter_loss_p2108(f_ghz, 20.0, 50.0), abs=1e-12)
    loss, _ = evaluate_clutter_arr([50], [LookupState.CLASS], [20.0], [True], ClutterConfig(), freq_hz)
    assert loss[0] == pytest.approx(clutter_loss_p2108(f_ghz, 20.0, 50.0), abs=1e-12)


@pytest.mark.parametrize("freq_hz", (0.3e9, 120e9))
@pytest.mark.parametrize("elevation", (-30.0, float("nan")))
def test_below_horizon_is_refused_outside_the_window_too(freq_hz, elevation):
    """A skipped visibility check is the caller's bug at any frequency."""
    with pytest.raises(ValueError, match="elevation outside 0-90 degrees"):
        evaluate_clutter_loss(ClutterLookup(LookupState.CLASS, 50, "x"), ClutterConfig(), freq_hz, elevation)


@pytest.mark.parametrize("class_id", [50.9, 10.5, "50x"])
def test_a_non_integral_class_code_is_unknown_not_truncated(class_id):
    result = evaluate_clutter_loss(ClutterLookup(LookupState.CLASS, class_id, "x"), ClutterConfig(), 20e9, 20.0)
    assert result.lookup_state == LookupState.UNKNOWN_CLASS
    assert result.branch == ClutterBranch.NONE
    assert result.loss_db == 0.0


def test_an_integral_float_class_code_is_still_that_class():
    result = evaluate_clutter_loss(ClutterLookup(LookupState.CLASS, 50.0, "x"), ClutterConfig(), 20e9, 20.0)
    assert result.class_id == 50
    assert result.branch == ClutterBranch.P2108


def test_disabled_config_carries_no_percentile():
    """Spec section 5.4 / section 8 item 8: the normalizer builds ClutterConfig("disabled", None)."""
    assert ClutterConfig("disabled", None).clutter_percentile is None
    assert ClutterConfig(ClutterModel.DISABLED).clutter_percentile is None
    # not live, so not validated and not carried
    assert ClutterConfig("disabled", 150.0).clutter_percentile is None
    with pytest.raises(ValueError, match="requires a percentile"):
        ClutterConfig("worldcover_p2108_p833", None)

    loss, mask = evaluate_clutter_arr(
        [50, 10], [LookupState.CLASS, LookupState.CLASS], [20.0, -30.0], [True, False],
        ClutterConfig("disabled", None), 20e9,
    )
    assert loss.tolist() == [0.0, 0.0]
    assert mask.tolist() == [True, False]
    evaluator = make_clutter_arr_evaluator(ClutterConfig("disabled", None), 20e9)
    assert evaluator([50], [LookupState.CLASS], [20.0], [True])[0].tolist() == [0.0]


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
