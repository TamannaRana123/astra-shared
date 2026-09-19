from astra_shared.coverage_identity import (
    build_fixed_manifest,
    build_walker_manifest,
    canonical_coverage_cell_key,
    canonical_coverage_sat_id,
)


def test_coverage_ids_and_manifests_are_deterministic():
    assert canonical_coverage_sat_id({"id": "  SAT-1  "}, source="static") == "SAT-1"
    assert canonical_coverage_cell_key({"row": 2, "col": 4}) == (2, 4)
    assert build_fixed_manifest(1, 2, 550)[0]["id"] == "fixed-0"
    first = build_walker_manifest(2, 3, 550, 53, 30, 0, "delta")
    second = build_walker_manifest(2, 3, 550, 53, 30, 0, "delta")
    assert first == second
    assert len({row["id"] for row in first}) == 6
