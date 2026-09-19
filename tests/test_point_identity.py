from astra_shared.point_identity import canonical_point_sat_id, prepare_point_satellite_ids


def test_point_identity_is_canonical_and_unique():
    assert canonical_point_sat_id({"canonical_sat_id": " sat-1 "}, source="tle", order_index=0) == "sat-1"
    rows = prepare_point_satellite_ids([{"name": "A"}, {"name": "B"}], source="static")
    assert [row["id"] for row in rows] == ["A", "B"]
    assert all("canonical_sat_id" not in row for row in rows)
