"""Stable Point Analysis satellite identity."""

from .coverage_identity import _stable_text


def canonical_point_sat_id(
    sat: dict,
    *,
    source: str,
    order_index: int,
    legacy_unique_names: set[str] | None = None,
) -> str:
    for key in ("canonical_sat_id", "id", "catalog_id", "norad_id"):
        value = sat.get(key)
        if value not in (None, ""):
            return _stable_text(value, field=key)
    name = sat.get("name")
    if name not in (None, ""):
        candidate = _stable_text(name, field="name")
        if legacy_unique_names is not None and candidate in legacy_unique_names:
            return candidate
    source_key = str(source).strip().lower()
    if source_key == "rf_log":
        return f"rf-log-sat-{order_index}"
    if source_key == "scalar":
        return "scalar-point-sat-0"
    raise ValueError("point_satellite_identity_invalid")


def prepare_point_satellite_ids(satellites: list[dict], *, source: str) -> list[dict]:
    names = {}
    for sat in satellites:
        name = sat.get("name")
        if name not in (None, ""):
            names.setdefault(str(name).strip(), 0)
            names[str(name).strip()] += 1
    unique_names = {name for name, count in names.items() if count == 1}
    result = []
    ids = set()
    for index, original in enumerate(satellites):
        sat = dict(original)
        sat_id = canonical_point_sat_id(sat, source=source, order_index=index, legacy_unique_names=unique_names)
        if sat_id in ids:
            raise ValueError("point_satellite_identity_invalid")
        ids.add(sat_id)
        sat["id"] = sat_id
        result.append(sat)
    return result
