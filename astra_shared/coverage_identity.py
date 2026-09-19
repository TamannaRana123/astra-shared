"""Shared stable identity helpers for Earth-fixed Coverage."""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

GEO_ALTITUDE_KM = 35786.0
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _stable_text(value: Any, *, field: str) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{field} must be a scalar string")
    text = str(value).strip()
    if not text or len(text) > 128 or _CONTROL_RE.search(text):
        raise ValueError(f"{field} must be nonblank and free of control characters")
    return text


def canonical_coverage_sat_id(
    sat: dict | None = None,
    *,
    source: str = "runtime_snapshot",
    order_index: int = 0,
    slot_index: int | None = None,
    configured_id: Any = None,
) -> str:
    """Return one stable Coverage satellite id without changing internal spelling."""
    row = sat or {}
    for key in ("canonical_sat_id", "id", "catalog_id", "norad_id", "client_sat_id"):
        value = configured_id if key == "id" and configured_id is not None else row.get(key)
        if value not in (None, ""):
            return _stable_text(value, field=key)
    source_key = str(source).strip().lower().replace("_", "-")
    if source_key in {"geo", "geo-manual"}:
        return f"geo-slot-{slot_index if slot_index is not None else order_index}"
    if source_key == "walker":
        plane = row.get("plane_index", 0)
        slot = row.get("sat_index", slot_index if slot_index is not None else order_index)
        return f"walker-plane-{int(plane)}-sat-{int(slot)}"
    if source_key == "fixed":
        return f"fixed-{order_index}"
    if source_key in {"static", "static-json"}:
        return f"static-json-{order_index}"
    return f"runtime-snapshot-{order_index}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_orbit_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def hash_text(value: Any) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()


def epoch_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def inc_or_0(value: Any) -> float:
    return float(value or 0.0)


def build_tle_manifest(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": canonical_coverage_sat_id(row, source="tle", order_index=index),
            "order_index": index,
            "tle_line1_hash": hash_text(row["line1"]),
            "tle_line2_hash": hash_text(row["line2"]),
        }
        for index, row in enumerate(rows)
    ]


def build_fixed_manifest(lat: Any, lon: Any, alt_km: Any, epoch: Any = None) -> list[dict]:
    return [{"id": "fixed-0", "order_index": 0, "orbit_hash": canonical_orbit_hash({"mode": "fixed", "lat": float(lat), "lon": float(lon), "alt_km": float(alt_km), "epoch_iso": epoch_iso(epoch)})}]


def build_geo_manifest(rows: list[dict], epoch: Any = None) -> list[dict]:
    normalized_epoch = epoch_iso(epoch)
    return [
        {"id": canonical_coverage_sat_id(row, source="geo", slot_index=index, order_index=index), "order_index": index, "orbit_hash": canonical_orbit_hash({"mode": "geo", "slot_index": index, "lon": float(row.get("lon", 0.0)), "inc": inc_or_0(row.get("inc")), "alt_km": GEO_ALTITUDE_KM, "epoch_iso": normalized_epoch})}
        for index, row in enumerate(rows)
    ]


def build_walker_manifest(planes: int, sats_per_plane: int, alt_km: Any, inc_deg: Any, raan_spacing_deg: Any, phase_offset_deg: Any, pattern: str, epoch: Any = None) -> list[dict]:
    normalized_epoch = epoch_iso(epoch)
    rows = []
    for plane in range(int(planes)):
        for slot in range(int(sats_per_plane)):
            rows.append({"id": canonical_coverage_sat_id({"plane_index": plane, "sat_index": slot}, source="walker", order_index=plane * int(sats_per_plane) + slot), "order_index": plane * int(sats_per_plane) + slot, "orbit_hash": canonical_orbit_hash({"mode": "walker", "planes": int(planes), "sats_per_plane": int(sats_per_plane), "alt_km": float(alt_km), "inc_deg": float(inc_deg), "raan_spacing_deg": float(raan_spacing_deg), "phase_offset_deg": float(phase_offset_deg), "pattern": pattern, "epoch_iso": normalized_epoch, "plane_index": plane, "sat_index": slot})})
    return rows


def canonical_coverage_cell_key(point: dict, bounds: dict | None = None) -> tuple[int, int]:
    row, col = point.get("row"), point.get("col")
    if isinstance(row, bool) or isinstance(col, bool) or not isinstance(row, int) or not isinstance(col, int):
        raise ValueError("Coverage cell requires integer row and col")
    if bounds:
        lat_steps = bounds.get("lat_steps", bounds.get("latSteps"))
        lon_steps = bounds.get("lon_steps", bounds.get("lonSteps"))
        if lat_steps is None or lon_steps is None:
            raise ValueError("Coverage bounds require lat/lon step counts")
        if row < 0 or col < 0 or row >= int(lat_steps) or col >= int(lon_steps):
            raise ValueError("Coverage cell is outside grid bounds")
    return row, col


def validate_unique_coverage_ids(satellites: list[dict], *, source: str) -> list[str]:
    ids = [canonical_coverage_sat_id(sat, source=source, order_index=index) for index, sat in enumerate(satellites)]
    if len(ids) != len(set(ids)):
        raise ValueError("configured_satellite_manifest_invalid")
    return ids
