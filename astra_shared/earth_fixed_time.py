"""Shared millisecond-quantized time rules for Point and Coverage results."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_simulation_time_instant(value) -> tuple[datetime | None, str | None]:
    """Parse a time as UTC and truncate it to millisecond identity precision."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "coverage_time_invalid"
    try:
        if isinstance(value, datetime):
            instant = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            instant = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            text = str(value).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            instant = datetime.fromisoformat(text)
    except (TypeError, ValueError, OverflowError, OSError):
        return None, "coverage_time_invalid"
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    instant = instant.astimezone(timezone.utc)
    instant = instant.replace(microsecond=(instant.microsecond // 1000) * 1000)
    return instant, None


def normalize_simulation_time(value) -> tuple[str | None, str | None]:
    """Return canonical ``YYYY-MM-DDTHH:MM:SS.mmmZ`` time text."""
    instant, error = parse_simulation_time_instant(value)
    if error:
        return None, error
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z", None
