from datetime import UTC, date, datetime, time, timedelta, timezone

# The hospital runs on Indian Standard Time; everything is stored in UTC.
IST = timezone(timedelta(hours=5, minutes=30))


def now_utc() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def today_ist() -> date:
    return datetime.now(IST).date()


def ist_day_bounds(day: date) -> tuple[datetime, datetime]:
    """UTC start (inclusive) and end (exclusive) of a hospital-local calendar day."""
    start = datetime.combine(day, time.min, tzinfo=IST)
    return start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC)
