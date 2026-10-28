"""Beijing-time daily scheduling, independent of the Windows time zone."""
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

BEIJING = timezone(timedelta(hours=8))


def parse_time(value: str = '23:00') -> tuple[int, int]:
    try:
        hour, minute = value.split(':')
        if len(hour) != 2 or len(minute) != 2:
            raise ValueError
        hour, minute = int(hour), int(minute)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
        return hour, minute
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('每日分析时间须为 HH:MM，例如 23:00。') from exc


def in_daily_window(at: datetime, time: str) -> bool:
    if at.tzinfo is None:
        raise ValueError('调度时间必须包含时区。')
    local = at.astimezone(BEIJING)
    hour, minute = parse_time(time)
    due = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return 0 <= (local - due).total_seconds() < 300


def create_scheduler(callback, time: str = '23:00') -> AsyncIOScheduler:
    hour, minute = parse_time(time)
    scheduler = AsyncIOScheduler(timezone=BEIJING)
    scheduler.add_job(callback, CronTrigger(hour=hour, minute=minute, timezone=BEIJING),
                      id='persona_editor_daily', replace_existing=True,
                      coalesce=True, max_instances=1, misfire_grace_time=300)
    return scheduler
