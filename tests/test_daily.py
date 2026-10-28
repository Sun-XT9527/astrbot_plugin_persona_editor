import unittest
from datetime import datetime, timedelta, timezone

from apscheduler.triggers.cron import CronTrigger

from astrbot_plugin_persona_editor.daily import BEIJING, create_scheduler, in_daily_window, parse_time
from astrbot_plugin_persona_editor.review_store import ReviewError, ReviewStore
from . import test_persona_editor as adapter_tests

FEEDBACK = '以后说话简短一些，不要每次结尾都追问。'


class ClockTests(unittest.TestCase):
    def test_23_beijing_is_15_utc(self):
        trigger = CronTrigger(hour=23, minute=0, timezone=BEIJING)
        at = datetime(2026, 10, 28, 14, 59, tzinfo=timezone.utc)
        first = trigger.get_next_fire_time(None, at)
        self.assertEqual(first.astimezone(timezone.utc), at.replace(hour=15, minute=0))
        second = trigger.get_next_fire_time(first, first + timedelta(seconds=1))
        self.assertEqual(second - first, timedelta(days=1))

    def test_window_does_not_run_early_or_make_up_next_morning(self):
        at = datetime(2026, 10, 28, 23, 0, tzinfo=BEIJING)
        self.assertTrue(in_daily_window(at, '23:00'))
        self.assertTrue(in_daily_window(at + timedelta(minutes=4), '23:00'))
        for other in (at - timedelta(seconds=1), at + timedelta(minutes=5), at + timedelta(hours=10)):
            self.assertFalse(in_daily_window(other, '23:00'))

    def test_invalid_clock_and_naive_time_are_rejected(self):
        for value in ('24:00', '23:60', '11pm', '3:00', None):
            with self.assertRaises(ValueError):
                parse_time(value)
        with self.assertRaises(ValueError):
            in_daily_window(datetime(2026, 10, 28, 23), '23:00')


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from pathlib import Path
        self.temp = adapter_tests.workspace_temp()
        self.addCleanup(self.temp.cleanup)
        self.store = ReviewStore(Path(self.temp.name) / 'review.json')

    async def test_feedback_survives_restart_and_deduplicates(self):
        entry = await self.store.queue_feedback('fish', FEEDBACK, 'owner', 'private')
        restarted = ReviewStore(self.store.path)
        self.assertEqual((await restarted.queued_batch('fish'))[0]['id'], entry['id'])
        self.assertEqual((await restarted.queue_feedback('fish', FEEDBACK, 'owner', 'private'))['id'], entry['id'])
        self.assertEqual((await restarted.snapshot())['attempts'], [])

    async def test_batch_bounds_size_and_leaves_remaining_feedback(self):
        for text in ('甲' * 800, '乙' * 800):
            await self.store.queue_feedback('fish', text, 'owner', 'private')
        batch = await self.store.queued_batch('fish')
        self.assertEqual(len(batch), 1)
        at = datetime(2026, 10, 28, 23, tzinfo=BEIJING)
        attempt = await self.store.reserve('fish', 'core', batch[0]['feedback'], actor='daily',
            cooldown_minutes=0, daily_limit=1, feedback_ids=[batch[0]['id']], at=at)
        await self.store.finish_attempt(attempt, 'skipped', '没有改进')
        remaining = await self.store.queued_batch('fish')
        self.assertEqual(len(remaining), 1)
        self.assertTrue(remaining[0]['feedback'].startswith('乙'))

    async def test_interruption_returns_batch_but_never_retries_same_day(self):
        entry = await self.store.queue_feedback('fish', FEEDBACK, 'owner', 'private')
        at = datetime(2026, 10, 28, 23, tzinfo=BEIJING)
        args = dict(actor='daily', cooldown_minutes=0, daily_limit=1, feedback_ids=[entry['id']])
        await self.store.reserve('fish', 'core', FEEDBACK, at=at, **args)
        await self.store.recover_interrupted()
        self.assertEqual(len(await self.store.queued_batch('fish')), 1)
        with self.assertRaisesRegex(ReviewError, '上限'):
            await self.store.reserve('fish', 'core', FEEDBACK, at=at, **args)
        await self.store.reserve('fish', 'core', FEEDBACK, at=at + timedelta(days=1), **args)


class DailyAdapterTests(adapter_tests.AdapterTests):
    async def test_scheduler_has_one_daily_job_and_stops_cleanly(self):
        self.plugin.context.registered_web_apis = []
        await self.plugin.initialize()
        scheduler = self.plugin.scheduler
        self.assertTrue(scheduler.running)
        jobs = scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].misfire_grace_time, 300)
        self.assertTrue(jobs[0].coalesce)
        self.assertEqual(jobs[0].next_run_time.astimezone(BEIJING).hour, 23)
        await self.plugin.terminate()
        import asyncio
        await asyncio.sleep(0)
        self.assertFalse(scheduler.running)

    async def test_early_empty_and_repeated_ticks_never_generate_extra_analysis(self):
        self.plugin.cfg['daily_time'] = '23:00'
        calls = []

        async def process(persona_id, feedback, actor, origin, attempt, before):
            calls.append(feedback)
            await self.plugin.store.finish_attempt(attempt, 'skipped', '测试：没有改善')
            self.plugin.busy.discard(persona_id)

        self.plugin._process = process
        at = datetime(2026, 10, 28, 23, tzinfo=BEIJING)
        await self.plugin.run_daily(at)
        self.assertEqual(calls, [])
        await self.plugin._collect('fish', FEEDBACK, 'owner', 'private')
        await self.plugin.run_daily(at - timedelta(minutes=1))
        self.assertEqual(calls, [])
        await self.plugin.run_daily(at)
        await self.plugin._collect('fish', FEEDBACK + '还有别总是说教。', 'owner', 'private')
        await self.plugin.run_daily(at + timedelta(minutes=1))
        self.assertEqual(len(calls), 1)
        await self.plugin.run_daily(at + timedelta(days=1))
        self.assertEqual(len(calls), 2)

    async def test_pending_review_keeps_feedback_for_a_later_day(self):
        await self.proposal()
        await self.plugin._collect('fish', FEEDBACK, 'owner', 'private')
        await self.plugin.run_daily(datetime(2026, 10, 28, 23, tzinfo=BEIJING))
        self.assertEqual((await self.plugin.store.snapshot())['attempts'], [])
        self.assertEqual(len(await self.plugin.store.queued_batch('fish')), 1)
