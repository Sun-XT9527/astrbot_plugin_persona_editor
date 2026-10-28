import asyncio
import json
import unittest

from astrbot_plugin_persona_editor.evolution import (
    AUDIT, REFLECT, EvolutionEngine, likely_feedback, parse_json, render_prompt, split_prompt,
)
from astrbot_plugin_persona_editor.review_store import ReviewError, ReviewStore
from . import test_persona_editor as adapter_tests

workspace_temp = adapter_tests.workspace_temp

FEEDBACK = '以后请少一点说教，也不要总在每次回复结尾追问。'


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.reflection = {'should_update': True, 'evidence': FEEDBACK, 'reason': '用户明确要求长期减少说教和追问',
                           'rules': ['少说教，减少不必要的追问，让用户有自然结束话题的空间。'],
                           'focus_case': {'prompt': '用户说今天终于把手上的事情忙完了。请接话。', 'rubric': '自然回应，不说教，不强行追加问题。'}}
        self.safety = True
        self.worse = False
        self.tie = False
        self.calls = []

        async def call(system, prompt):
            self.calls.append((system, prompt))
            if system == REFLECT:
                return json.dumps(self.reflection, ensure_ascii=False)
            if system == AUDIT:
                data = json.loads(prompt)
                scores = []
                for i, case in enumerate(data['cases']):
                    new_x = case['response_x'] == 'new'
                    old, new = (4, 3) if self.worse and i == 0 else ((4, 4) if self.tie else (3, 4))
                    scores.append({'case_id': case['id'], 'x': new if new_x else old, 'y': old if new_x else new})
                return json.dumps({'safe': self.safety, 'reason': '测试审查理由', 'scores': scores})
            return 'new' if self.reflection['rules'][0] in system else 'old'

        self.engine = EvolutionEngine(call)

    async def test_success_preserves_core_and_saves_three_real_pairs(self):
        result = await self.engine.run('核心身份不可改\n完整原文', FEEDBACK)
        self.assertEqual(split_prompt(result['after'])[0], '核心身份不可改\n完整原文')
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(len(result['evaluation']['cases']), 3)
        self.assertEqual(result['evidence'], FEEDBACK)

    async def test_no_stable_feedback_stops_after_one_call(self):
        self.reflection['should_update'] = False
        with self.assertRaisesRegex(ReviewError, '没有形成'):
            await self.engine.run('core', FEEDBACK)
        self.assertEqual(len(self.calls), 1)

    async def test_invented_evidence_never_generates_trials(self):
        self.reflection['evidence'] = '用户没有说过的话'
        with self.assertRaisesRegex(ReviewError, '逐字'):
            await self.engine.run('core', FEEDBACK)
        self.assertEqual(len(self.calls), 1)

    async def test_rejects_personal_date_before_trials(self):
        self.reflection['rules'] = ['请提醒用户在2026-10-27早上去办事。']
        with self.assertRaises(ReviewError):
            await self.engine.run('core', FEEDBACK)
        self.assertEqual(len(self.calls), 1)

    async def test_any_regression_blocks_despite_total_gain(self):
        self.worse = True
        with self.assertRaisesRegex(ReviewError, '退步'):
            await self.engine.run('core', FEEDBACK)

    async def test_equal_scores_do_not_count_as_progress(self):
        self.tie = True
        with self.assertRaisesRegex(ReviewError, '明确改善'):
            await self.engine.run('core', FEEDBACK)

    async def test_auditor_can_block_even_with_improvement(self):
        self.safety = False
        with self.assertRaisesRegex(ReviewError, '审查未通过'):
            await self.engine.run('core', FEEDBACK)

    async def test_timeout_is_bounded(self):
        async def slow(*args):
            await asyncio.sleep(1)
        with self.assertRaises(TimeoutError):
            await EvolutionEngine(slow, timeout=.01).run('core', FEEDBACK)

    async def test_existing_rules_are_merged_in_single_bounded_block(self):
        before = render_prompt('core', ['保持原有的简短互动习惯。'])
        self.reflection['rules'].append('保持原有的简短互动习惯。')
        result = await self.engine.run(before, FEEDBACK)
        self.assertEqual(result['after'].count('persona_editor:learned:start'), 1)
        self.assertEqual(split_prompt(result['after'])[0], 'core')


class ValidationTests(unittest.TestCase):
    def test_feedback_gate_ignores_chat_schedules_and_commands(self):
        for text in ('我刚健身回来', '周二早上九点提醒我', '/pe approve abc', '你好呀', '下次记得周二办事'):
            self.assertFalse(likely_feedback(text), text)
        for text in (FEEDBACK, '我不喜欢你每次都反驳我，以后先核对再说。'):
            self.assertTrue(likely_feedback(text), text)

    def test_core_marker_tampering_does_not_silently_reset(self):
        with self.assertRaises(ReviewError):
            split_prompt('core\n<!-- persona_editor:learned:start -->')
        with self.assertRaises(ReviewError):
            split_prompt(render_prompt('core', ['合适的互动习惯。']) + '\nmanual after block')

    def test_rule_count_duplicates_and_format_bounded(self):
        for rules in ([], ['短'], ['同一条重复习惯。'] * 2, ['包含\n换行的规则。'], ['x' * 161], ['一般的习惯规则。'] * 9):
            with self.assertRaises(ReviewError):
                render_prompt('core', rules)

    def test_json_is_not_eval_and_requires_whole_object(self):
        self.assertEqual(parse_json('```json\n{"safe": true}\n```'), {'safe': True})
        for value in ('prefix {"safe": true}', '[]', '__import__("os").system("bad")'):
            with self.assertRaises(ReviewError):
                parse_json(value)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from pathlib import Path
        self.temp = workspace_temp()
        self.addCleanup(self.temp.cleanup)
        self.store = ReviewStore(Path(self.temp.name) / 'reviews.json')

    async def reserve(self, feedback=FEEDBACK, cooldown=60, limit=3):
        return await self.store.reserve('fish', 'core', feedback, actor='admin', cooldown_minutes=cooldown, daily_limit=limit)

    async def test_restart_preserves_duplicate_and_cooldown(self):
        await self.reserve()
        with self.assertRaisesRegex(ReviewError, '已分析'):
            await ReviewStore(self.store.path).reserve('fish', 'core', FEEDBACK, actor='admin', cooldown_minutes=60, daily_limit=3)
        with self.assertRaisesRegex(ReviewError, '冷却'):
            await self.reserve(FEEDBACK + '以后记得。')

    async def test_daily_cap_and_pending_backpressure(self):
        await self.reserve()
        with self.assertRaisesRegex(ReviewError, '上限'):
            await self.reserve(FEEDBACK + '第二次', cooldown=0, limit=1)
        await self.store.propose('fish', 'core', 'after', 'reason', 'admin', 'private')
        with self.assertRaisesRegex(ReviewError, '待处理'):
            await self.reserve(FEEDBACK + '第三次', cooldown=0)

    async def test_interrupted_work_is_recovered_without_persona_writes(self):
        attempt = await self.reserve()
        await self.store.recover_interrupted()
        record = (await self.store.snapshot())['attempts'][0]
        self.assertEqual(record['id'], attempt)
        self.assertEqual(record['status'], 'interrupted')


class NewAdapterTests(adapter_tests.AdapterTests):
    async def test_unevaluated_proposal_cannot_be_approved(self):
        p = await self.plugin.store.propose('fish', 'original', 'rewritten', 'reason', 'admin', 'private')
        self.event.is_admin = lambda: True
        self.assertIn('未经过', await self.plugin._manage(self.event, p['id'], 'approve'))
        self.assertEqual(self.calls, [])

    async def test_core_rewrite_is_blocked_even_with_evaluation(self):
        p = await self.plugin.store.propose('fish', 'original', 'new identity', 'reason', 'admin', 'private', evaluation={'kind': 'model_comparison'})
        self.event.is_admin = lambda: True
        self.assertIn('核心人设', await self.plugin._manage(self.event, p['id'], 'approve'))
        self.assertEqual(self.calls, [])

    async def test_auto_detection_is_private_and_owner_only(self):
        self.event.get_group_id = lambda: ''
        self.event.get_message_str = lambda: FEEDBACK
        self.event.get_extra = lambda *args: False
        await self.plugin.observe_feedback(self.event)
        self.assertEqual((await self.plugin.store.snapshot())['attempts'], [])
        self.event.is_admin = lambda: True
        self.event.get_group_id = lambda: 'group'
        await self.plugin.observe_feedback(self.event)
        self.assertEqual((await self.plugin.store.snapshot())['attempts'], [])

    async def test_auto_hook_and_tool_cannot_process_same_feedback_twice(self):
        self.event.get_group_id = lambda: ''
        self.event.get_message_str = lambda: FEEDBACK
        self.event.is_admin = lambda: True
        extras = {}
        self.event.get_extra = lambda key, default=None: extras.get(key, default)
        self.event.set_extra = lambda key, value: extras.update({key: value})
        reply = await self.plugin.evolve_persona(self.event)
        await self.plugin.observe_feedback(self.event)
        self.assertIn('23:00', reply)
        snapshot = await self.plugin.store.snapshot()
        self.assertEqual(len(snapshot['queued']), 1)
        self.assertEqual(snapshot['attempts'], [])
