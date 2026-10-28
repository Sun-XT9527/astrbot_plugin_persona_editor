import ast
import importlib
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from astrbot_plugin_persona_editor.review_store import ReviewError, ReviewStore
from astrbot_plugin_persona_editor.evolution import render_prompt


def workspace_temp():
    parent = Path(__file__).resolve().parent / '_tmp'
    parent.mkdir(exist_ok=True)
    temp = tempfile.TemporaryDirectory(dir=parent)
    assert Path(temp.name).resolve().is_relative_to(parent.resolve())
    return temp


class StoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = workspace_temp()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'reviews.json'
        self.store = ReviewStore(self.path, max_pending=2)
        self.prompts = {'fish': 'original persona'}
        self.writes = []

    async def read(self, persona_id):
        return self.prompts[persona_id]

    async def write(self, persona_id, text):
        self.writes.append((persona_id, text))
        self.prompts[persona_id] = text

    async def proposal(self, after='improved persona'):
        return await self.store.propose('fish', self.prompts['fish'], after,
                                        '用户明确要求减少重复追问', 'owner', 'private')

    async def test_proposal_only_saves_review_and_survives_restart(self):
        p = await self.proposal()
        self.assertEqual(self.prompts['fish'], 'original persona')
        self.assertEqual(self.writes, [])
        self.assertEqual((await ReviewStore(self.path).get(p['id']))['status'], 'pending')

    async def test_approve_and_rollback_preserve_original(self):
        p = await self.proposal()
        await self.store.apply(p['id'], rollback=False, read=self.read, write=self.write)
        self.assertEqual(self.prompts['fish'], 'improved persona')
        await self.store.apply(p['id'], rollback=True, read=self.read, write=self.write)
        self.assertEqual(self.prompts['fish'], 'original persona')
        self.assertEqual((await self.store.get(p['id']))['status'], 'rolled_back')

    async def test_stale_proposal_cannot_overwrite_manual_edit(self):
        p = await self.proposal()
        self.prompts['fish'] = 'new manual change'
        with self.assertRaisesRegex(ReviewError, '过时'):
            await self.store.apply(p['id'], rollback=False, read=self.read, write=self.write)
        self.assertEqual(self.writes, [])

    async def test_stale_rollback_cannot_overwrite_later_edit(self):
        p = await self.proposal()
        await self.store.apply(p['id'], rollback=False, read=self.read, write=self.write)
        self.prompts['fish'] = 'later intentional change'
        with self.assertRaises(ReviewError):
            await self.store.apply(p['id'], rollback=True, read=self.read, write=self.write)
        self.assertEqual(self.prompts['fish'], 'later intentional change')

    async def test_duplicate_proposal_is_deduplicated(self):
        p = await self.proposal()
        again = await self.proposal()
        self.assertEqual(p['id'], again['id'])
        self.assertEqual(len(await self.store.list_pending()), 1)

    async def test_pending_limit_does_not_block_existing_duplicate(self):
        p = await self.proposal()
        await self.proposal('second improvement')
        self.assertEqual((await self.proposal())['id'], p['id'])
        with self.assertRaisesRegex(ReviewError, '上限'):
            await self.proposal('third improvement')

    async def test_invalid_content_does_not_create_a_file(self):
        with self.assertRaises(ReviewError):
            await self.proposal(' ')
        self.assertFalse(self.path.exists())

    async def test_rejected_proposal_cannot_be_applied(self):
        p = await self.proposal()
        await self.store.reject(p['id'])
        with self.assertRaises(ReviewError):
            await self.store.apply(p['id'], rollback=False, read=self.read, write=self.write)
        self.assertEqual(self.writes, [])

    async def test_retry_after_uncertain_write_does_not_write_twice(self):
        p = await self.proposal()

        async def uncertain_write(persona_id, text):
            await self.write(persona_id, text)
            raise OSError('process lost confirmation')

        with self.assertRaises(OSError):
            await self.store.apply(p['id'], rollback=False, read=self.read, write=uncertain_write)
        restarted = ReviewStore(self.path)
        result = await restarted.apply(p['id'], rollback=False, read=self.read, write=self.write)
        self.assertEqual(result['status'], 'approved')
        self.assertEqual(len(self.writes), 1)

    async def test_noop_update_is_not_reported_successful(self):
        p = await self.proposal()

        async def noop(persona_id, text):
            pass

        with self.assertRaisesRegex(ReviewError, '未确认'):
            await self.store.apply(p['id'], rollback=False, read=self.read, write=noop)
        self.assertEqual((await self.store.get(p['id']))['status'], 'applying')

    async def test_corrupt_store_is_never_silently_replaced(self):
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            await self.proposal()
        self.assertEqual(self.path.read_text(encoding='utf-8'), '{broken')


def fake_api(data_dir):
    def decorator(*args, **kwargs):
        return lambda fn: fn

    def command_group(*args, **kwargs):
        def bind(fn):
            fn.command = decorator
            return fn
        return bind

    class Star:
        def __init__(self, context):
            self.context = context

    modules = {name: types.ModuleType(name) for name in (
        'astrbot', 'astrbot.api', 'astrbot.api.all',
        'astrbot.api.event', 'astrbot.api.star',
    )}
    modules['astrbot.api'].logger = logging.getLogger('persona-editor-test')
    modules['astrbot.api.all'].Star = Star
    modules['astrbot.api.all'].Context = object
    modules['astrbot.api.all'].AstrMessageEvent = object
    modules['astrbot.api.all'].register = decorator
    modules['astrbot.api.event'].filter = types.SimpleNamespace(
        llm_tool=decorator, command_group=command_group, after_message_sent=decorator,
    )
    modules['astrbot.api.star'].StarTools = types.SimpleNamespace(get_data_dir=lambda *args: data_dir)
    return modules


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = workspace_temp()
        self.addCleanup(self.temp.cleanup)
        self.manager = types.SimpleNamespace()
        self.persona = types.SimpleNamespace(system_prompt='original', tools=['existing'],
                                             skills=['existing-skill'], begin_dialogs=['hello'])
        self.calls = []

        async def get(persona_id):
            return self.persona

        async def update(**kwargs):
            self.calls.append(kwargs)
            self.persona.system_prompt = kwargs['system_prompt']
            return self.persona

        async def resolve(**kwargs):
            return ('fish', None, None, False)

        async def cid(umo):
            return None

        self.manager.get_persona = get
        self.manager.update_persona = update
        self.manager.resolve_selected_persona = resolve
        context = types.SimpleNamespace(
            persona_manager=self.manager,
            conversation_manager=types.SimpleNamespace(get_curr_conversation_id=cid),
            get_config=lambda **kwargs: {},
            register_web_api=lambda *args: None,
        )
        with patch.dict(sys.modules, fake_api(Path(self.temp.name))):
            sys.modules.pop('astrbot_plugin_persona_editor.main', None)
            module = importlib.import_module('astrbot_plugin_persona_editor.main')
            self.plugin = module.PersonaEditor(context, {'enabled': True, 'persona_ids': ['fish']})
        self.event = types.SimpleNamespace(
            unified_msg_origin='private', get_sender_id=lambda: 'user',
            get_platform_name=lambda: 'qq_official', is_admin=lambda: False,
        )

    async def proposal(self):
        return await self.plugin.store.propose('fish', 'original',
            render_prompt('original', ['回复时减少不必要的重复追问。']), '明确长期反馈', 'owner', 'private',
            evidence='以后请少追问', evaluation={'kind': 'model_comparison'})

    async def test_non_admin_cannot_apply_changes(self):
        p = await self.proposal()
        reply = await self.plugin._manage(self.event, p['id'], 'approve')
        self.assertIn('只有', reply)
        self.assertEqual(self.calls, [])

    async def test_approval_updates_only_system_prompt(self):
        p = await self.proposal()
        self.event.is_admin = lambda: True
        reply = await self.plugin._manage(self.event, p['id'], 'approve')
        self.assertIn('已更新', reply)
        self.assertEqual(self.calls, [{'persona_id': 'fish', 'system_prompt': p['after']}])
        self.assertEqual(self.persona.tools, ['existing'])
        self.assertEqual(self.persona.skills, ['existing-skill'])
        self.assertEqual(self.persona.begin_dialogs, ['hello'])

    async def test_unlisted_persona_cannot_propose(self):
        self.plugin.cfg['persona_ids'] = ['another']
        with self.assertRaisesRegex(ReviewError, '不在允许'):
            await self.plugin._current_persona(self.event)
        self.assertEqual(await self.plugin.store.list_pending(), [])

    async def test_disabled_plugin_cannot_write_pending_change(self):
        p = await self.proposal()
        self.event.is_admin = lambda: True
        self.plugin.cfg['enabled'] = False
        self.assertIn('未获准', await self.plugin._manage(self.event, p['id'], 'approve'))
        self.assertEqual(self.calls, [])


class BoundaryTests(unittest.TestCase):
    def test_one_tool_no_request_injection_schedules_or_outbound_messages(self):
        source = (Path(__file__).parents[1] / 'main.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        names = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertEqual(names.count('llm_tool'), 1)
        self.assertEqual(names.count('after_message_sent'), 1)
        for forbidden in ('on_llm_request', 'on_llm_response', 'event_message_type',
                          'add_job', 'text_chat', 'stop_event', 'send'):
            self.assertNotIn(forbidden, names)


if __name__ == '__main__':
    unittest.main()
