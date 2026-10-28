"""Feedback, comparison, review and versioned writes to the persona field."""
from __future__ import annotations

import asyncio
import difflib
from pathlib import Path
from datetime import datetime

from astrbot.api import logger
from astrbot.api.all import AstrMessageEvent, Context, Star, register
from astrbot.api.event import filter
from astrbot.api.star import StarTools
from quart import request
from starlette.responses import HTMLResponse

from .evolution import EvolutionEngine, likely_feedback, split_prompt
from .review_store import ReviewError, ReviewStore
from .daily import BEIJING, create_scheduler, in_daily_window

PLUGIN = 'astrbot_plugin_persona_editor'
PREFIX = f'/{PLUGIN}/page/'


def diff(before, after):
    return '\n'.join(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                         fromfile='修改前', tofile='候选', lineterm=''))


@register(PLUGIN, 'Sun-XT9527', '每天23点汇总反馈改进人格，比较、审核与版本回退', '0.3.0')
class PersonaEditor(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.cfg = config
        self.store = ReviewStore(Path(StarTools.get_data_dir(PLUGIN)) / 'persona_reviews.json')
        self.tasks: set[asyncio.Task] = set()
        self.busy: set[str] = set()
        self.locks: dict[str, asyncio.Lock] = {}
        self.stopping = False
        self.scheduler = None
        for route, handler, method in (
            ('view', self.view, 'GET'), ('records', self.records, 'GET'),
            ('analyze', self.analyze, 'POST'), ('action', self.action, 'POST'),
        ):
            context.register_web_api(PREFIX + route, handler, [method], '人格进化（仅人设）')

    async def initialize(self):
        await self.store.recover_interrupted()
        time = self.cfg.get('daily_time', '23:00')
        self.scheduler = create_scheduler(self._daily_tick, time)
        self.scheduler.start()
        logger.info('[PersonaEditor] v0.3.0 已加载；北京时间每天 %s 分析一次，审核后写入', time)

    async def terminate(self):
        self.stopping = True
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.context.registered_web_apis[:] = [
            api for api in self.context.registered_web_apis if not api[0].startswith(PREFIX)]

    def _allowed(self, persona_id: str) -> bool:
        return (self.cfg.get('enabled', True) is True
                and persona_id not in ('', 'default', '[%None]', '_chatui_default_')
                and persona_id in self.cfg.get('persona_ids', ['笨蛋大肥鱼']))

    def _trusted(self, event: AstrMessageEvent) -> bool:
        return event.is_admin() or str(event.get_sender_id()) in self.cfg.get('feedback_users', [])

    async def _current_persona(self, event: AstrMessageEvent) -> str:
        umo = event.unified_msg_origin
        manager = self.context.conversation_manager
        cid = await manager.get_curr_conversation_id(umo)
        conversation = await manager.get_conversation(umo, cid) if cid else None
        selected = await self.context.persona_manager.resolve_selected_persona(
            umo=umo, conversation_persona_id=getattr(conversation, 'persona_id', None),
            platform_name=event.get_platform_name(),
            provider_settings=self.context.get_config(umo=umo).get('provider_settings', {}))
        persona_id = selected[0] or ''
        if not self._allowed(persona_id):
            raise ReviewError('当前人格不在允许修改的自定义人格列表中。')
        return persona_id

    async def _read(self, persona_id: str) -> str:
        if not self._allowed(persona_id):
            raise ReviewError('该人格当前未获准修改。')
        return (await self.context.persona_manager.get_persona(persona_id)).system_prompt

    async def _write(self, persona_id: str, text: str) -> None:
        if not self._allowed(persona_id):
            raise ReviewError('该人格当前未获准修改。')
        result = await self.context.persona_manager.update_persona(persona_id=persona_id, system_prompt=text)
        if result is None:
            raise ReviewError('人格接口没有返回更新结果，请检查后重试。')

    async def _reserve(self, persona_id: str, feedback: str, actor: str, *, feedback_ids=None, at=None) -> tuple[str, str]:
        if self.stopping or persona_id in self.busy:
            raise ReviewError('这个人格正在分析，或插件正在停止。')
        before = await self._read(persona_id)
        if not isinstance(feedback, str) or not 4 <= len(feedback.strip()) <= 1500:
            raise ReviewError('请提供 4～1500 字的原始风格反馈。')
        attempt = await self.store.reserve(
            persona_id, before, feedback, actor=actor,
            cooldown_minutes=0, daily_limit=1, feedback_ids=feedback_ids, at=at)
        self.busy.add(persona_id)
        return attempt, before

    async def _collect(self, persona_id, feedback, actor, origin):
        if self.stopping or not self._allowed(persona_id):
            raise ReviewError('该人格当前未获准收集反馈。')
        return await self.store.queue_feedback(persona_id, feedback, actor, origin)

    async def _daily_tick(self):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            await self.run_daily()
        finally:
            self.tasks.discard(task)

    async def run_daily(self, at=None):
        current = at or datetime.now(BEIJING)
        if self.stopping or not self.cfg.get('enabled', True):
            return
        if not in_daily_window(current, self.cfg.get('daily_time', '23:00')):
            return
        for persona_id in self.cfg.get('persona_ids', ['笨蛋大肥鱼']):
            try:
                entries = await self.store.queued_batch(persona_id)
                if not entries:
                    continue
                feedback = '\n\n'.join(entry['feedback'] for entry in entries)
                attempt, before = await self._reserve(persona_id, feedback, 'daily-schedule',
                    feedback_ids=[entry['id'] for entry in entries], at=current)
                await self._process(persona_id, feedback, 'daily-schedule', entries[-1]['origin'], attempt, before)
            except (ReviewError, ValueError) as exc:
                logger.info('[PersonaEditor] 每日分析保留原样：%s', str(exc))

    async def _process(self, persona_id, feedback, actor, origin, attempt, before):
        try:
            provider = str(self.cfg.get('provider_id', '') or await self.context.get_current_chat_provider_id(origin))
            if not provider:
                raise ReviewError('没有可用的分析模型，请先在插件配置选择模型。')

            async def call(system, prompt):
                response = await self.context.llm_generate(
                    chat_provider_id=provider, system_prompt=system, prompt=prompt, contexts=[], tools=None)
                return response.completion_text

            result = await EvolutionEngine(call).run(before, feedback)
            if await self._read(persona_id) != before:
                raise ReviewError('分析期间人设已变化；请基于最新人设重新分析。')
            result['evaluation']['provider_id'] = provider
            proposal = await self.store.propose(
                persona_id, before, result['after'], result['reason'], actor, origin,
                evidence=result['evidence'], evaluation=result['evaluation'])
            await self.store.finish_attempt(attempt, 'proposed', '已通过比较，等待审核：' + proposal['id'])
            logger.info('[PersonaEditor] 候选已入队 id=%s', proposal['id'])
            return proposal
        except asyncio.CancelledError:
            await self.store.finish_attempt(attempt, 'interrupted', '分析被停用或重载中断，人设未自动写入')
            raise
        except (ReviewError, ValueError, TimeoutError) as exc:
            await self.store.finish_attempt(attempt, 'skipped', str(exc) or '模型调用超时，人设保持原样')
            raise ReviewError(str(exc) or '模型调用超时，人设保持原样') from exc
        except Exception as exc:
            logger.warning('[PersonaEditor] 分析失败: %s', type(exc).__name__)
            await self.store.finish_attempt(attempt, 'error', '分析失败（' + type(exc).__name__ + '），人设未修改')
            raise ReviewError('分析失败，请查看后台记录；人设未修改。') from exc
        finally:
            self.busy.discard(persona_id)

    @filter.after_message_sent()
    async def observe_feedback(self, event: AstrMessageEvent):
        if (not self.cfg.get('auto_propose', True) or not self._trusted(event)
                or event.get_group_id() or getattr(event, '_proactive_plugin', None)
                or event.get_extra('persona_editor_submitted', False)):
            return
        feedback = event.get_message_str().strip()
        if not likely_feedback(feedback):
            return
        try:
            persona_id = await self._current_persona(event)
            await self._collect(persona_id, feedback, str(event.get_sender_id()), str(event.unified_msg_origin))
        except (ReviewError, ValueError):
            pass
        except Exception as exc:
            logger.warning('[PersonaEditor] 反馈检测失败: %s', type(exc).__name__)

    @filter.llm_tool(name='evolve_persona')
    async def evolve_persona(self, event: AstrMessageEvent, new_system_prompt: str = '', reason: str = '') -> str:
        """根据当前用户明确的长期互动反馈，提出人格修订，等待后台审核。

        不为记忆、日程或临时任务调用。插件记录用户原话，到北京时间每日固定时间再集中分析。
        保留核心身份，不能宣称已经永久改变；实际结果以工具返回为准。

        Args:
            new_system_prompt(string): 可留空；兼容旧参数，不会直接采用这段全文。
            reason(string): 简述用户的反馈，仅作调用说明，原始依据由插件读取。
        """
        if not self._trusted(event):
            return '只有管理员或配置中允许的反馈用户可以提出人格改进。'
        event.set_extra('persona_editor_submitted', True)
        feedback = event.get_message_str().strip()
        if not likely_feedback(feedback):
            return '当前消息不包含明确的长期互动风格反馈，未修改人设。'
        try:
            persona_id = await self._current_persona(event)
            entry = await self._collect(persona_id, feedback, str(event.get_sender_id()), str(event.unified_msg_origin))
            if entry['status'] == 'consumed':
                return '这段反馈已经分析过，不会重复提交。人设是否生效以后台审核状态为准。'
            return f"已记录反馈，计划北京时间每天 {self.cfg.get('daily_time', '23:00')} 集中分析一次，审核后才会写入人设。"
        except (ReviewError, ValueError) as exc:
            return str(exc)

    async def _act(self, proposal_id: str, action: str) -> str:
        p = await self.store.get(proposal_id)
        async with self.locks.setdefault(p['persona_id'], asyncio.Lock()):
            if action == 'reject':
                await self.store.reject(proposal_id)
                return '已拒绝建议，人格未修改。'
            if action not in ('approve', 'rollback'):
                raise ReviewError('不支持这项操作。')
            if action == 'approve':
                if not p.get('evaluation') or p['evaluation'].get('kind') != 'model_comparison':
                    raise ReviewError('这条旧建议未经过效果比较，请重新生成。')
                if split_prompt(p['before'])[0] != split_prompt(p['after'])[0]:
                    raise ReviewError('候选改动涉及核心人设，已拒绝写入。')
            p = await self.store.apply(proposal_id, rollback=action == 'rollback', read=self._read, write=self._write)
            return f"已{'回退' if action == 'rollback' else '更新'}“{p['persona_id']}”的人格设定；版本 {p['id']}。"

    @filter.command_group('pe')
    def pe(self):
        """人格进化审核：review / approve / reject / rollback。"""

    @pe.command('review')
    async def review(self, event: AstrMessageEvent, proposal_id: str = ''):
        if not event.is_admin():
            yield event.plain_result('只有 AstrBot 管理员可以查看人格审核。')
            return
        try:
            if not proposal_id:
                pending = await self.store.list_pending()
                text = '\n'.join(f"{p['id']} · {p['persona_id']} · {p['status']}\n{p['reason']}" for p in pending) or '没有待审建议；分析结果可在后台“人格进化”查看。'
            else:
                p = await self.store.get(proposal_id)
                text = f"{p['id']} · {p['status']}\n依据：{p.get('evidence', '')}\n理由：{p['reason']}\n\n{diff(p['before'], p['after'])}\n\n/pe approve {p['id']}\n/pe reject {p['id']}"
            yield event.plain_result(text)
        except (ReviewError, ValueError) as exc:
            yield event.plain_result(str(exc))

    async def _manage(self, event, proposal_id, action):
        if not event.is_admin():
            return '只有 AstrBot 管理员可以批准、拒绝或回退人格修改。'
        try:
            return await self._act(proposal_id, action)
        except (ReviewError, ValueError) as exc:
            return str(exc)
        except Exception:
            return '操作中断，不能确认是否完成；请用同一操作重试核对。'

    @pe.command('approve')
    async def approve(self, event: AstrMessageEvent, proposal_id: str):
        yield event.plain_result(await self._manage(event, proposal_id, 'approve'))

    @pe.command('reject')
    async def reject(self, event: AstrMessageEvent, proposal_id: str):
        yield event.plain_result(await self._manage(event, proposal_id, 'reject'))

    @pe.command('rollback')
    async def rollback(self, event: AstrMessageEvent, proposal_id: str):
        yield event.plain_result(await self._manage(event, proposal_id, 'rollback'))

    async def view(self):
        root = Path(__file__).parent / 'pages' / 'dashboard'
        html = (root / 'index.html').read_text(encoding='utf-8')
        html = html.replace('<link rel="stylesheet" href="style.css">', '<style>' + (root / 'style.css').read_text(encoding='utf-8') + '</style>')
        html = html.replace('<script defer src="app.js"></script>', '<script>' + (root / 'app.js').read_text(encoding='utf-8') + '</script>')
        return HTMLResponse(html, headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})

    async def records(self):
        snapshot = await self.store.snapshot()
        for p in snapshot['proposals']:
            p['diff'] = diff(p['before'], p['after'])
        return {'status': 'ok', 'data': {**snapshot, 'enabled': self.cfg.get('enabled', True),
                'auto_propose': self.cfg.get('auto_propose', True), 'personas': self.cfg.get('persona_ids', []),
                'feedback_users_count': len(self.cfg.get('feedback_users', [])), 'busy': list(self.busy),
                'provider_id': self.cfg.get('provider_id', '') or '当前会话模型',
                'daily_time': self.cfg.get('daily_time', '23:00'), 'daily_limit': 1,
                'next_run_time': (self.scheduler.get_job('persona_editor_daily').next_run_time.isoformat()
                                  if self.scheduler and self.scheduler.running else None)}}

    async def analyze(self):
        try:
            body = await request.get_json(silent=True)
            if not isinstance(body, dict):
                raise ReviewError('请填写人格和原始反馈。')
            persona_id, feedback = body.get('persona_id', ''), body.get('feedback', '')
            if not isinstance(persona_id, str) or not isinstance(feedback, str):
                raise ReviewError('人格和反馈格式不正确。')
            entry = await self._collect(persona_id, feedback, 'dashboard-admin', 'persona-editor-dashboard')
            message = ('这段反馈已分析过，不会重复排队。' if entry['status'] == 'consumed'
                       else f"反馈已保存，等待北京时间 {self.cfg.get('daily_time', '23:00')} 的每日分析。")
            return {'status': 'ok', 'data': {'id': entry['id'], 'message': message}}
        except (ReviewError, ValueError) as exc:
            return {'status': 'error', 'message': str(exc)}

    async def action(self):
        try:
            body = await request.get_json(silent=True)
            if not isinstance(body, dict) or not all(isinstance(body.get(k), str) for k in ('id', 'action')):
                raise ReviewError('请指定有效的建议编号和操作。')
            result = await self._act(body['id'], body['action'])
            return {'status': 'ok', 'data': {'message': result}}
        except (ReviewError, ValueError) as exc:
            return {'status': 'error', 'message': str(exc)}
        except Exception:
            return {'status': 'error', 'message': '操作结果未确认，请刷新记录并用同一操作重试核对。'}
