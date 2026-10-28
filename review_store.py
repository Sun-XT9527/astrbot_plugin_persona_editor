"""Reviewed persona changes. No chat history, memory retrieval, or scheduled work."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable


class ReviewError(ValueError):
    pass


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    def __init__(self, path: Path, max_pending: int = 8):
        self.path = path
        self.max_pending = max_pending
        self.lock = asyncio.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {'version': 1, 'proposals': {}}
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('version') != 1 or not isinstance(data.get('proposals'), dict):
            raise ReviewError('审核文件格式异常；已停止写入，请保留文件并检查。')
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)

    async def propose(self, persona_id: str, before: str, after: str,
                      reason: str, actor: str, origin: str, *,
                      evidence: str = '', evaluation: dict | None = None) -> dict:
        if not isinstance(after, str) or not after.strip() or len(after) > 20000:
            raise ReviewError('请提交非空、20000 字以内的完整新人设。')
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1200:
            raise ReviewError('请用 1～1200 字说明修改理由。')
        if before == after:
            raise ReviewError('新人设与当前人设相同，无需修改。')
        async with self.lock:
            data = self._load()
            pending = [p for p in data['proposals'].values()
                       if p['persona_id'] == persona_id
                       and p['status'] in ('pending', 'applying', 'rollback_applying')]
            for p in pending:
                if p['base_hash'] == digest(before) and p['after'] == after:
                    return copy.deepcopy(p)
            if len(pending) >= self.max_pending:
                raise ReviewError('这个人格的待处理建议已达上限，请先审核已有建议。')
            proposal = {
                'id': uuid.uuid4().hex[:12], 'persona_id': persona_id,
                'before': before, 'after': after, 'base_hash': digest(before),
                'reason': reason.strip(), 'actor': actor, 'origin': origin,
                'created_at': now(), 'status': 'pending',
                'evidence': evidence, 'evaluation': evaluation,
            }
            data['proposals'][proposal['id']] = proposal
            self._save(data)
            return copy.deepcopy(proposal)

    async def list_pending(self) -> list[dict]:
        async with self.lock:
            return [copy.deepcopy(p) for p in self._load()['proposals'].values()
                    if p['status'] in ('pending', 'applying', 'rollback_applying')]

    async def snapshot(self) -> dict:
        async with self.lock:
            data = self._load()
            return {'proposals': sorted(copy.deepcopy(list(data['proposals'].values())),
                                        key=lambda p: p['created_at'], reverse=True)[:100],
                    'attempts': copy.deepcopy(data.get('attempts', [])[-30:][::-1]),
                    'queued': copy.deepcopy([f for f in data.get('feedback_queue', []) if f['status'] == 'queued'])}

    async def queue_feedback(self, persona_id: str, feedback: str, actor: str, origin: str) -> dict:
        if not isinstance(feedback, str) or not 4 <= len(feedback.strip()) <= 1500:
            raise ReviewError('请提供 4～1500 字的原始风格反馈。')
        async with self.lock:
            data = self._load()
            queue = data.setdefault('feedback_queue', [])
            key = digest(persona_id + '\0' + feedback.strip())
            for entry in queue:
                if entry['key'] == key:
                    return copy.deepcopy(entry)
            if sum(f['persona_id'] == persona_id and f['status'] in ('queued', 'processing') for f in queue) >= 30:
                raise ReviewError('这个人格已有较多反馈等待每日分析，请先处理。')
            entry = {'id': uuid.uuid4().hex[:12], 'key': key, 'persona_id': persona_id,
                     'feedback': feedback.strip(), 'actor': actor, 'origin': origin,
                     'at': now(), 'status': 'queued'}
            queue.append(entry)
            # Preserve all pending entries; bound only completed collection records.
            data['feedback_queue'] = [f for f in queue if f['status'] != 'consumed'] + [f for f in queue if f['status'] == 'consumed'][-200:]
            self._save(data)
            return copy.deepcopy(entry)

    async def queued_batch(self, persona_id: str) -> list[dict]:
        async with self.lock:
            selected = []
            size = 0
            for entry in self._load().get('feedback_queue', []):
                if entry['persona_id'] != persona_id or entry['status'] != 'queued':
                    continue
                if size + len(entry['feedback']) + (2 if selected else 0) > 1500:
                    break
                selected.append(copy.deepcopy(entry))
                size += len(entry['feedback']) + (2 if len(selected) > 1 else 0)
            return selected

    async def reserve(self, persona_id: str, before: str, feedback: str, *,
                      actor: str, cooldown_minutes: int, daily_limit: int,
                      feedback_ids: list[str] | None = None, at: datetime | None = None) -> str:
        async with self.lock:
            data = self._load()
            attempts = data.setdefault('attempts', [])
            key = digest(persona_id + '\0' + before + '\0' + feedback)
            relevant = [a for a in attempts if a['persona_id'] == persona_id]
            if any(a['key'] == key and a['status'] != 'interrupted' for a in relevant):
                raise ReviewError('这段反馈已分析过，请查看记录，不会重复调用模型。')
            current = (at or datetime.now(timezone.utc)).astimezone(timezone.utc)
            if relevant and (current - datetime.fromisoformat(relevant[-1]['at'])).total_seconds() < cooldown_minutes * 60:
                raise ReviewError('人格分析仍在冷却期，请稍后再试。')
            day = (current + timedelta(hours=8)).date()
            if sum((datetime.fromisoformat(a['at']) + timedelta(hours=8)).date() == day for a in relevant) >= daily_limit:
                raise ReviewError('今天的人格分析次数已达上限，明天再继续。')
            if any(p['persona_id'] == persona_id and p['status'] in ('pending', 'applying', 'rollback_applying')
                   for p in data['proposals'].values()):
                raise ReviewError('这个人格已有待处理建议，请先审核。')
            attempt = {'id': uuid.uuid4().hex[:12], 'key': key, 'persona_id': persona_id,
                       'actor': actor, 'at': current.isoformat(), 'status': 'processing',
                       'feedback': feedback, 'message': '正在分析和比较修改效果'}
            if feedback_ids:
                chosen = [f for f in data.get('feedback_queue', []) if f['id'] in feedback_ids
                          and f['persona_id'] == persona_id and f['status'] == 'queued']
                if len(chosen) != len(set(feedback_ids)):
                    raise ReviewError('待分析反馈已变化，请下次重新检查。')
                for entry in chosen:
                    entry.update(status='processing', attempt_id=attempt['id'])
            attempts.append(attempt)
            data['attempts'] = attempts[-200:]
            self._save(data)
            return attempt['id']

    async def finish_attempt(self, attempt_id: str, status: str, message: str) -> None:
        async with self.lock:
            data = self._load()
            for attempt in data.get('attempts', []):
                if attempt['id'] == attempt_id:
                    attempt.update(status=status, message=message[:600], finished_at=now())
                    for entry in data.get('feedback_queue', []):
                        if entry.get('attempt_id') == attempt_id:
                            entry['status'] = 'queued' if status == 'interrupted' else 'consumed'
                    self._save(data)
                    return

    async def recover_interrupted(self) -> None:
        async with self.lock:
            data = self._load()
            changed = False
            for attempt in data.get('attempts', []):
                if attempt['status'] == 'processing':
                    attempt.update(status='interrupted', message='插件重载中断了分析，人设未自动写入', finished_at=now())
                    changed = True
                    for entry in data.get('feedback_queue', []):
                        if entry.get('attempt_id') == attempt['id']:
                            entry['status'] = 'queued'
            if changed:
                self._save(data)

    async def get(self, proposal_id: str) -> dict:
        async with self.lock:
            proposal = self._load()['proposals'].get(proposal_id)
            if proposal is None:
                raise ReviewError('找不到这条人格修改建议。')
            return copy.deepcopy(proposal)

    async def reject(self, proposal_id: str) -> dict:
        async with self.lock:
            data = self._load()
            proposal = data['proposals'].get(proposal_id)
            if proposal is None or proposal['status'] != 'pending':
                raise ReviewError('只能拒绝尚未执行的待审建议。')
            proposal.update(status='rejected', reviewed_at=now())
            self._save(data)
            return copy.deepcopy(proposal)

    async def apply(self, proposal_id: str, *, rollback: bool,
                    read: Callable[[str], Awaitable[str]],
                    write: Callable[[str, str], Awaitable[None]]) -> dict:
        async with self.lock:
            data = self._load()
            p = data['proposals'].get(proposal_id)
            if p is None:
                raise ReviewError('找不到这条人格修改建议。')
            ready, working, done = (('approved', 'rollback_applying', 'rolled_back')
                                    if rollback else ('pending', 'applying', 'approved'))
            if p['status'] not in (ready, working, done):
                raise ReviewError('这条建议的当前状态不允许执行此操作。')
            current = await read(p['persona_id'])
            expected, target = ((p['after'], p['before']) if rollback
                                else (p['before'], p['after']))
            if p['status'] == done:
                if current != target:
                    raise ReviewError('该操作已完成，但人格后来又有变化；不会重复覆盖。')
                return copy.deepcopy(p)
            # A previous write may have succeeded before persistence or process exit.
            if p['status'] == working and current == target:
                p.update(status=done, reviewed_at=now())
                self._save(data)
                return copy.deepcopy(p)
            if current != expected:
                raise ReviewError('人格已被其他操作修改，建议已过时；请基于最新人设重新提出。')
            p.update(status=working, reviewed_at=now())
            self._save(data)  # Persist the original prompt before changing AstrBot.
            await write(p['persona_id'], target)
            if await read(p['persona_id']) != target:
                raise ReviewError('未确认人格字段已更新，请检查后重试，不要假定已生效。')
            p.update(status=done, reviewed_at=now())
            self._save(data)
            return copy.deepcopy(p)
