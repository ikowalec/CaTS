from __future__ import annotations

import os
import hashlib
import json
from functools import lru_cache

from celery import Celery, Task
from celery.utils import uuid

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reaction_hub.settings")

app = Celery("reaction_hub")
app.config_from_object("django.conf:settings", namespace="CELERY")


@lru_cache(maxsize=1)
def _singleflight_redis():
    import redis
    from django.conf import settings

    return redis.Redis.from_url(settings.CELERY_BROKER_URL)


class SingleFlightTask(Task):
    """Keep one queued or running copy of an identical periodic task."""

    abstract = True
    lock_seconds = 6 * 60 * 60

    def _lock_key(self, args, kwargs) -> str:
        payload = json.dumps([self.name, args or (), kwargs or {}], sort_keys=True, default=str)
        return "reaction_hub:celery:singleflight:" + hashlib.sha256(payload.encode()).hexdigest()

    def apply_async(self, args=None, kwargs=None, task_id=None, **options):
        task_id = task_id or uuid()
        key = self._lock_key(args, kwargs)
        broker = _singleflight_redis()
        if not broker.set(key, task_id, nx=True, ex=self.lock_seconds):
            existing_id = broker.get(key)
            return self.AsyncResult(existing_id.decode() if existing_id else task_id)
        try:
            return super().apply_async(args=args, kwargs=kwargs, task_id=task_id, **options)
        except Exception:
            broker.eval(
                "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) end return 0",
                1,
                key,
                task_id,
            )
            raise

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        _singleflight_redis().eval(
            "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) end return 0",
            1,
            self._lock_key(args, kwargs),
            task_id,
        )
        return super().after_return(status, retval, task_id, args, kwargs, einfo)


app.autodiscover_tasks()
