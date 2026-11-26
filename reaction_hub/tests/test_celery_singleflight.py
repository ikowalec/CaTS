from celery import Task

from reaction_hub import celery as celery_module


class FakeRedis:
    def __init__(self):
        self.values = {}

    def set(self, key, value, *, nx, ex):
        if nx and key in self.values:
            return False
        self.values[key] = str(value).encode()
        return True

    def get(self, key):
        return self.values.get(key)

    def eval(self, _script, _keys, key, value):
        if self.values.get(key) == str(value).encode():
            del self.values[key]
            return 1
        return 0


def test_singleflight_returns_existing_task_until_completion(monkeypatch):
    broker = FakeRedis()
    submitted = []
    monkeypatch.setattr(celery_module, "_singleflight_redis", lambda: broker)
    monkeypatch.setattr(
        Task,
        "apply_async",
        lambda _self, **options: submitted.append(options["task_id"]) or options["task_id"],
    )
    task = celery_module.SingleFlightTask()
    task.name = "test.periodic"

    first = task.apply_async(kwargs={})
    duplicate = task.apply_async(kwargs={})

    assert submitted == [first]
    assert duplicate.id == first
    task.after_return("SUCCESS", None, first, (), {}, None)
    task.apply_async(kwargs={})
    assert len(submitted) == 2
