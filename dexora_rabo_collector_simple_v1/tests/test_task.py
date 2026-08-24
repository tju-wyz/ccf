from __future__ import annotations

import threading
import unittest

from rabo_collector.task import RaboNutHandoffTask


class _FakeArm:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def move_to(self, *args, **kwargs):
        result = self.results[self.calls]
        self.calls += 1
        return result

    def move_joints(self, joints):
        result = self.results[self.calls]
        self.calls += 1
        return result


def _make_task() -> RaboNutHandoffTask:
    # 只测重试/中止逻辑，绕过 __init__ 里的 SDK 设备初始化。
    task = object.__new__(RaboNutHandoffTask)
    task._abort_event = threading.Event()
    return task


class CheckedMotionTest(unittest.TestCase):
    def test_move_to_retries_explicit_false(self):
        arm = _FakeArm([False, True])
        _make_task()._move_to_checked(
            arm, "test", 1.0, 2.0, 3.0, roll=0.0, pitch=0.0, yaw=0.0
        )
        self.assertEqual(arm.calls, 2)

    def test_none_is_accepted_for_sdk_compatibility(self):
        arm = _FakeArm([None])
        _make_task()._move_joints_checked(arm, "test", [0.0] * 7)
        self.assertEqual(arm.calls, 1)

    def test_two_false_results_stop_bad_demonstration(self):
        arm = _FakeArm([False, False])
        with self.assertRaisesRegex(RuntimeError, "连续两次执行失败"):
            _make_task()._move_joints_checked(arm, "test", [0.0] * 7)

    def test_abort_event_raises_before_move(self):
        arm = _FakeArm([True])
        task = _make_task()
        task._abort_event.set()
        with self.assertRaisesRegex(RuntimeError, "任务已中止"):
            task._move_joints_checked(arm, "test", [0.0] * 7)
        self.assertEqual(arm.calls, 0)


if __name__ == "__main__":
    unittest.main()
