from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rabo_collector.config import load_config
from rabo_collector.sensors import Ros2SensorBackend


def image_message() -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        encoding="bgr8",
        width=2,
        height=1,
        step=6,
        data=bytes([1, 2, 3, 4, 5, 6]),
    )


class LatestCacheTest(unittest.TestCase):
    def test_low_fps_camera_is_reused_instead_of_blocking(self):
        project = Path(__file__).parents[1]
        config = load_config(project / "config.yaml")
        config.raw["dataset"]["cameras"] = {"top": "/mock/top"}
        config.raw["dataset"]["primary_camera"] = "top"

        backend = Ros2SensorBackend(
            config,
            joint_reader=lambda: np.zeros(
                len(config.full_state_names), dtype=np.float32
            ),
        )

        with backend._cond:
            backend._camera_rings["top"].append(1.0, image_message())
            backend._camera_last_arrival["top"] = time.monotonic()

        backend.reset_clock()
        stop = threading.Event()
        first = backend.sample_next(0.01, 1.0, stop)
        second = backend.sample_next(0.01, 1.0, stop)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertFalse(first.camera_reused["top"])
        self.assertTrue(second.camera_reused["top"])
        self.assertAlmostEqual(second.sim_time - first.sim_time, 0.01)


if __name__ == "__main__":
    unittest.main()
