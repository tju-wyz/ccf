from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class _FakeVideoWriter:
    def __init__(self, path, fourcc, fps, size):
        self.path = Path(path)
        self.frames = 0

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames += 1

    def release(self):
        self.path.write_bytes(f"fake-mp4:{self.frames}".encode())


fake_cv2 = types.ModuleType("cv2")
fake_cv2.VideoWriter = _FakeVideoWriter
fake_cv2.VideoWriter_fourcc = lambda *args: 0
fake_cv2.INTER_AREA = 0
fake_cv2.resize = lambda frame, size, interpolation=0: frame
sys.modules.setdefault("cv2", fake_cv2)

from rabo_collector.config import load_config
from rabo_collector.episode import EpisodeRecorder, PhaseTracker
from rabo_collector.lerobot_writer import LeRobotV21Writer
from rabo_collector.sensors import MockSensorBackend
from rabo_collector.task import MockNutHandoffTask


class SimplePipelineTest(unittest.TestCase):
    def test_mock_episode_is_trainable(self):
        project = Path(__file__).parents[1]
        config = load_config(project / "config.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            config.raw["dataset"]["root"] = temporary
            config.raw["dataset"]["fps"] = 20
            config.raw["dataset"]["image_width"] = 16
            config.raw["dataset"]["image_height"] = 12

            phase = PhaseTracker()
            sensors = MockSensorBackend(config)
            task = MockNutHandoffTask(phase, phase_duration_s=0.006)
            writer = LeRobotV21Writer(config)
            sensors.start()
            sensors.wait_ready(1.0)
            recorder = EpisodeRecorder(config, sensors, phase)
            recorder.start()
            task.run()
            episode = recorder.stop()

            self.assertGreaterEqual(episode.length, 2)
            self.assertEqual(episode.writer_queue_drops, 0)
            self.assertEqual(set(episode.video_files), {"top"})
            np.testing.assert_allclose(episode.actions[:-1], episode.states[1:])

            writer.add_episode(
                episode,
                task_text=str(config.dataset["task"]),
                success=True,
            )
            writer.rebuild_metadata()
            info_path = Path(temporary) / "meta" / "info.json"
            with info_path.open(encoding="utf-8") as handle:
                info = json.load(handle)
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_videos"], 1)
            table = pq.read_table(
                Path(temporary) / "data/chunk-000/episode_000000.parquet"
            )
            self.assertEqual(table.num_rows, info["total_frames"])


if __name__ == "__main__":
    unittest.main()
