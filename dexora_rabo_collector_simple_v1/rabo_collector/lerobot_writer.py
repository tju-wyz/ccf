from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CollectorConfig
from .episode import RecordedEpisode


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


class _StreamingStats:
    """常量内存统计；q01/q99 使用固定容量的随机优先级样本近似。"""

    def __init__(self, dimension: int, seed: int, reservoir_size: int = 100_000):
        self.dimension = dimension
        self.reservoir_size = reservoir_size
        self.count = 0
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_sq = np.zeros(dimension, dtype=np.float64)
        self.samples = np.empty((0, dimension), dtype=np.float32)
        self.priorities = np.empty(0, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"统计向量维度错误：{values.shape}")
        values64 = values.astype(np.float64)
        self.count += len(values)
        self.minimum = np.minimum(self.minimum, np.min(values64, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(values64, axis=0))
        self.total += np.sum(values64, axis=0)
        self.total_sq += np.sum(values64 * values64, axis=0)

        priorities = self.rng.random(len(values))
        merged_values = np.concatenate([self.samples, values], axis=0)
        merged_priorities = np.concatenate([self.priorities, priorities])
        if len(merged_values) > self.reservoir_size:
            keep = np.argpartition(merged_priorities, self.reservoir_size - 1)[: self.reservoir_size]
            merged_values = merged_values[keep]
            merged_priorities = merged_priorities[keep]
        self.samples = merged_values
        self.priorities = merged_priorities

    def finish(self) -> dict[str, list[float]]:
        if self.count == 0:
            return {}
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "q01": np.quantile(self.samples, 0.01, axis=0).tolist(),
            "q99": np.quantile(self.samples, 0.99, axis=0).tolist(),
        }


class LeRobotV21Writer:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.root = config.root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)
        indices = self.episode_indices()
        self._next_episode = indices[-1] + 1 if indices else 0
        self._frame_cursor = sum(
            pq.read_metadata(self._parquet_path(index)).num_rows for index in indices
        )

    @property
    def task_texts(self) -> list[str]:
        values = list(self.config.dataset.get("language_variants") or [])
        canonical = str(self.config.dataset["task"])
        if canonical not in values:
            values.insert(0, canonical)
        return values

    def episode_indices(self) -> list[int]:
        found: list[int] = []
        for path in self.root.glob("data/chunk-*/episode_*.parquet"):
            try:
                found.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return sorted(found)

    def next_episode_index(self) -> int:
        return self._next_episode

    def total_frames(self) -> int:
        return self._frame_cursor

    def _chunk(self, episode_index: int) -> int:
        return episode_index // int(self.config.dataset["chunk_size"])

    def _parquet_path(self, episode_index: int) -> Path:
        return self.root / "data" / f"chunk-{self._chunk(episode_index):03d}" / f"episode_{episode_index:06d}.parquet"

    def _video_path(self, episode_index: int, camera: str) -> Path:
        return (
            self.root
            / "videos"
            / f"chunk-{self._chunk(episode_index):03d}"
            / f"observation.images.{camera}"
            / f"episode_{episode_index:06d}.mp4"
        )

    def add_episode(
        self,
        episode: RecordedEpisode,
        *,
        task_text: str,
        success: bool,
        failure_reason: str = "",
        randomization: dict[str, Any] | None = None,
    ) -> int:
        episode_index = self._next_episode
        start_index = self._frame_cursor
        length = episode.length
        state_dim = len(self.config.state_names)
        vector_type = pa.list_(pa.float32(), state_dim)
        full_vector_type = pa.list_(pa.float32(), len(self.config.full_state_names))
        try:
            task_index = self.task_texts.index(task_text)
        except ValueError as exc:
            raise ValueError(f"task_text不在dataset.language_variants中：{task_text}") from exc

        table = pa.table(
            {
                "observation.state": pa.array(episode.states.tolist(), type=vector_type),
                "observation.full_state": pa.array(
                    episode.full_states.tolist(), type=full_vector_type
                ),
                "action": pa.array(episode.actions.tolist(), type=vector_type),
                "timestamp": pa.array(
                    (episode.sim_times - episode.sim_times[0]).astype(np.float32)
                ),
                "frame_index": pa.array(np.arange(length, dtype=np.int64)),
                "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(start_index, start_index + length, dtype=np.int64)),
                "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
                "phase": pa.array(episode.phases, type=pa.string()),
                "phase_index": pa.array(episode.phase_indices, type=pa.int64()),
                "sensor_monotonic_time": pa.array(episode.monotonic_times, type=pa.float64()),
                "sensor_sim_time": pa.array(episode.sim_times, type=pa.float64()),
            }
        )

        parquet_path = self._parquet_path(episode_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_parquet, compression="zstd")

        moved_videos: list[Path] = []
        try:
            for camera, source in episode.video_files.items():
                destination = self._video_path(episode_index, camera)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved_videos.append(destination)
            temporary_parquet.replace(parquet_path)
        except BaseException:
            temporary_parquet.unlink(missing_ok=True)
            for path in moved_videos:
                path.unlink(missing_ok=True)
            raise
        finally:
            episode.discard()

        sim_span = float(episode.sim_times[-1] - episode.sim_times[0]) if length > 1 else 0.0
        sim_fps = float(length - 1) / sim_span if sim_span > 0 else None
        quality_pass = bool(
            sim_fps is not None
            and sim_fps >= self.config.fps * 0.9
            and episode.max_stamp_gap_s <= self.config.max_stamp_gap_s
            and episode.camera_skew_s <= self.config.max_camera_skew_s
            and episode.joint_skew_s <= self.config.max_joint_age_s
            and episode.writer_queue_drops == 0
        )
        sidecar = {
            "episode_index": episode_index,
            "length": length,
            "tasks": [task_text],
            "success": bool(success),
            "failure_reason": failure_reason,
            "sample_timeouts": episode.sample_timeouts,
            "stale_samples": episode.stale_samples,
            "camera_stale_samples": episode.camera_stale_samples,
            "camera_skew_s": episode.camera_skew_s,
            "joint_skew_s": episode.joint_skew_s,
            "writer_queue_drops": episode.writer_queue_drops,
            "max_stamp_gap_s": episode.max_stamp_gap_s,
            "camera_reuse_counts": episode.camera_reuse_counts,
            "sim_fps": sim_fps,
            "quality_pass": quality_pass,
            "randomization": randomization or {},
        }
        sidecar_path = self.root / "meta" / "episode_details" / f"episode_{episode_index:06d}.json"
        _json_dump(sidecar_path, sidecar)
        self._next_episode += 1
        self._frame_cursor += length
        return episode_index

    @staticmethod
    def _feature_stats(values: np.ndarray) -> dict[str, list[float]]:
        values = np.asarray(values, dtype=np.float64)
        return {
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    def rebuild_metadata(self) -> None:
        indices = self.episode_indices()
        episodes: list[dict[str, Any]] = []
        episode_stats: list[dict[str, Any]] = []
        total_frames = 0
        state_stats = _StreamingStats(len(self.config.state_names), seed=17)
        full_state_stats = _StreamingStats(len(self.config.full_state_names), seed=23)
        action_stats = _StreamingStats(len(self.config.state_names), seed=29)

        task = str(self.config.dataset["task"])
        for index in indices:
            table = pq.read_table(
                self._parquet_path(index),
                columns=["observation.state", "observation.full_state", "action"],
            )
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            full_states = np.asarray(table["observation.full_state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            length = len(states)
            total_frames += length
            detail_path = self.root / "meta" / "episode_details" / f"episode_{index:06d}.json"
            detail: dict[str, Any] = {}
            if detail_path.exists():
                with detail_path.open("r", encoding="utf-8") as handle:
                    detail = json.load(handle)
            episodes.append(
                {
                    "episode_index": index,
                    "tasks": detail.get("tasks", [task]),
                    "length": length,
                    "success": bool(detail.get("success", True)),
                }
            )
            episode_stats.append(
                {
                    "episode_index": index,
                    "stats": {
                        "observation.state": self._feature_stats(states),
                        "observation.full_state": self._feature_stats(full_states),
                        "action": self._feature_stats(actions),
                    },
                }
            )
            state_stats.update(states)
            full_state_stats.update(full_states)
            action_stats.update(actions)

        stats: dict[str, Any] = {}
        if total_frames:
            stats = {
                "observation.state": state_stats.finish(),
                "observation.full_state": full_state_stats.finish(),
                "action": action_stats.finish(),
            }

        _write_jsonl(
            self.root / "meta" / "tasks.jsonl",
            [{"task_index": index, "task": text} for index, text in enumerate(self.task_texts)],
        )
        _write_jsonl(self.root / "meta" / "episodes.jsonl", episodes)
        _write_jsonl(self.root / "meta" / "episodes_stats.jsonl", episode_stats)
        _json_dump(self.root / "meta" / "stats.json", stats)
        _json_dump(self.root / "meta" / "modality.json", self._modality())
        _json_dump(self.root / "meta" / "info.json", self._info(indices, total_frames))

    def _video_feature(self) -> dict[str, Any]:
        height = int(self.config.dataset["image_height"])
        width = int(self.config.dataset["image_width"])
        return {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": str(self.config.dataset.get("video_fourcc", "mp4v")),
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": self.config.fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    def _info(self, indices: list[int], total_frames: int) -> dict[str, Any]:
        names = list(self.config.state_names)
        full_names = list(self.config.full_state_names)
        features: dict[str, Any] = {
            f"observation.images.{name}": self._video_feature()
            for name in self.config.camera_topics
        }
        features.update(
            {
                "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
                "observation.full_state": {
                    "dtype": "float32", "shape": [len(full_names)], "names": full_names
                },
                "action": {"dtype": "float32", "shape": [len(names)], "names": names},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "phase": {"dtype": "string", "shape": [1], "names": None},
                "phase_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        chunk_count = math.ceil(len(indices) / int(self.config.dataset["chunk_size"])) if indices else 0
        return {
            "codebase_version": "v2.1",
            "robot_type": str(self.config.dataset["robot_type"]),
            "total_episodes": len(indices),
            "total_frames": total_frames,
            "total_tasks": len(self.task_texts),
            "total_videos": len(indices) * len(self.config.camera_topics),
            "total_chunks": chunk_count,
            "chunks_size": int(self.config.dataset["chunk_size"]),
            "fps": self.config.fps,
            "splits": {"train": f"0:{len(indices)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

    def _modality(self) -> dict[str, Any]:
        names = list(self.config.state_names)
        full_names = list(self.config.full_state_names)
        return {
            "state": {"start": 0, "end": len(names), "names": names},
            "full_state": {"start": 0, "end": len(full_names), "names": full_names},
            "action": {"start": 0, "end": len(names), "names": names},
            "images": list(self.config.camera_topics),
        }
