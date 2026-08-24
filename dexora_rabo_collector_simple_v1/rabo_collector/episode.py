from __future__ import annotations

import queue
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CollectorConfig
from .sensors import SensorBackend


class PhaseTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._name = "initializing"
        self._index = 0

    def set(self, name: str) -> None:
        with self._lock:
            if name != self._name:
                self._name = name
                self._index += 1

    def get(self) -> tuple[str, int]:
        with self._lock:
            return self._name, self._index


@dataclass
class RecordedEpisode:
    states: np.ndarray
    full_states: np.ndarray
    actions: np.ndarray
    phases: list[str]
    phase_indices: np.ndarray
    monotonic_times: np.ndarray
    sim_times: np.ndarray
    video_files: dict[str, Path]
    sample_timeouts: int
    stale_samples: int
    camera_stale_samples: int
    camera_skew_s: float
    joint_skew_s: float
    writer_queue_drops: int
    max_stamp_gap_s: float
    camera_reuse_counts: dict[str, int]

    @property
    def length(self) -> int:
        return int(self.states.shape[0])

    def discard(self) -> None:
        parents = {path.parent for path in self.video_files.values()}
        for parent in parents:
            shutil.rmtree(parent, ignore_errors=True)


class EpisodeRecorder:
    """固定频率记录最新传感器缓存，并在独立线程编码视频。"""

    def __init__(
        self,
        config: CollectorConfig,
        sensors: SensorBackend,
        phase: PhaseTracker,
    ) -> None:
        self.config = config
        self.sensors = sensors
        self.phase = phase
        self._stop_event = threading.Event()
        self._first_sample_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._writer_error: BaseException | None = None
        self._states: list[np.ndarray] = []
        self._full_states: list[np.ndarray] = []
        self._phases: list[str] = []
        self._phase_indices: list[int] = []
        self._times: list[float] = []
        self._sim_times: list[float] = []
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._video_paths: dict[str, Path] = {}
        self._frame_queue: queue.Queue = queue.Queue(maxsize=int(config.writer_queue_size))
        self._sample_timeouts = 0
        self._stale_samples = 0
        self._camera_stale_samples = 0
        self._camera_skew = 0.0
        self._joint_skew = 0.0
        self._writer_queue_drops = 0
        self._max_stamp_gap = 0.0
        self._camera_reuse_counts = {name: 0 for name in config.camera_topics}
        self._staging_dir: Path | None = None
        self._queue_full_reported = False
        self._stale_reported = False
        self._camera_skew_reported = False
        self._gap_reported = False

    def start(self) -> None:
        staging_root = self.config.root / ".staging"
        self._staging_dir = staging_root / f"episode-{uuid.uuid4().hex}"
        self._staging_dir.mkdir(parents=True, exist_ok=False)
        self._stop_event.clear()
        self._first_sample_event.clear()
        self.sensors.reset_clock()
        self._thread = threading.Thread(target=self._run, name="episode-recorder", daemon=True)
        self._writer_thread = threading.Thread(
            target=self._write_loop, name="episode-video-writer", daemon=True
        )
        self._writer_thread.start()
        self._thread.start()

        # 在专家任务开始前确认采集线程确实拿到一帧，避免整段动作执行完才发现0帧。
        first_sample_timeout = min(
            5.0, float(self.config.dataset.get("readiness_timeout_s", 30.0))
        )
        if not self._first_sample_event.wait(first_sample_timeout):
            self.abort()
            raise RuntimeError(
                "采集线程启动后未在"
                f"{first_sample_timeout:.1f}s内生成首帧；请检查相机解码和关节缓存"
            )
        if self._error is not None:
            error = self._error
            self.abort()
            raise RuntimeError("采集线程生成首帧失败") from error

    def _open_video_writer(self, name: str, frame: np.ndarray) -> cv2.VideoWriter:
        assert self._staging_dir is not None
        height = int(self.config.dataset["image_height"])
        width = int(self.config.dataset["image_width"])
        path = self._staging_dir / f"{name}.mp4"
        fourcc_name = str(self.config.dataset.get("video_fourcc", "mp4v"))
        if len(fourcc_name) != 4:
            raise ValueError("dataset.video_fourcc 必须是4个字符，例如 mp4v")
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc_name), self.config.fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频 {path}，请检查 OpenCV/FFmpeg 编码器")
        self._video_paths[name] = path
        self._writers[name] = writer
        return writer

    @staticmethod
    def _center_crop_resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """保持宽高比中心裁剪，禁止把16:9图像直接拉伸成其他比例。"""
        source_h, source_w = frame.shape[:2]
        source_ratio = source_w / source_h
        target_ratio = width / height
        if source_ratio > target_ratio:
            crop_w = max(1, int(round(source_h * target_ratio)))
            left = (source_w - crop_w) // 2
            frame = frame[:, left:left + crop_w]
        elif source_ratio < target_ratio:
            crop_h = max(1, int(round(source_w / target_ratio)))
            top = (source_h - crop_h) // 2
            frame = frame[top:top + crop_h, :]
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _run(self) -> None:
        dt = 1.0 / self.config.fps
        timeout = self.config.sample_wait_timeout_s
        last_sim: float | None = None
        try:
            while not self._stop_event.is_set():
                snapshot = self.sensors.sample_next(dt, timeout, self._stop_event)
                if snapshot is None:
                    if not self._stop_event.is_set():
                        self._sample_timeouts += 1
                    continue

                # 新项目按消息到达墙钟计算年龄。超限只做记录，不再丢弃整帧。
                joint_age = max((abs(a) for a in snapshot.joint_ages.values()), default=0.0)
                self._joint_skew = max(self._joint_skew, joint_age)
                if joint_age > self.config.max_joint_age_s:
                    self._stale_samples += 1
                    if not self._stale_reported:
                        self._stale_reported = True
                        stale_names = [
                            n
                            for n, a in snapshot.joint_ages.items()
                            if abs(a) > self.config.max_joint_age_s
                        ]
                        print(
                            f"[采集] 关节缓存年龄 {joint_age:.3f}s > "
                            f"{self.config.max_joint_age_s:.3f}s，仍保留该采样点："
                            f"{', '.join(stale_names[:6])}"
                        )

                for name, reused in snapshot.camera_reused.items():
                    if reused:
                        self._camera_reuse_counts[name] += 1

                # 跨相机 skew（相对主相机）。先记录但不在这里丢整帧，避免平台偶发
                # 腕相机抖动导致整个episode只剩极少样本；sidecar会给出质量判定。
                camera_skew = max(
                    (
                        abs(snapshot.sim_time - stamp)
                        for name, stamp in snapshot.image_stamps.items()
                        if name != self.config.primary_camera
                    ),
                    default=0.0,
                )
                self._camera_skew = max(self._camera_skew, camera_skew)
                if camera_skew > self.config.max_camera_skew_s:
                    self._camera_stale_samples += 1
                if camera_skew > self.config.max_camera_skew_s and not self._camera_skew_reported:
                    self._camera_skew_reported = True
                    print(
                        f"[采集] 跨相机时间偏差 {camera_skew:.3f}s > "
                        f"{self.config.max_camera_skew_s:.3f}s"
                    )

                # 一个队列条目包含三路相机同一采样点。队列满时整帧跳过，禁止只丢
                # 某一路视频而仍写入state，确保三路MP4与Parquet帧数始终一致。
                try:
                    self._frame_queue.put_nowait(snapshot.images)
                except queue.Full:
                    self._writer_queue_drops += 1
                    if not self._queue_full_reported:
                        self._queue_full_reported = True
                        print("[采集] 写盘队列已满，整条样本跳过（状态和三路视频均不写）")
                    continue

                if last_sim is not None:
                    gap = snapshot.sim_time - last_sim
                    self._max_stamp_gap = max(self._max_stamp_gap, gap)
                    if gap > self.config.max_stamp_gap_s and not self._gap_reported:
                        self._gap_reported = True
                        print(
                            f"[采集] 相邻样本仿真间隔 {gap:.3f}s > "
                            f"{self.config.max_stamp_gap_s:.3f}s"
                        )
                last_sim = snapshot.sim_time

                phase_name, phase_index = self.phase.get()
                self._states.append(snapshot.state.astype(np.float32, copy=True))
                self._full_states.append(snapshot.full_state.astype(np.float32, copy=True))
                self._phases.append(phase_name)
                self._phase_indices.append(phase_index)
                self._times.append(snapshot.monotonic_time)
                self._sim_times.append(snapshot.sim_time)
                self._first_sample_event.set()
        except BaseException as exc:
            self._error = exc
            self._stop_event.set()
            self._first_sample_event.set()

    def _write_loop(self) -> None:
        width = int(self.config.dataset["image_width"])
        height = int(self.config.dataset["image_height"])
        try:
            while True:
                item = self._frame_queue.get()
                if item is None:
                    break
                if self._writer_error is not None:
                    continue  # 已出错则丢弃后续帧，避免队列阻塞采样线程
                try:
                    images = item
                    for name in self.config.camera_topics:
                        frame = self._center_crop_resize(images[name], width, height)
                        writer = self._writers.get(name)
                        if writer is None:
                            writer = self._open_video_writer(name, frame)
                        writer.write(np.ascontiguousarray(frame, dtype=np.uint8))
                except BaseException as exc:
                    self._writer_error = exc
        finally:
            for writer in self._writers.values():
                writer.release()

    def stop(self) -> RecordedEpisode:
        # 1) 先停止采样线程。
        self._stop_event.set()
        self.sensors.wake()
        if self._thread is not None:
            self._thread.join(timeout=10)
        # 2) 采样停止后放哨兵，排空写盘队列。
        try:
            # 允许写盘线程先排空队列，再收到哨兵；不能用put_nowait后静默失败。
            self._frame_queue.put(None, timeout=10)
        except queue.Full:
            self.abort()
            raise RuntimeError("写盘队列无法排空")
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10)

        if self._thread is not None and self._thread.is_alive():
            self.abort()
            raise RuntimeError("采集线程无法停止")
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self.abort()
            raise RuntimeError("写盘线程无法停止")
        if self._error is not None:
            self.abort()
            raise RuntimeError("episode采集失败") from self._error
        if self._writer_error is not None:
            self.abort()
            raise RuntimeError("episode视频写盘失败") from self._writer_error
        if len(self._states) < 2:
            accepted = len(self._states)
            diagnostic = (
                f"accepted={accepted}, sample_timeouts={self._sample_timeouts}, "
                f"stale_samples={self._stale_samples}, "
                f"writer_queue_drops={self._writer_queue_drops}, "
                f"camera_reuse={self._camera_reuse_counts}"
            )
            self.abort()
            raise RuntimeError("episode少于2帧，无法生成动作序列；" + diagnostic)

        states = np.stack(self._states).astype(np.float32)
        full_states = np.stack(self._full_states).astype(np.float32)
        if states.shape[1] != len(self.config.state_names) or full_states.shape[1] != len(
            self.config.full_state_names
        ):
            self.abort()
            raise RuntimeError(
                f"state/full_state维度错误：{states.shape[1]}/{full_states.shape[1]}，"
                f"期望{len(self.config.state_names)}/{len(self.config.full_state_names)}"
            )
        # 脚本使用末端路径点控制而非关节遥操作，训练动作采用下一帧实际关节位置。
        # 这与关节位置行为克隆一致，也避免把笛卡尔6维命令误写入26维动作。
        actions = np.concatenate([states[1:], states[-1:]], axis=0)
        return RecordedEpisode(
            states=states,
            full_states=full_states,
            actions=actions,
            phases=list(self._phases),
            phase_indices=np.asarray(self._phase_indices, dtype=np.int64),
            monotonic_times=np.asarray(self._times, dtype=np.float64),
            sim_times=np.asarray(self._sim_times, dtype=np.float64),
            video_files=dict(self._video_paths),
            sample_timeouts=self._sample_timeouts,
            stale_samples=self._stale_samples,
            camera_stale_samples=self._camera_stale_samples,
            camera_skew_s=self._camera_skew,
            joint_skew_s=self._joint_skew,
            writer_queue_drops=self._writer_queue_drops,
            max_stamp_gap_s=self._max_stamp_gap,
            camera_reuse_counts=dict(self._camera_reuse_counts),
        )

    def abort(self) -> None:
        self._stop_event.set()
        self.sensors.wake()
        try:
            self._frame_queue.put(None, timeout=3)
        except queue.Full:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3)
        for writer in self._writers.values():
            writer.release()
        if self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
