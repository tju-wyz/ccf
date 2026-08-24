from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from .config import CollectorConfig


@dataclass(frozen=True)
class SensorSnapshot:
    """一个固定频率采样点。

    本项目有意不使用相机 header.stamp 驱动采样。Rabo 中相机可能低频、延迟或
    暂时不更新，因此采样线程按墙钟运行，并在每个采样点读取最新缓存。
    """

    sim_time: float
    monotonic_time: float
    state: np.ndarray
    full_state: np.ndarray
    images: dict[str, np.ndarray]
    joint_ages: dict[str, float]
    image_stamps: dict[str, float]
    camera_reused: dict[str, bool]


class SensorBackend(Protocol):
    def start(self) -> None: ...
    def wait_ready(self, timeout_s: float) -> None: ...
    def reset_clock(self) -> None: ...
    def sample_next(
        self, dt: float, timeout_s: float, stop_event: threading.Event
    ) -> SensorSnapshot | None: ...
    def wake(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class _Entry:
    seq: int
    source_stamp: float
    arrival_time: float
    value: Any


class _Ring:
    """只用于保存少量最新消息；采样时直接取 newest。"""

    def __init__(self, maxlen: int):
        self._data: collections.deque[_Entry] = collections.deque(maxlen=maxlen)
        self._seq = 0

    def append(self, source_stamp: float, value: Any) -> int:
        self._seq += 1
        self._data.append(
            _Entry(
                seq=self._seq,
                source_stamp=float(source_stamp),
                arrival_time=time.monotonic(),
                value=value,
            )
        )
        return self._seq

    def __len__(self) -> int:
        return len(self._data)

    def newest(self) -> _Entry | None:
        return self._data[-1] if self._data else None


def header_stamp_seconds(msg: Any) -> float:
    """读取 ROS header.stamp；无时间戳时回退到消息到达时的墙钟。"""

    try:
        stamp = msg.header.stamp
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return time.monotonic()
    if sec == 0 and nanosec == 0:
        return time.monotonic()
    return float(sec) + float(nanosec) * 1e-9


def image_to_bgr8(msg: Any) -> np.ndarray:
    """不依赖 cv_bridge 解码 sensor_msgs/Image。"""

    encoding = str(msg.encoding).lower()
    layouts = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }
    if encoding not in layouts:
        raise ValueError(
            f"不支持相机编码 {encoding!r}；请使用 bgr8/rgb8/bgra8/rgba8/mono8，"
            "深度图请使用独立深度管线"
        )
    channels = layouts[encoding]
    row_bytes = int(msg.width) * channels
    step = int(msg.step)
    if step < row_bytes:
        raise ValueError(f"Image.step={step} 小于有效行字节数 {row_bytes}")
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    required = int(msg.height) * step
    if raw.size < required:
        raise ValueError(f"Image.data={raw.size}字节，小于height*step={required}")

    rows = raw[:required].reshape(int(msg.height), step)
    pixels = rows[:, :row_bytes].reshape(int(msg.height), int(msg.width), channels)
    if encoding == "rgb8":
        pixels = pixels[..., ::-1]
    elif encoding == "rgba8":
        pixels = pixels[..., [2, 1, 0]]
    elif encoding == "bgra8":
        pixels = pixels[..., :3]
    elif encoding == "mono8":
        pixels = np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(pixels, dtype=np.uint8).copy()


class CameraStallError(RuntimeError):
    """相机停更（超过阈值无新帧），采集必须立即终止。"""


class Ros2SensorBackend:
    """只订阅相机、关节状态由外部 SDK 服务读取的 ROS2 后端。

    本后端**不再订阅任何关节话题**：关节状态由外部 ``joint_reader`` 通过
    Rabo SDK 的 ``get_joint_angles()`` 服务按需读取，避免 36 路高频关节消息
    挤占相机图像流。相机使用 RELIABLE QoS，与平台发布端对齐。

    ``joint_reader`` 返回 36 维 ``full_state``（顺序同 ``config.full_state_names``），
    失败返回 ``None``（跳过该采样点）。
    """

    def __init__(
        self,
        config: CollectorConfig,
        joint_reader: Callable[[], np.ndarray | None] | None = None,
        abort_event: threading.Event | None = None,
    ):
        self.config = config
        self._joint_reader = joint_reader
        self._abort_event = abort_event or threading.Event()
        self._cond = threading.Condition()
        depth = max(2, int(config.ring_buffer_depth))
        self._camera_rings = {name: _Ring(depth) for name in config.camera_topics}
        self._camera_msg_count = {name: 0 for name in config.camera_topics}
        self._camera_last_arrival = {name: 0.0 for name in config.camera_topics}

        self._camera_node = None
        self._camera_executor = None
        self._camera_thread: threading.Thread | None = None
        self._started = False
        self._closed = False

        self._clock_lock = threading.Lock()
        self._episode_wall_t0 = 0.0
        self._next_sample_wall = 0.0
        self._sample_index = 0
        self._last_camera_seq = {name: -1 for name in config.camera_topics}

    def start(self) -> None:
        if self._started:
            return
        try:
            import rclpy
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
            from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError("ROS2采集需要 rclpy 与 sensor_msgs") from exc

        if not rclpy.ok():
            rclpy.init(args=None)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=6,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._camera_node = rclpy.create_node("dexora_simple_camera_recorder")

        for name, topic in self.config.camera_topics.items():
            group = MutuallyExclusiveCallbackGroup()
            self._camera_node.create_subscription(
                Image,
                topic,
                lambda msg, camera_name=name: self._on_image(camera_name, msg),
                qos,
                callback_group=group,
            )

        try:
            self._camera_executor = MultiThreadedExecutor(num_threads=2)
        except Exception:
            self._camera_executor = SingleThreadedExecutor()

        self._camera_executor.add_node(self._camera_node)
        self._camera_thread = threading.Thread(
            target=self._camera_executor.spin,
            name="dexora-simple-cameras",
            daemon=True,
        )
        self._camera_thread.start()
        self._started = True

    def _on_image(self, name: str, msg: Any) -> None:
        with self._cond:
            self._camera_rings[name].append(header_stamp_seconds(msg), msg)
            self._camera_msg_count[name] += 1
            self._camera_last_arrival[name] = time.monotonic()
            self._cond.notify_all()

    def _camera_not_ready_locked(self) -> list[str]:
        need = int(self.config.dataset.get("camera_ready_min_frames", 5))
        return [
            f"camera:{name}"
            for name in self.config.camera_topics
            if self._camera_msg_count[name] < need
        ]

    def wait_ready(self, timeout_s: float) -> None:
        need = int(self.config.dataset.get("camera_ready_min_frames", 5))
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                missing = self._camera_not_ready_locked()
                if not missing:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "等待相机就绪超时，仍缺少："
                        + ", ".join(missing[:16])
                        + f"（每路需收到≥{need}帧新图）"
                    )
                self._cond.wait(remaining)

    def reset_clock(self) -> None:
        now = time.monotonic()
        with self._clock_lock:
            self._episode_wall_t0 = now
            self._next_sample_wall = now
            self._sample_index = 0
            self._last_camera_seq = {name: -1 for name in self.config.camera_topics}

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    @staticmethod
    def _wait_for_tick(target: float, stop_event: threading.Event) -> bool:
        while True:
            if stop_event.is_set():
                return False
            remaining = target - time.monotonic()
            if remaining <= 0:
                return True
            stop_event.wait(min(remaining, 0.02))

    def _read_joints(self) -> np.ndarray | None:
        if self._joint_reader is None:
            return None
        try:
            values = self._joint_reader()
        except Exception as exc:
            raise RuntimeError("读取关节状态失败") from exc
        if values is None:
            return None
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(self.config.full_state_names),):
            raise RuntimeError(
                f"关节读数维度错误：{values.shape}，"
                f"期望({len(self.config.full_state_names)},)"
            )
        return values

    def sample_next(
        self, dt: float, timeout_s: float, stop_event: threading.Event
    ) -> SensorSnapshot | None:
        del timeout_s  # 固定频率采样不等待新相机帧。
        with self._clock_lock:
            target = self._next_sample_wall
        if not self._wait_for_tick(target, stop_event):
            return None

        now = time.monotonic()
        stall_timeout = float(self.config.dataset.get("camera_stall_timeout_s", 1.0))

        with self._cond:
            for name in self.config.camera_topics:
                last = self._camera_last_arrival[name]
                if last > 0.0 and now - last > stall_timeout:
                    self._abort_event.set()
                    raise CameraStallError(
                        f"相机{name}超过{stall_timeout:.1f}s无新帧，终止episode"
                    )
            camera_entries = {
                name: ring.newest() for name, ring in self._camera_rings.items()
            }

        if any(entry is None for entry in camera_entries.values()):
            return None

        full_values = self._read_joints()
        if full_values is None:
            return None

        with self._clock_lock:
            index = self._sample_index
            self._sample_index += 1
            self._next_sample_wall += dt
            # 编码过慢时不补发一串突发样本，直接从当前墙钟继续。
            if self._next_sample_wall < now - dt:
                self._next_sample_wall = now + dt
            sim_time = index * dt

        images: dict[str, np.ndarray] = {}
        camera_reused: dict[str, bool] = {}
        image_stamps: dict[str, float] = {}
        for name, entry in camera_entries.items():
            if entry is None:
                continue
            try:
                images[name] = image_to_bgr8(entry.value)
            except Exception as exc:
                raise RuntimeError(f"解码相机{name}失败") from exc
            camera_reused[name] = self._last_camera_seq[name] == entry.seq
            self._last_camera_seq[name] = entry.seq
            age = max(0.0, now - entry.arrival_time)
            image_stamps[name] = sim_time - age

        by_name = dict(zip(self.config.full_state_names, full_values))
        state_values = [by_name[name] for name in self.config.state_names]
        joint_ages = {name: 0.0 for name in self.config.full_state_names}

        return SensorSnapshot(
            sim_time=sim_time,
            monotonic_time=now,
            state=np.asarray(state_values, dtype=np.float32),
            full_state=full_values,
            images=images,
            joint_ages=joint_ages,
            image_stamps=image_stamps,
            camera_reused=camera_reused,
        )

    @staticmethod
    def _shutdown_executor(executor: Any) -> None:
        try:
            executor.shutdown(timeout_sec=5.0)
        except TypeError:
            executor.shutdown()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.wake()

        if self._camera_executor is not None:
            try:
                self._shutdown_executor(self._camera_executor)
            except Exception as exc:
                print(f"[清理] camera executor shutdown失败：{exc!r}")
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=5.0)
        if self._camera_thread is not None and self._camera_thread.is_alive():
            # 仍有回调运行时销毁node会触发 cannot use Destroyable。
            print("[清理] camera executor线程仍在运行，跳过node销毁")
        else:
            if self._camera_executor is not None and self._camera_node is not None:
                try:
                    self._camera_executor.remove_node(self._camera_node)
                except Exception:
                    pass
            if self._camera_node is not None:
                try:
                    self._camera_node.destroy_node()
                except Exception as exc:
                    print(f"[清理] camera node销毁失败：{exc!r}")

        self._camera_node = None
        self._camera_executor = None
        self._camera_thread = None


class MockSensorBackend:
    """不依赖ROS2的端到端自测后端。"""

    def __init__(self, config: CollectorConfig):
        self.config = config
        self._started_at = 0.0
        self._next_sample_wall = 0.0
        self._index = 0

    def start(self) -> None:
        self.reset_clock()

    def wait_ready(self, timeout_s: float) -> None:
        del timeout_s

    def reset_clock(self) -> None:
        self._started_at = time.monotonic()
        self._next_sample_wall = self._started_at
        self._index = 0

    def wake(self) -> None:
        return

    def sample_next(
        self, dt: float, timeout_s: float, stop_event: threading.Event
    ) -> SensorSnapshot | None:
        del timeout_s
        if not Ros2SensorBackend._wait_for_tick(self._next_sample_wall, stop_event):
            return None
        index = self._index
        self._index += 1
        self._next_sample_wall = self._started_at + self._index * dt
        sim_time = index * dt
        now = time.monotonic()
        state = np.sin(sim_time + np.arange(26, dtype=np.float32) * 0.07)
        full_state = np.sin(sim_time + np.arange(36, dtype=np.float32) * 0.05)
        height = int(self.config.dataset["image_height"])
        width = int(self.config.dataset["image_width"])
        images: dict[str, np.ndarray] = {}
        for camera_index, name in enumerate(self.config.camera_topics):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., camera_index % 3] = int((index * 7 + camera_index * 50) % 255)
            images[name] = frame
        return SensorSnapshot(
            sim_time=sim_time,
            monotonic_time=now,
            state=state.astype(np.float32),
            full_state=full_state.astype(np.float32),
            images=images,
            joint_ages={name: 0.0 for name in self.config.full_state_names},
            image_stamps={name: sim_time for name in self.config.camera_topics},
            camera_reused={name: False for name in self.config.camera_topics},
        )

    def close(self) -> None:
        return
