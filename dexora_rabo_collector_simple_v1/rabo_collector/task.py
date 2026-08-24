from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import CollectorConfig
from .episode import PhaseTracker


@dataclass(frozen=True)
class NutMotion:
    name: str
    approach_z: float
    handoff_offset: float
    right_fingers: tuple[int, ...]
    right_strength: float
    fine_start_z: float
    fine_grasp_z: float
    left_strength: float
    right_retreat_z: float
    after_handoff: tuple[float, float, float, float, float, float] | None
    place_waypoints: tuple[
        tuple[float, float, float, float, float, float], ...
    ]
    retreat: tuple[float, float, float, float, float, float]


MOTIONS = {
    "B": NutMotion(
        "B",
        -0.331,
        0.0,
        (1, 2, 3, 4),
        1.0,
        -0.2,
        -0.178,
        0.6,
        0.12,
        (0.44, 0.06, -0.05, 0.0, 1.3, 1.57),
        (
            (0.38, 0.25, -0.22, 0.0, -0.8, 0.0),
        ),
        (0.38, 0.25, -0.12, 0.0, -0.8, 0.0),
    ),
    "C": NutMotion(
        "C",
        -0.33,
        0.0062,
        (1, 2, 3, 4),
        1.0,
        -0.25,
        -0.188,
        0.8,
        0.0962,
        (0.44, 0.06, -0.1, 0.0, 1.3, 1.57),
        (
            (0.34, 0.25, -0.12, 0.0, 0.0, 0.0),
            (0.34, 0.25, -0.2, 0.0, 0.0, 0.0),
        ),
        (0.34, 0.25, -0.1, 0.0, 0.0, 0.0),
    ),
    "A": NutMotion(
        "A",
        -0.328,
        0.0009,
        (1, 2, 3, 4, 5),
        1.0,
        -0.2,
        -0.180,
        0.5,
        0.0509,
        None,
        (
            (0.43, 0.25, -0.12, 0.0, 0.0, 0.0),
            (0.43, 0.25, -0.2, 0.0, 0.0, 0.0),
        ),
        (0.43, 0.25, -0.1, 0.0, 0.0, 0.0),
    ),
}


def _parallel(*functions: Callable[[], None]) -> None:
    errors: list[BaseException] = []
    lock = threading.Lock()

    def invoke(function: Callable[[], None]) -> None:
        try:
            function()
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=invoke, args=(function,))
        for function in functions
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError("并行动作失败") from errors[0]


class RaboNutHandoffTask:
    """双臂协作抓取并交接B、C、A三个螺母。"""

    def __init__(
        self,
        config: CollectorConfig,
        phase: PhaseTracker,
        abort_event: threading.Event | None = None,
    ):
        self.config = config
        self.phase = phase
        self._abort_event = abort_event or threading.Event()

        try:
            from rabo_dev_kit import SetEntityPose
            from rabo_robocap import (
                LinkerArmA7,
                LinkerHandO6Left,
                LinkerHandO6Right,
            )
        except ImportError as exc:
            raise RuntimeError(
                "缺少Rabo SDK，请在Rabo/ROS2运行环境执行"
            ) from exc

        rabo = config.rabo
        mode = str(rabo.get("mode", "sim"))

        self.mode = mode

        self.pose_setter = (
            SetEntityPose(world=rabo["world_id"])
            if mode == "sim"
            else None
        )

        self.right_arm = LinkerArmA7(
            robot_id=rabo["right_arm_id"],
            mode=mode,
        )
        self.left_arm = LinkerArmA7(
            robot_id=rabo["left_arm_id"],
            mode=mode,
        )
        self.right_hand = LinkerHandO6Right(
            robot_id=rabo["right_hand_id"],
            mode=mode,
        )
        self.left_hand = LinkerHandO6Left(
            robot_id=rabo["left_hand_id"],
            mode=mode,
        )

        self.right_arm_base = (-0.6816, -0.004)
        self._nut_positions: dict[str, list[float]] = {}

    def _check_abort(self) -> None:
        if self._abort_event.is_set():
            raise RuntimeError("相机停更，任务已中止")

    def read_full_state(self) -> list[float] | None:
        """按 config.full_state 顺序读取 4 个 SDK 设备的当前关节角。

        顺序：left_arm(7) + right_arm(7) + left_hand(11) + right_hand(11) = 36。
        任一设备返回空列表（SDK 错误信号）时整体返回 None。
        """
        parts = (
            self.left_arm.get_joint_angles(),
            self.right_arm.get_joint_angles(),
            self.left_hand.get_joint_angles(),
            self.right_hand.get_joint_angles(),
        )
        expected = (7, 7, 11, 11)
        if any(
            not isinstance(part, (list, tuple)) or len(part) != count
            for part, count in zip(parts, expected)
        ):
            return None
        return [float(value) for part in parts for value in part]

    def reset(self, rng: random.Random) -> dict[str, Any]:
        self.phase.set("reset_scene")

        if self.mode != "sim":
            self._nut_positions = {
                "B": [
                    -0.3413,
                    -0.1710,
                    0.2806,
                    0,
                    0,
                    0.5233,
                ],
                "A": [
                    -0.2286,
                    -0.0999,
                    0.2815,
                    0,
                    0,
                    0.5233,
                ],
                "C": [
                    -0.2975,
                    -0.0527,
                    0.2868,
                    0,
                    0,
                    0.5233,
                ],
            }

            return {
                "mode": "real",
                "note": "物体由操作员复位",
            }

        positions = {
            "B": [
                -0.3413 + rng.uniform(-0.02, 0.02),
                -0.1710 + rng.uniform(-0.02, 0.02),
                0.2806,
                0,
                0,
                0.5233,
            ],
            "A": [
                -0.2286 + rng.uniform(-0.03, 0.01),
                -0.0999 + rng.uniform(-0.04, 0.0),
                0.2819,
                0,
                0,
                0.5233,
            ],
            "C": [
                -0.2975 + rng.uniform(-0.02, 0.02),
                -0.0527 + rng.uniform(-0.02, 0.02),
                0.2872,
                0,
                0,
                0.5233,
            ],
        }

        for name, pose in positions.items():
            self.pose_setter.set(
                self.config.rabo["nut_ids"][name],
                tuple(pose),
            )

        self._nut_positions = positions

        time.sleep(0.5)

        return {
            "nut_poses": positions,
        }

    def _move_to_checked(
        self,
        arm,
        label: str,
        x: float,
        y: float,
        z: float,
        *,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> None:
        """保持原坐标不变；仅在SDK明确返回False时重试一次并停止坏示范。"""
        for attempt in range(2):
            self._check_abort()
            result = arm.move_to(
                x,
                y,
                z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
            )
            if result is not False:
                return
            print(f"[控制] {label} 第{attempt + 1}次返回False")
            if attempt == 0:
                time.sleep(0.10)
        raise RuntimeError(f"{label} 连续两次执行失败")

    def _move_joints_checked(self, arm, label: str, joints: list[float]) -> None:
        for attempt in range(2):
            self._check_abort()
            result = arm.move_joints(joints)
            if result is not False:
                return
            print(f"[控制] {label} 第{attempt + 1}次返回False")
            if attempt == 0:
                time.sleep(0.10)
        raise RuntimeError(f"{label} 连续两次执行失败")

    def _move_pose(
        self,
        arm,
        pose: tuple[
            float,
            float,
            float,
            float,
            float,
            float,
        ],
    ) -> None:
        x, y, z, roll, pitch, yaw = pose

        self._move_to_checked(
            arm,
            "左臂路径点",
            x,
            y,
            z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

    def pre_position(self) -> None:
        self._check_abort()
        self.phase.set("pre_position")

        self.right_hand.clench(
            0,
            0,
            0,
            0,
            0,
            0,
        )
        self.left_hand.clench(
            0,
            0,
            0,
            0,
            0,
            0,
        )

        def right() -> None:
            self._move_joints_checked(
                self.right_arm,
                "右臂预备位1",
                [-1.57, -1.5, 0, -1.57, 0, -1, 0]
            )
            self._move_joints_checked(
                self.right_arm,
                "右臂预备位2",
                [0, 0, 0, -2, 0, 1, 0]
            )

        def left() -> None:
            self._move_joints_checked(
                self.left_arm,
                "左臂预备位1",
                [0, -1.57, 0, 0, 0, 0, 0]
            )
            self._move_joints_checked(
                self.left_arm,
                "左臂预备位2",
                [-1.57, -0.7, 0, 0, 0, 0, 0]
            )

        _parallel(right, left)

    def _process_nut(
        self,
        motion: NutMotion,
    ) -> None:
        self._check_abort()
        name = motion.name
        position = self._nut_positions[name]
        offset = motion.handoff_offset

        # 右臂逼近螺母
        self.phase.set(f"approach_{name}")

        target_x = (
            self.right_arm_base[0]
            - position[0]
            + 0.06
        )

        # 仅螺母A的右手抓取X坐标减小0.02米
        if name == "A":
            target_x -= 0.02

        target_y = (
            self.right_arm_base[1]
            - position[1]
            - 0.01
        )

        self._move_to_checked(
            self.right_arm,
            f"右臂接近{name}",
            target_x,
            target_y,
            motion.approach_z,
            roll=0,
            pitch=0.8,
            yaw=0,
        )

        # 右手抓取螺母
        self.phase.set(f"right_grasp_{name}")

        self.right_hand.clench(
            thumb_rotation=1.0,
        )

        self.right_hand.grasp_force(
            strength=motion.right_strength,
            fingers=list(motion.right_fingers),
        )

        # 右臂提起，左臂同时前往交接区
        self.phase.set(
            f"lift_and_left_approach_{name}"
        )

        def right_lift() -> None:
            self._move_to_checked(
                self.right_arm,
                f"右臂{name}提起",
                -0.4,
                0.12,
                -0.03 + offset,
                roll=0,
                pitch=0.8,
                yaw=0,
            )

            self._move_to_checked(
                self.right_arm,
                f"右臂{name}进入交接位",
                -0.4,
                0,
                -0.03 + offset,
                roll=0,
                pitch=0.8,
                yaw=0,
            )

        def left_coarse() -> None:
            self._move_to_checked(
                self.left_arm,
                f"左臂{name}粗接近",
                0.43,
                0.3,
                -0.1 + offset,
                roll=0,
                pitch=1.3,
                yaw=1.57,
            )

        _parallel(
            right_lift,
            left_coarse,
        )

        # 输出右臂真实交接位姿
        right_pos, right_ori = (
            self.right_arm.get_pose()
        )

        print(
            f"[{name}右手交接位] "
            f"x={right_pos[0]:.4f}, "
            f"y={right_pos[1]:.4f}, "
            f"z={right_pos[2]:.4f} | "
            f"roll={right_ori[0]:.4f}, "
            f"pitch={right_ori[1]:.4f}, "
            f"yaw={right_ori[2]:.4f}"
        )

        # 左臂精调到接取位置
        self.phase.set(
            f"left_fine_approach_{name}"
        )

        self._move_to_checked(
            self.left_arm,
            f"左臂{name}精接近",
            0.44,
            0.06,
            motion.fine_start_z,
            roll=0,
            pitch=1.3,
            yaw=1.57,
        )

        # 左手只做预成形，不执行力控抓取
        self.left_hand.clench(
            0,
            0.5,
            0.3,
            0.3,
            0.3,
            0.3,
        )

        self._move_to_checked(
            self.left_arm,
            f"左臂{name}接取高度",
            0.44,
            0.06,
            motion.fine_grasp_z,
            roll=0,
            pitch=1.3,
            yaw=1.57,
        )

        # 交接顺序：
        # 1. 右手先松开
        # 2. 右臂向上抬起并离开
        # 3. 左手最后抓取
        self.phase.set(
            f"handoff_{name}_right_release"
        )

        self.right_hand.clench(
            thumb_rotation=None,
            thumb_bend=0,
            index=0,
            middle=0,
            ring=0,
            pinky=0,
        )

        time.sleep(0.10)

        self.phase.set(
            f"handoff_{name}_right_lift_clear"
        )

        self._move_to_checked(
            self.right_arm,
            f"右臂{name}释放后抬起",
            -0.4,
            0,
            motion.right_retreat_z,
            roll=0,
            pitch=0.8,
            yaw=0,
        )

        time.sleep(0.10)

        self.phase.set(
            f"handoff_{name}_left_grasp"
        )

        if name == "A":
            time.sleep(0.5)

        self.left_hand.clench(
            0,
            0.5,
            0.3,
            0.3,
            0.3,
            0.3,
        )

        self.left_hand.grasp_force(
            strength=motion.left_strength,
        )

        # 左手放置螺母
        self.phase.set(f"place_{name}")

        if motion.after_handoff is not None:
            self._move_pose(
                self.left_arm,
                motion.after_handoff,
            )

        for waypoint in motion.place_waypoints:
            self._move_pose(
                self.left_arm,
                waypoint,
            )

        # 左手释放螺母
        self.phase.set(f"release_{name}")

        self.left_hand.clench(
            0,
            0,
            0,
            0,
            0,
            0,
        )

        time.sleep(0.2)

        self._move_pose(
            self.left_arm,
            motion.retreat,
        )

    def run(self) -> None:
        self._check_abort()
        self.pre_position()

        for index, name in enumerate(
            ("B", "C", "A")
        ):
            self._check_abort()
            self._process_nut(
                MOTIONS[name]
            )

            if index < 2:
                self.phase.set(
                    f"wait_after_{name}"
                )
                time.sleep(1.0)

        self.phase.set("completed")
        time.sleep(0.15)

    def shutdown(self) -> None:
        for device in (
            self.right_arm,
            self.left_arm,
            self.right_hand,
            self.left_hand,
        ):
            try:
                device.shutdown()
            except Exception:
                pass


class MockNutHandoffTask:
    def __init__(
        self,
        phase: PhaseTracker,
        phase_duration_s: float = 0.08,
    ):
        self.phase = phase
        self.phase_duration_s = (
            phase_duration_s
        )

    def reset(
        self,
        rng: random.Random,
    ) -> dict[str, Any]:
        self.phase.set("reset_scene")
        time.sleep(self.phase_duration_s)

        return {
            "mock_seed_sample": rng.random(),
        }

    def run(self) -> None:
        self.phase.set("pre_position")
        time.sleep(self.phase_duration_s)

        for name in ("B", "C", "A"):
            for stage in (
                "approach",
                "right_grasp",
                "lift",
                "handoff",
                "place",
                "release",
            ):
                self.phase.set(
                    f"{stage}_{name}"
                )
                time.sleep(
                    self.phase_duration_s
                )

        self.phase.set("completed")
        time.sleep(self.phase_duration_s)

    def shutdown(self) -> None:
        return
