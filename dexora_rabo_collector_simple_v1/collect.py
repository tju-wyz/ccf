from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

from rabo_collector.config import load_config
from rabo_collector.episode import EpisodeRecorder, PhaseTracker
from rabo_collector.lerobot_writer import LeRobotV21Writer
from rabo_collector.sensors import MockSensorBackend, Ros2SensorBackend
from rabo_collector.task import MockNutHandoffTask, RaboNutHandoffTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 Rabo 双臂螺母交接 LeRobot v2.1 数据集")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--backend", choices=("ros2", "mock"), default="ros2")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-failed", action="store_true", help="保存失败或人工判失败的episode")
    return parser.parse_args()


def ask_success() -> tuple[bool, str]:
    if not sys.stdin.isatty():
        # 非交互运行不臆断成功；成功与否由人工复核视频判定。
        return False, "非交互运行，未确认成功"
    answer = input("本条是否完整成功？[y/N] ").strip().lower()
    if answer in {"y", "yes", "1", "是"}:
        return True, ""
    reason = input("失败原因（可留空）：").strip()
    return False, reason


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    episode_count = args.episodes if args.episodes is not None else int(config.collection["episodes"])
    base_seed = args.seed if args.seed is not None else int(config.collection["seed"])
    keep_failed = args.keep_failed or not bool(config.collection.get("discard_failed", False))

    if episode_count <= 0:
        raise ValueError("--episodes 必须大于0")

    phase = PhaseTracker()
    abort_event = threading.Event()
    if args.backend == "mock":
        sensors = MockSensorBackend(config)
        task = MockNutHandoffTask(phase)
    else:
        # Rabo SDK 会自行初始化 rclpy，所以先创建设备，再创建只订阅相机的节点。
        # 关节状态复用 task 已创建的 4 个 SDK 设备（get_joint_angles），
        # 采集器自身不再订阅 36 路关节话题。
        task = RaboNutHandoffTask(config, phase, abort_event=abort_event)
        sensors = Ros2SensorBackend(
            config,
            joint_reader=task.read_full_state,
            abort_event=abort_event,
        )

    writer = LeRobotV21Writer(config)
    try:
        sensors.start()
        print("等待相机话题就绪……")
        sensors.wait_ready(float(config.dataset["readiness_timeout_s"]))
        print(
            "采样模式：墙钟定频读取最新缓存；"
            f"fps={config.fps}，相机={list(config.camera_topics)}"
        )
        print(f"开始采集：{episode_count} 条，输出目录：{config.root}")
        for local_index in range(episode_count):
            rng = random.Random(base_seed + writer.next_episode_index())
            randomization = task.reset(rng)
            language_variants = list(config.dataset.get("language_variants") or [config.dataset["task"]])
            task_text = language_variants[rng.randrange(len(language_variants))]
            recorder = EpisodeRecorder(config, sensors, phase)
            recorder.start()
            control_error: BaseException | None = None
            try:
                task.run()
            except KeyboardInterrupt:
                recorder.abort()
                raise
            except BaseException as exc:
                control_error = exc

            try:
                episode = recorder.stop()
            except BaseException:
                if control_error is not None:
                    raise RuntimeError("控制与采集均失败") from control_error
                raise

            if control_error is None:
                success, failure_reason = ask_success()
            else:
                success, failure_reason = False, repr(control_error)

            if not success and not keep_failed:
                episode.discard()
                print(f"[{local_index + 1}/{episode_count}] 已丢弃失败episode：{failure_reason}")
            else:
                saved_index = writer.add_episode(
                    episode,
                    task_text=task_text,
                    success=success,
                    failure_reason=failure_reason,
                    randomization=randomization,
                )
                print(
                    f"[{local_index + 1}/{episode_count}] 保存 episode {saved_index:06d}，"
                    f"{episode.length}帧，success={success}"
                )
            if control_error is not None:
                raise RuntimeError("机器人控制流程失败，已停止批量采集") from control_error
            time.sleep(float(config.collection.get("wait_between_episodes_s", 0)))
    except KeyboardInterrupt:
        print("收到中断，已停止采集。已提交的episode不会丢失。")
        return 130
    finally:
        try:
            # 每批结束统一扫描一次，避免每保存一条都重算全数据集统计量。
            writer.rebuild_metadata()
        finally:
            try:
                sensors.close()
            finally:
                task.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
