from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 LeRobot v2.1 数据、视频与索引一致性")
    parser.add_argument("dataset", type=Path)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    root = args.dataset.resolve()
    info = load_json(root / "meta" / "info.json")
    errors: list[str] = []
    warnings: list[str] = []
    total_frames = 0
    expected_global_index = 0
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    full_state_dim = int(info["features"]["observation.full_state"]["shape"][0])
    camera_keys = [
        key.removeprefix("observation.images.")
        for key in info["features"]
        if key.startswith("observation.images.")
    ]

    for episode_index in range(int(info["total_episodes"])):
        chunk = episode_index // int(info["chunks_size"])
        parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        if not parquet_path.exists():
            errors.append(f"缺少 {parquet_path}")
            continue
        table = pq.read_table(parquet_path)
        length = table.num_rows
        total_frames += length
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        full_states = np.asarray(table["observation.full_state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states.shape != (length, state_dim) or actions.shape != (length, state_dim):
            errors.append(f"episode {episode_index}: state/action维度错误")
        if full_states.shape != (length, full_state_dim):
            errors.append(f"episode {episode_index}: full_state维度错误")
        if not np.isfinite(states).all() or not np.isfinite(full_states).all() or not np.isfinite(actions).all():
            errors.append(f"episode {episode_index}: 存在NaN或Inf")
        indices = np.asarray(table["index"])
        expected = np.arange(expected_global_index, expected_global_index + length)
        if not np.array_equal(indices, expected):
            errors.append(f"episode {episode_index}: 全局index不连续")
        expected_global_index += length

        for camera in camera_keys:
            video_path = (
                root / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{camera}"
                / f"episode_{episode_index:06d}.mp4"
            )
            if not video_path.exists():
                errors.append(f"缺少 {video_path}")
                continue
            capture = cv2.VideoCapture(str(video_path))
            video_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                errors.append(f"无法解码 {video_path}")
            if video_frames and video_frames != length:
                errors.append(
                    f"episode {episode_index}/{camera}: 视频{video_frames}帧，Parquet {length}帧"
                )
            if ok:
                feature = info["features"][f"observation.images.{camera}"]
                if list(frame.shape) != list(feature["shape"]):
                    errors.append(f"episode {episode_index}/{camera}: 图像尺寸不符")

    if total_frames != int(info["total_frames"]):
        errors.append(f"info.total_frames={info['total_frames']}，实际={total_frames}")
    staging = root / ".staging"
    if staging.exists() and any(staging.iterdir()):
        warnings.append(f"{staging} 中有未提交的临时episode，可人工检查后删除")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"校验失败：{len(errors)}项错误，{len(warnings)}项警告")
        return 1
    print(
        f"校验通过：{info['total_episodes']}个episode，{total_frames}帧，"
        f"状态/动作{state_dim}维，full_state {full_state_dim}维，{len(camera_keys)}路视频"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
