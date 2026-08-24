# TECH_SPEC — 基于 task.py 重写其余采集文件的技术约定

> 目的：把 `rabo_collector/task.py` 作为唯一“专家策略”，围绕它重写采集链路，
> 避免此前反复出现的相机饿死、episode 0 帧、停更误判、帧数不一致等问题。
> 交给 codex 时，请要求它**逐条遵守**本文件。

## 1. 总体架构与职责划分

| 文件 | 职责 | 是否需重写 |
|---|---|---|
| `rabo_collector/task.py` | 专家策略：双臂抓取/交接/放置 B、C、A 螺母 | **保留，不动** |
| `rabo_collector/config.py` | 加载并校验 `config.yaml`，提供 `CollectorConfig` | 重写 |
| `rabo_collector/sensors.py` | 只订阅相机；关节状态通过 task 的 SDK 服务读取 | 重写 |
| `rabo_collector/episode.py` | 采样线程、视频编码线程、`PhaseTracker`、`EpisodeRecorder` | 重写 |
| `rabo_collector/lerobot_writer.py` | 写 LeRobot v2.1 格式的 Parquet + 视频 + meta | 重写 |
| `collect.py` | 入口：装配 task + sensors + recorder + writer，跑 N 条 episode | 重写 |
| `config.yaml` | 采集参数 + 相机/关节/机器人 ID | 重写 |
| `validate_dataset.py` | 校验 Parquet/视频/meta 一致性 | 重写 |

## 2. task.py 对外契约（其余文件依赖这些，不得改动 task.py）

```python
from .config import CollectorConfig
from .episode import PhaseTracker

class RaboNutHandoffTask:
    def __init__(self, config: CollectorConfig, phase: PhaseTracker,
                 abort_event: threading.Event | None = None): ...

    def reset(self, rng: random.Random) -> dict: ...
    # sim 模式用 SetEntityPose 随机化 B/A/C 螺母位姿，返回 randomization dict
    # 非 sim 模式返回 {"mode": "real", "note": ...}

    def run(self) -> None: ...
    # 执行 pre_position() + 依次处理 B、C、A 三个螺母
    # 内部用 phase.set(...) 标记阶段
    # 每次动作前调 self._check_abort()，abort_event 置位则抛 RuntimeError

    def read_full_state(self) -> list[float] | None: ...
    # 返回 36 维关节角：left_arm(7) + right_arm(7) + left_hand(11) + right_hand(11)
    # 任一设备返回空/维度不对 → 返回 None

    def shutdown(self) -> None: ...
```

**关键点（codex 必须遵守）：**

- task 内部通过 `phase.set("阶段名")` 写阶段标记，采集链路只读 `PhaseTracker.get()`，不要反向调用。
- `abort_event` 是采集链路（相机停更检测）通知 task 停止的通道；task 每次动作前检查它。
- task 依赖 Rabo SDK：`rabo_dev_kit.SetEntityPose`、`rabo_robocap.LinkerArmA7 / LinkerHandO6Left / LinkerHandO6Right`。

## 3. config.yaml 契约（CollectorConfig 必须暴露的字段）

```yaml
dataset:
  root: ./datasets/xxx           # 输出目录
  fps: 8                         # 采样/视频帧率（Hz）
  chunk_size: 1000               # 每 chunk 的 episode 数
  robot_type: linker_a7_o6_bimanual
  task: "..."
  language_variants: ["...", "..."]
  cameras:                       # 相机名 -> ROS topic
    top: r6ef2dc_tp_cam_303d2b1ce0
  primary_camera: top
  image_width: 480
  image_height: 270
  video_fourcc: mp4v
  readiness_timeout_s: 30.0
  camera_ready_min_frames: 5     # 每路相机就绪需连续收到的新帧数
  camera_stall_timeout_s: 3.0    # 相机停更判定阈值（秒）
  sample_wait_timeout_s: 0.5
  ring_buffer_depth: 4
  writer_queue_size: 64
  max_joint_age_s: 1.0
  max_camera_skew_s: 1.0
  max_stamp_gap_s: 0.6           # 必须 >= 1/fps，否则每次采样都误报
state:
  active_hand_indices: [0, 1, 3, 5, 7, 9]   # O6 手 11 关节里取 6 个
  hand_joint_names: [thumb_cmc_yaw, ..., pinky_dip]  # 11 个
  joints:
    left_arm:  [{name, topic, ros_name} x7]
    right_arm: [{name, topic, ros_name} x7]
    left_hand: [{name, topic, ros_name} x11]
    right_hand:[{name, topic, ros_name} x11]
rabo:
  mode: sim
  world_id: ...
  right_arm_id / left_arm_id / right_hand_id / left_hand_id: ...
  nut_ids: {B: ..., A: ..., C: ...}
collection:
  episodes: 10
  seed: ...
  interactive_success: true
  discard_failed: true
  wait_between_episodes_s: 1.0
```

**维度约定（必须精确）：**

- `full_state_names` = 36（7+7+11+11）
- `state_names` = 26（7+7+6+6，手部按 `active_hand_indices` 从 11 取 6）
- 顺序严格为 `left_arm → right_arm → left_hand → right_hand`，与 `task.read_full_state()` 返回顺序一致。

## 4. 采集链路设计要点（最重要）

1. **采样线程墙钟定频**：`dt = 1/fps`，每拍读“最新缓存”，**绝不等待下一张新相机帧**。
2. **相机只订阅、关节用服务读**：只创建订阅 `cameras` 的 node；关节状态**不订阅任何 joint topic**，
   改为调用 `task.read_full_state()`。这是历史上 72 路关节订阅挤占传输层、把相机饿死的根因修复，**绝不能回退**。
3. **相机 QoS 用 RELIABLE**（与平台发布端对齐），独立 node + 独立 executor 线程。
4. **就绪判定**：每路相机连续收到 ≥ `camera_ready_min_frames` 帧才算 ready。
5. **停更保护**：相机超过 `camera_stall_timeout_s` 无新帧 → 置 `abort_event` → task 抛异常终止 episode
   （阈值要放宽到能容忍运动掉帧，1s 太严）。
6. **非交互运行不臆断成功**：`success=False`，`failure_reason="非交互运行，未确认成功"`，由人工看视频复核。
7. **视频与 Parquet 帧数严格一致**：三路相机作为**一个队列条目**写盘；队列满时**整帧跳过**（状态和三路视频都不写），
   禁止只丢某一路视频。
8. **相机解码不依赖 cv_bridge**：手写 `image_to_bgr8`（支持 bgr8/rgb8/bgra8/rgba8/mono8）。
9. **图像尺寸统一**：中心裁剪 + resize 到 `image_width × image_height`（保持宽高比，禁止拉伸）。

## 5. LeRobot v2.1 数据格式

- **Parquet 列**：`observation.state`(26)、`observation.full_state`(36)、`action`(26)、`timestamp`、
  `frame_index`、`episode_index`、`index`、`task_index`、`phase`、`phase_index`、
  `sensor_monotonic_time`、`sensor_sim_time`。
- **action 定义**：`actions = concat([states[1:], states[-1:]])` —— 即“下一帧的关节位置”，
  关节位置行为克隆的标准做法（任务用末端路径点控制，不是关节遥操作，不能用笛卡尔命令当 action）。
- **视频路径**：`videos/chunk-{NNN}/observation.images.{camera}/episode_{index:06d}.mp4`。
- **meta**：`info.json`（features/total_episodes/total_frames/fps/splits 等）、`stats.json`、
  `episodes.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl`、`modality.json`、
  `episode_details/episode_{index}.json`（sidecar，记录 `camera_reuse_counts`、`sim_fps`、`quality_pass` 等）。

## 6. 必须避免的历史坑

1. ❌ 不要订阅 36/72 路关节 topic —— 会挤占传输层把相机饿死。
2. ❌ 不要用主相机 header.stamp 驱动采样 —— 相机低频/停顿时 episode 会 0 帧。
3. ❌ 相机 `always_on` 必须为 true（场景侧配置），否则挂静态底座的 top 相机无触发源、永远收不到帧。
4. ❌ `camera_stall_timeout_s` 别设太严（运动时相机掉帧，1s 会误判停更中止任务）。
5. ❌ `max_stamp_gap_s` 必须 ≥ `1/fps`，否则每个采样点都刷误报警告。
6. ❌ 非交互运行别 `auto-success`。
7. ❌ 相机分辨率别超平台范围（宽/高 1~480）。
8. ❌ 三路视频帧数必须与 Parquet 一致，不能只丢某一路。
