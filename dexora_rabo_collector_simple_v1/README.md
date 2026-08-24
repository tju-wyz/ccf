# Dexora Rabo Simple V1 数据采集器

这是为 Rabo 纯仿真双臂螺母任务重新构建的最小采集项目。它保留原 `task.py`
中的机器人、螺母 ID、运动轨迹和 26 维策略状态，只重写数据采集链路。

## 为什么不会再卡在“episode少于2帧”

旧版由主相机新时间戳驱动采样；相机低频、时间戳停止或条件变量被高频关节消息
反复唤醒时，任务执行期间可能一个样本也产不出来。

Simple V1 改成：

- 独立线程按墙钟 10 Hz 运行；
- 每个时刻读取三台相机和 36 路关节的最新缓存；
- 相机没有新帧时复用上一帧，不阻塞采样；
- 关节/相机缓存较旧时只记录质量统计，不删除样本；
- 开始专家任务前必须先成功生成首帧；
- 三路相机作为一个队列条目写盘，视频与 Parquet 帧数保持一致。

## 相机

默认配置使用已经在平台检查到的三路 `sensor_msgs/msg/Image`：

- `top`: `r6ef2dc_tp_cam_303d2b1ce0`
- `wrist_left`: `rbd03eb_tp_cam_069a6739f3`
- `wrist_right`: `r412d23_tp_cam_3c67aef2bc`

如果运行环境没有继承 `/gs_...` ROS namespace，需要把 `config.yaml` 中的话题改成
`ros2 topic list` 显示的完整绝对名称。

## 第一次运行

先验证 Python 数据管线，不控制真实仿真机器人：

```bash
cd /workspace/agent_system/dexora_rabo_collector_simple_v1
python collect.py --backend mock --episodes 1 --auto-success
python validate_dataset.py datasets/rabo_nut_handoff_simple_v1_3cam
```

然后进行一条正式示范：

```bash
python collect.py --backend ros2 --episodes 1 --auto-success
python validate_dataset.py datasets/rabo_nut_handoff_simple_v1_3cam
```

成功时应看到：

```text
采样模式：墙钟定频读取最新缓存；fps=10，相机=['top', 'wrist_left', 'wrist_right']
[1/1] 保存 episode 000000，...帧，success=True
```

每条 episode 的 `meta/episode_details/*.json` 会记录 `camera_reuse_counts`。少量重复帧
可以接受；如果某台相机几乎每一帧都被复用，应检查该话题，或暂时从配置中删除该
相机再收集。

## 推荐的初赛采集量

先跑 3 条并逐条验证视频；确认轨迹与动作正确后，收集 30～50 条成功示范。不要把
旧采集器生成的半成品数据混入本目录。
