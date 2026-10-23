# LMDB 采集数据读取

本工具支持 `indexed_npz_lmdb_v1`：这是当前采集程序使用的结构，并非所有 LMDB 数据库共有的激光数据标准。LMDB 规定键值存储方式，不规定图像、点云或标定字段。

## 界面操作

左侧“数据来源与配置”中：

1. 数据格式选择 **LMDB 采集目录（index + shards）**。
2. 选择采集根目录，例如 `/home/qwer/Downloads/zgc/lmdb_20260730T122821Z`，不要选择其中单独的 `index.lmdb`。
3. 保持勾选 **读取 LMDB 数据（只读源数据库）**；点击“完整流程”选中语义、实例和时序优化，并保持勾选“生成标签与可视化”。
4. 设置导出目录。问题帧留空读取全部，或输入 `63 79 824` 读取指定帧。
5. 点击开始执行。结果页可以按原始全局帧号加载预览。

配置文件、摘要、LMDB 选择和阶段控件都在左侧；右侧留给日志与结果。左侧内容超出窗口高度时可滚动。

## 当前能力范围

LMDB 界面通过 `annotate` 入口执行完整标注，全部阶段可选择；详见 [完整流程](fresh_lmdb_workflow.md)。

下面的 `lmdb` 命令仍是独立原始导入器，仅导出图像、点云、标定、时间戳、位姿和原有语义 PNG；它本身不生成标签。原始 PNG 的未知类别编码不套用到新标注。

## 目录与索引

```text
recording/
  index.lmdb/
    data.mdb
    lock.mdb
  shards/
    shard_000000.lmdb/
      data.mdb
      lock.mdb
    shard_000001.lmdb/
      ...
```

`index.lmdb` 中使用 `frame_index/<12位全局帧号>` 键。值为 MessagePack 字典，必需字段：

| 字段 | 含义 |
| --- | --- |
| `global_frame_id` | 全局整数帧号；读取后仍保留原编号，不因选择子集重新编号 |
| `shard_id` | 分片整数编号，对应 `shards/shard_XXXXXX.lmdb` |
| `lmdb_key` | 该帧在分片里的完整键，通常以 `frame/` 开头 |
| `timestamp_ns` | 原始索引时间戳，纳秒 |
| `payload_sha256` | 可选；存在时会严格核对分片载荷哈希 |

分片键对应的值为 NPZ 字节流，使用 `allow_pickle=False` 解码。不会修改原始 LMDB；不创建或写入它的锁文件。

## 帧载荷字段

| 字段 | 结构 | 说明 |
| --- | --- | --- |
| `lidar_points` | `[N,D]`，`D≥3` | 每行一个激光点；前三列按原值保存为 XYZ，单位米，其余列原样保留，不擅自解释 |
| `image_jpeg_bytes` | 一维 `uint8` | JPEG 图像原始字节；原样导出，不重新编码 |
| `cam_intrinsic` | `[3,3]` | 相机内参；像素坐标顺序为 `(u,v)`，对应图像列、行 |
| `cam_extrinsic` | `[4,4]` | 点云坐标到相机坐标的变换；当前投影计算 `p_camera=R*p_cloud+t` |
| `lidar_timestamp_ns` | `int64` 标量 | 点云时间戳，纳秒；保留原时钟数值 |
| `pose_world` | 可选 `[4]` | 原始位姿数组，按原值保留；不改变其参考系或重估里程计 |
| `image_timestamp_ns` | 可选标量 | 原始图像时间戳 |
| `image_time_offset_ms` | 可选标量 | 原始图像时间偏移，毫秒；不做额外同步修正 |
| `semantic_class_png_bytes` | 可选一维字节 | 原有语义图；原样保留，类别 ID 不映射到本工具的新标签体系 |

`N` 为当前帧点数，可逐帧变化；`D` 为点属性列数。
点云坐标不进行轴变换；BEV 预览按当前约定显示 X 向上、Y 向左。相机内参与图像须匹配；当前预览使用针孔投影，不额外校正畸变。

## 导出结果

```text
/path/to/lidar_data/jobs/lmdb_readout/
  index.json                # 原始帧号、图像路径、NPZ、分片/键与来源记录
  frames/000063.npz         # 点云与投影字段
  images/000063.jpg         # 原始 JPEG
  source_semantics/         # 原有语义 PNG，保留其原始类别编码
  odometry.npz              # poses、timestamps、frame_ids；缺失位姿填 NaN
  status.json               # 当前读取进度或错误
  report.json               # 完成情况、帧数、耗时与能力边界
  review/index.html         # 勾选预览时生成
  review/visualization/     # 左图像与投影点、右当前帧 BEV
```

每帧 NPZ 包括 `xyz`、完整 `lidar_points`、标定、`pixels_uv`、`camera_depth` 和 `in_image`。
`camera_depth` 是变换后的相机 Z，单位米；`in_image` 仅表示正深度且投影落在图像范围内，**不表示经过遮挡检验**。
导出的原始索引采用 `lidar_raw_import_v1`，不是已有 `zgc_cached_v1` 语义/实例缓存，不能直接当作已标注训练数据使用。

## 终端命令

```bash
./run.sh lmdb --source /home/qwer/Downloads/zgc/lmdb_20260730T122821Z \
  --output /path/to/lidar_data/jobs/lmdb_readout --frames 63 79 824 --preview

# 只检查索引与执行范围，不导出
./run.sh lmdb --source /path/to/lidar_data/samples/testdata/lmdb_sample \
  --output /path/to/lidar_data/jobs/lmdb_sample --dry-run
```

随包 `/path/to/lidar_data/samples/testdata/lmdb_sample` 含真实采集中的 63、79、824 帧，可离线测试 LMDB 读取。
