# 从 LMDB 生成完整标注

桌面“LMDB 导出目录”和 `annotate --output` 现在直接表示标准数据根目录：`image_2/`、`velodyne/`、`calib.txt`、`times.txt`、`poses.txt`、`labels/`。任务工作文件统一放在外部数据根目录的 `jobs/`；日志会给出实际工作位置，也可通过 `--work-dir` 指定。原始 LMDB 只读。

在工程根目录运行 `./run.sh gui`，选择 **LMDB 采集目录（index + shards）**。选择采集根目录与一个新的任务输出目录，勾选“读取 LMDB 数据”，点击“完整流程”，保持“输出结果”中的“生成标签与可视化”勾选后开始执行。原始 LMDB 始终只读。“处理帧”留空处理全部；支持 `60-80 125 824`。已有全量任务时，指定子集自动使用 `debug/frames_<摘要>/` 独立目录，不覆盖全量结果。子集调试不代表全序列精度验证。

终端使用同一个后端：

```bash
./run.sh annotate --source /home/qwer/Downloads/zgc/lmdb_20260730T122821Z \
  --output /path/to/lidar_data/datasets/zgc_fresh --workers 6

# 预览计划，不写文件、不加载模型
./run.sh annotate --source /path/to/lidar_data/samples/testdata/lmdb_sample --output /path/to/lidar_data/datasets/sample_job --dry-run
```

界面仅暴露三个处理流程：**语义标注与目标跟踪 → 行人/车辆实例标签 → 行人/车辆时序优化**。另设“输出结果”区域的 **生成标签与可视化**，固定保留视觉和点云索引。时序流程内部的位姿准备、读取 track ID、选点、优化与校验由一次选择执行，子步骤在日志中显示。

第一阶段先生成并缓存行人、车辆的 SAM3 视频 mask/track ID；静态语义使用图像分割。后续实例阶段复用视频 mask，时序阶段读取已保存的 ID，不再重新调用 SAM3 视频模型。静态语义按类别保存合并缓存，连续视频片段保留实例 mask/track 缓存。类别缓存保留精确 float32 类别得分与地面/屋顶规则所需的汇总，不再持久保存每个静态提示词的完整像素数组；最终语义 PNG、置信度、实例 mask、点云标签和可视化保持不变。原生视频得分用于二值 mask 的目标置信度，不冒充逐像素后验概率。

图像语义采用工程内 `recipe.json` 已确认的提示词、门槛、屋顶并集规则和类别映射。新采集没有已确认的上一版语义标签，因此从 SAM3 新预测开始，不将原始 LMDB 中未知编码的语义 PNG 当成旧教师标签。图像编码在同帧多提示词之间复用。FOV 标记 `in_image` 保留；另用逐像素最近深度生成 `visible`，这是投影深度筛选，不等于完整遮挡真值。监督范围 XY±20m、Z[-3,6)m，原始坐标不裁掉。

行人由 YOLO26x conf=0.4、imgsz=1280 确认；SAM3 提供实例 mask 和视频关联。车辆使用真实 WildDet3D 推理，再由 mask 点云及历史观测优化。位姿采用记录的 xyz/yaw，假定点云与重力对齐的机器人坐标一致；没有凭空加入 IMU，也没有消除 odom 累积误差。缺失或非法位姿会明确报错，不伪造地图位姿。

行人、车辆原始视频 track ID 保存在 `data/actor_video/` 和 `data/visual_instances/`；供行人轨迹显示的紧凑 ID 与原始 ID 对照表保存在 `temporal/map_review/sam3_full_video/native_to_display_ids.json`。

每帧一个行人位置；轨迹不足、射线冲突等情况保留原始观测及忽略原因。连续观测之间超过两秒或时间倒退时，SAM3 视频状态重新初始化，ID 在不同片段间不复用。轨迹恢复来源单独保存。静止车辆先使用质量较高的观测作锚点、历史点拟合和旋转 BEV IoU 去重；不要求两面支持。去重后增加车辆几何细化：读取已有 temporal 观测、poses 和 history-fit 基线，对每个候选使用 top 8 多锚拟合，并留出 20% 观测作验证。朝向修正仍限制在原始有向角的 ±90° 以内。通过验证的框写回原有 `temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz`，诊断报告写入 `temporal/map_review/history_fit/multi_anchor/report.json`；验证失败时保留既有 history-fit 框。保留去重后的车辆 ID 和顺序（当前这批数据为 10 个）。此步骤只运行几何脚本，不调用 OpenPCDet 推理；优化阶段缓存有效时会复用，不重复运行。

优化阶段最后统一生成行人最终位置：保留原始 recorded odom，以 `reference_xy_map` 作为地图 XY，Z 沿用已有射线修正位置变换到地图后的高度，再转换回当前帧 LiDAR 坐标。结果保存到 `consistent_tracks/final_centers.npz`，诊断用 `centers_only.npz` 不覆盖。最终位置不再强制保持原中心像素；显示、支持点筛选、审核图及标签导出共用最终文件。当前实例在参考位置 0.5m 范围内少于 5 个有效支持点时，标为 `track_only`，不当作有充分当前点支持的位置；仍只监督 XY，不构造行人三维尺寸。

输入和提示词没有变化时，普通“开始执行”会复用已完成的 SAM 类别缓存、视频 mask 和检测缓存，并更新发生变化的下游步骤。不要勾选“重新生成全部结果”或强制语义/实例刷新来更新最终位置。较旧任务没有最终位置文件时暂显示旧射线结果；完整流程执行后会自动生成新文件。最终位置依赖按内容哈希检查，同内容重写索引不会使其失效；过期结果会提示重新执行，不静默混用。

输出目录：

```text
job.json                         # 数据来源与原始帧选择；防止不同任务混用
raw/                             # 原始图像、完整点云、原始语义 PNG、odom
backend.json                     # 本次任务的后端路径
run_report.json, logs/           # 各阶段状态、错误、日志
 data/class_cache/               # 按类别合并的无损语义缓存及校验 JSON
 data/labels/zgc/                # 语义 mask、置信度、逐点/BEV 标签
 data/visual_instances/          # person/car/van/truck SAM 实例 mask
 data/yolo/, data/raw_wilddet/   # 模型原始检测与来源
 data/person/, data/vehicle/     # 单帧位置、车辆框、支持点索引
 temporal/                      # 位姿、关联、历史拟合、去重、行人优化
 labels/<原始帧号>.npz           # 语义 + 优化行人位置 + 地图车辆框
 annotations.json               # 视觉、点云标签与审核图的联合索引
 review/index.html              # 逐帧审核图库
 review_local/visualization/     # 优化审核图与 JSON 的唯一实体副本
 review/visualization/           # 原始帧号入口，以相对链接复用上述文件
```

后端内部采用连续局部帧号以索引位姿数组；`original_frame_id` 保留原始 LMDB 帧号，最终标签与图库恢复原始帧号。一个输出目录绑定一个数据源与帧集合；修改帧选择会为已有全量任务创建独立调试子目录。重新执行会复用来源、模型及策略哈希一致的推理缓存；失败记录与已完成输出保留。

实例处理的 `data/source/frames` 在语义帧已具备所需字段时，使用相对链接复用 `data/labels/zgc/frames`；缺字段时仍生成独立兼容帧，并通过临时文件原子替换，避免覆盖链接目标。WildDet3D 的兼容 `frames` 入口同样引用已保存的 `raw_predictions`。SAM3 类别缓存和视频实例缓存始终保留。界面的点云投影、语义 mask、跟踪 ID、联合标注和全帧地图继续从原始图像、点云及保存的标签/轨迹按需生成显示，不依赖这些输入的额外物理副本。

这些结果是自动标注候选，尚未经过人工审核，不能称为人工真值。车辆地图回投位置可能出现在相机视野之外；是否用于训练仍需根据逐帧观测与任务要求审核。当前输入契约是 `indexed_npz_lmdb_v1`，不是任意 LMDB 格式。

界面左侧只放数据选择与操作控件，流程说明在右侧“使用说明”页。底部区分流程阶段进度与当前阶段帧数进度；语义与实例推理通过 `progress.json` 更新计数，不在日志中逐帧打印。无逐帧计数的步骤显示处理中动画，警告和错误仍保留。终端直接运行时在同一行刷新进度。

“可视化并行数”只控制 CPU 逐帧绘图，未选择可视化时禁用；GPU 推理不受该值影响。“处理帧（可选）”用于选择原始帧号，留空处理全部；不自动识别问题帧。

当前阶段的可视化目录会显示在右侧标签页上方，可直接打开或复制路径。语义阶段对应 `data/labels/zgc/overlay/`；没有独立图像输出的阶段会说明联合图需等可视化阶段。点击停止属于正常取消，保留已完成结果和取消状态；仅真实错误显示堆栈。

任务开始每个有可视化输出的阶段时，实时日志会打印“可视化目录”链接。直接单击日志中的路径打开对应文件夹；路径包含空格或中文也支持。目录尚未创建时显示提示，重复进度刷新不重复打印链接。终端入口也打印实际目录。


## 可编辑标注配置与增量重算

LMDB 模式下，“标注配置（类别/提示词）”可以选择配置 JSON，“编辑类别与提示词”可修改类别名称、启用/停用以及每类提示词。类别 ID 0–9 和训练映射保持原有契约；不自动添加没有训练映射的新 ID。可恢复当前类别或全部默认提示词，内置 recipe 不覆盖。默认另存到 `configs/user_recipe.json`，每次运行还会在任务目录保存 `recipe.json` 快照。

普通执行自动比较类别缓存键。修改某一类提示词或门槛时，重新推理该类的提示词，其他未变化类别复用合并缓存；随后重新融合类别冲突、投影与导出。缓存使用无损编码，不采用 float16 量化。勾选“重新生成语义输出”会强制刷新，可在下拉框指定全部或单个类别。实例刷新开关同样保留。不改输入、模型、配置或算法时，后续阶段通过输入和产物校验后直接复用；损坏或缺失产物不会冒充完成。

```bash
./run.sh annotate --source /path/to/recording --output /path/to/lidar_data/datasets/new_job \
  --recipe configs/user_recipe.json --frames 60-80 125 824

# 明确强制重算类别 3；其他提示词可继续复用
./run.sh annotate --source /path/to/recording --output /path/to/lidar_data/datasets/new_job \
  --recipe configs/user_recipe.json --frames 60-80 125 824 \
  --stages semantic --refresh-semantic --semantic-classes 3
```

已有旧版整帧缓存若没有逐提示词证据或视频 track，首次切换新流程需要补建类别汇总或视频缓存；不能从旧整帧标签伪造缺失的推理证据。已有有效逐提示词缓存可通过 `python -m tools.lidar_annotator.compact_semantic_cache --job <任务目录> --workers 4 --cleanup` 迁移：全量重放并验证图像标签、置信度、点云、BEV、实例 ID 和保留产物哈希一致后，才删除已核验的逐提示词缓存；失败时不执行清理。

右侧可视化目录下显示 CPU、GPU 利用率及显存已用/总量，每两秒异步刷新。两条进度条位于右侧日志区；不做现场截图验证时，组件检查不能替代用户对最终标签质量的审核。
