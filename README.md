# LiDAR Annotator · 激光点云标注工具

面向 Linux 的激光点云自动标注与审核工具，支持从 LMDB 或 KITTI 序列生成图像语义、实例 mask、跟踪 ID、点云语义、行人位置和车辆 3D 框，并导出逐帧标签。

**首次使用请依次完成第 1–8 步。** 第 5 步可选择图形界面或命令行，两者使用同一套算法与导出规则。

源码、输入数据、工作文件和模型分别存放；程序不会通过旧目录链接查找标签，也不会自动迁移旧版任务。

## 1. 获取源码并安装环境

从 [GitHub 仓库](https://github.com/xuhan-rgb/lidar_annotator) 获取源码。下文命令均在源码根目录执行：

```bash
git clone https://github.com/xuhan-rgb/lidar_annotator.git
cd lidar_annotator
```

运行条件：

- Linux x86_64、Git、Python 3.12 及其 venv 支持。
- NVIDIA GPU，以及兼容 CUDA 12.8 的驱动；完整推理使用 CUDA bfloat16。
- 已验证的显卡为 RTX 4090 24GB，其他显卡的完整流程显存需求未逐一验证。
- 图形界面需要桌面会话；无桌面服务器可以使用命令行。

```bash
python3.12 --version
nvidia-smi
bash scripts/setup_env.sh
```

安装脚本会：

1. 在 `~/.cache/lidar_annotator/venvs/default/` 创建 Python 环境。
2. 按固定 GitHub commit 下载第三方源码与字体，校验 SHA256，应用本地补丁。
3. 安装 PyTorch 2.10.0 / torchvision 0.25.0 的 CUDA 12.8 wheel 和应用依赖。
4. 执行 `pip check` 及模块导入检查。

不需要单独安装系统 CUDA Toolkit。首次安装和模型下载需要联网，已有完整缓存可以复用。
安装选项、Qt 系统库和故障定位见 [源码部署说明](docs/source_deployment.md)。

## 2. 配置独立的数据目录

把采集数据和结果放在源码目录之外。默认数据根目录为 `~/.local/share/lidar_annotator/`，下面以 `~/lidar_data` 为例：

```bash
export LIDAR_DATA_ROOT="$HOME/lidar_data"
mkdir -p "$LIDAR_DATA_ROOT"
```

环境变量只影响当前终端及从它启动的程序。**使用应用菜单或左侧图标启动时，建议写入持久配置** `~/.config/lidar_annotator/storage.json`：

```json
{
  "data_root": "/absolute/path/to/lidar_data",
  "lmdb_source": "/absolute/path/to/lmdb_recording"
}
```

将路径替换成实际位置；`lmdb_source` 是可选的界面默认输入，也可设置可选的 `kitti_source`。没有配置默认输入时，可以在界面中手动选择。
`LIDAR_DATA_ROOT` 的优先级高于持久配置。后续命令中的 `$LIDAR_DATA_ROOT` 应与持久配置使用同一个目录。
示例文件见 [storage.example.json](configs/storage.example.json)，完整规则见 [存储布局](docs/storage_layout.md)。

```text
外部数据目录/
  datasets/      LMDB 转换后的标准序列与导出标签，也可存放自备 KITTI 序列
  jobs/          每个任务的工作文件、推理缓存、几何结果、日志与审核图
  samples/       单独安装的样例
  verification/  自检及验证记录
  experiments/   实验产物
  archives/      历史结果归档，不参与当前任务读取
```

工作目录和 LMDB 标准数据输出目录不能位于源码目录内。

## 3. 从官方来源下载模型并检查

当前完整流程使用 SAM3、YOLO26x 和 WildDet3D。先在 [Meta SAM3 官方模型页](https://huggingface.co/facebook/sam3)申请访问并接受条款，获批后登录：

```bash
~/.cache/lidar_annotator/venvs/default/bin/hf auth login
./run.sh models download
./run.sh deployment-check --gpu --models
```

检查成功后再开始标注。模型默认保存在 `~/.cache/lidar_annotator/models/`，三个权重合计约 8.3 GB，不放进源码仓库。
使用已有权重或自定义目录的方法见 [部署说明](docs/source_deployment.md)。

上游源码的完整 40 位 commit、下载 SHA256 和补丁校验值记录在 [dependencies.lock.json](configs/dependencies.lock.json)；模型来源和版本记录在 [sources.json](models/sources.json)，权重校验值记录在 [manifest.json](models/manifest.json)。

## 4. 准备待标注数据

选择以下一种格式。完整时序流程需要图像与点云对应关系、有效标定、时间戳和位姿。

### LMDB 采集目录

选择包含 `index.lmdb` 和 `shards` 的采集根目录，不要只选择其中一个数据库。

```text
recording/
  index.lmdb/data.mdb
  shards/
    shard_000000.lmdb/data.mdb
    shard_000001.lmdb/data.mdb
    ...
```

支持本项目的 `indexed_npz_lmdb_v1` 契约，索引与载荷需包含对应的图像、点云、标定、时间戳和位姿信息。LMDB 是存储容器，任意 LMDB 数据库不能直接套用该流程。
原始采集目录只读；程序另行导出标准序列。具体键名、载荷字段见 [LMDB 格式说明](docs/lmdb_input_format.md)。

### KITTI 序列

```text
sequence/
  image_2/000000.png       # 可使用其他 image_* 相机通道
  velodyne/000000.bin      # 小端 float32，每点 x、y、z、intensity 四列
  calib.txt
  times.txt
  poses.txt
```

图像与点云编号需要匹配，标定、时间戳和位姿需要遵循本工具支持的约定。多相机任务一次处理一个通道，使用独立工作目录。
原始图像和点云不改写，生成的标签保存在该序列的 `labels/` 下，因此需要对标签输出位置有写入权限。
标定示例、坐标约定与可选元数据见 [KITTI 格式说明](docs/kitti_input_format.md)。

如果输入是 ROS bag 或其他组织形式，需要先完成格式适配；不能仅通过重命名目录代替同步和标定。

## 5. 执行完整标注

### 方式 A：图形界面

```bash
./run.sh gui
```

按以下顺序操作：

1. 在左侧选择 **LMDB 采集目录** 或 **KITTI 序列**，选择第 4 步准备的输入；KITTI 还需选择相机通道。
2. 填写输出位置。LMDB 的“导出目录”用于存放新的标准序列，例如数据根目录下的 `datasets/sequence_01/`；KITTI 的“处理结果目录”用于存放独立工作文件。
3. 如需调整类别或提示词，点击 **编辑类别与提示词…**。修改会自动保存，任务执行时另存一份 `recipe.json`。类别 ID 保持现有 0–9 契约。
4. 在右侧 **运行配置** 页点击 **完整流程**，确认左侧三个处理阶段以及 **生成标签与可视化** 均已选中。
5. **首次处理新数据，选择“复用缓存继续执行（默认）”。** 没有有效缓存时会正常运行模型，不需要先选择“重新生成全部结果”。
6. “处理帧”留空处理全部序列；`0` 表示第一帧，`0-9` 表示前 10 帧，也支持 `60-80 125 824`。只做子集调试时，不能据此判断全序列质量。
7. 可先点击 **预览执行命令** 检查路径和阶段；确认后点击 **开始执行**，在 **实时日志** 页查看进度。
8. 等待任务完成，再按第 6、7 步审核并使用标签。

“可视化并行数”控制 CPU 逐帧绘图，不改变模型的 GPU 推理并行数。
点击 **停止** 会取消后台任务并保留已完成产物；需要继续时，使用相同数据、输出目录与配置，选择“复用缓存继续执行”。

三种执行方式的区别：

| 场景 | 执行方式 | 行为 |
|---|---|---|
| 首次标注、继续未完成任务、修改类别或提示词 | 复用缓存继续执行 | 运行所选阶段，按有效性复用缓存，更新变化的模型结果与下游产物 |
| 已有语义和实例结果，配置未改变，只想重算几何与输出 | 复用 SAM，重算后处理 | 保留推理结果，强制重做位姿、关联、优化、校验、审核图与标签导出 |
| 明确要丢弃同一任务的旧工作结果并重新运行模型 | 重新生成全部结果 | 清理该任务的旧工作结果与缓存，重新推理 |

**修改类别或提示词后，先用“复用缓存继续执行”更新模型结果。** “重算后处理”要求当前配置与任务快照一致，否则会拒绝启动，防止混用不同配方。
程序会恢复上次选择的执行方式，因此开始新任务前应再次确认。

### 方式 B：命令行

选择与自己输入格式对应的一组命令。下面显式指定工作目录，便于查找；界面默认会自动生成带任务标识的目录，以日志打印的位置为准。

```bash
# LMDB：先预览计划，不写数据、不运行模型。
./run.sh annotate \
  --source /absolute/path/to/lmdb_recording \
  --output "$LIDAR_DATA_ROOT/datasets/lmdb_sequence_01" \
  --work-dir "$LIDAR_DATA_ROOT/jobs/lmdb_sequence_01" \
  --workers 4 --dry-run

# 去掉 --dry-run 执行完整流程。
./run.sh annotate \
  --source /absolute/path/to/lmdb_recording \
  --output "$LIDAR_DATA_ROOT/datasets/lmdb_sequence_01" \
  --work-dir "$LIDAR_DATA_ROOT/jobs/lmdb_sequence_01" \
  --workers 4
```

```bash
# KITTI：输入已有标准序列，输出为独立工作目录。
./run.sh annotate \
  --source /absolute/path/to/kitti_sequence \
  --output "$LIDAR_DATA_ROOT/jobs/kitti_sequence_01_image_2" \
  --camera image_2 --workers 4
```

不传 `--stages` 即执行全部 8 个阶段：

| 顺序 | 阶段 | 主要产物或检查 |
|---|---|---|
| 1 | `semantic` | 图像语义、点云语义、目标视频 mask 与原生跟踪 ID |
| 2 | `instances` | 行人/车辆实例、检测记录及点云支持信息 |
| 3 | `temporal` | 位姿准备、跨帧关联 |
| 4 | `tracking` | 读取已保存的跟踪结果并准备行人观测 |
| 5 | `optimize` | 行人最终位置、静止车辆几何优化与去重 |
| 6 | `validate` | 存储、语义及几何/时序一致性检查 |
| 7 | `render` | 逐帧联合审核图与可用的全帧地图 |
| 8 | `catalog` | 整合逐帧标签、联合索引并导出标准标签 |

新任务默认使用内置配方；已有任务未指定 `--recipe` 时读取任务快照。要使用界面保存的自定义配置，显式添加：

```bash
./run.sh annotate \
  --source /absolute/path/to/lmdb_recording \
  --output "$LIDAR_DATA_ROOT/datasets/lmdb_sequence_01" \
  --work-dir "$LIDAR_DATA_ROOT/jobs/lmdb_sequence_01" \
  --recipe "$HOME/.config/lidar_annotator/recipe.json" --workers 4
```

已有语义与实例结果、配方保持不变，只重算后处理时：

```bash
./run.sh annotate \
  --source /absolute/path/to/lmdb_recording \
  --output "$LIDAR_DATA_ROOT/datasets/lmdb_sequence_01" \
  --work-dir "$LIDAR_DATA_ROOT/jobs/lmdb_sequence_01" \
  --refresh-postprocess --workers 4
```

其他参数见 `./run.sh annotate --help`。仅导入原始数据的 `lmdb` 命令本身不生成完整标注，请使用 `annotate` 执行上述流程。

## 6. 查看与审核结果

界面完成后打开 **结果可视化** 页，按原始帧号检查：

- **查看点云投影**：检查图像、点云同步与投影对齐。
- **查看语义 mask**：检查类别覆盖、边界、遮挡和误分。
- **查看跟踪 ID**：检查目标 ID、跨帧关联，以及优化前后的位置。
- **查看联合标注**：对照图像、点云支持、行人位置和车辆框。
- **查看全帧地图**：检查轨迹连续性、车辆重复框和地图中的明显离群结果。

查看按钮只读取已有产物，不会重新运行模型。可在运行配置中调整显示范围，查看操作不会改变保存的原始点云或训练配置。
注意区分“有当前点支持的位置”和“仅轨迹估计”的位置；轨迹补点、静态地图回投不等于当前帧直接观测。

如果发现类别或提示词问题，调整后按第 5 步重新执行。完整操作及位置策略说明见 [完整 LMDB 流程说明](docs/fresh_lmdb_workflow.md)。
这些是自动标注候选，需经过人工审核后决定是否用于训练，程序不会自动将其转为人工真值或启动训练。

## 7. 找到、导出并保存标签

标准标签的位置与输入格式有关：

| 输入格式 | 图像标签 | 点云标签 |
|---|---|---|
| LMDB | 指定导出目录下的 `labels/image_2/` | 指定导出目录下的 `labels/lidar/` |
| KITTI | 输入序列下的 `labels/image_*/` | 输入序列下的 `labels/lidar/` |

图像标签包含语义 mask、实例/置信度数组及 JSON 记录；点云标签包含逐点 `.label` 和对象、来源等 JSON 记录。
逐点标签为与原点顺序对应的小端 `uint32` 原始语义类别 ID，未知类别为 `255`，不是打包的实例 ID。
详细字段见 [标签格式说明](docs/kitti_input_format.md)。

每个工作目录还保存：

```text
job.json                任务的数据源、帧选择与导出目录
recipe.json             本次实际使用的配方快照
backend.json            各阶段的实际路径与几何约定
run_report.json         总状态、各阶段命令及耗时
logs/                   各阶段日志
annotations.json        图像、点云与审核产物的联合索引
label_export.json       标准标签导出结果、已有标签保护状态
labels/                 按原始帧号保存的工作标签 NPZ
review/index.html       审核图库
review/visualization/   逐帧审核图入口
```

完整任务成功时，`run_report.json` 的总状态和 8 个阶段均应为 `complete`，并应存在对应的标准标签文件。界面 **打开输出目录** 与日志中的目录链接可直接定位产物。

若导出提示“已保护已有标签”，查看 `label_export.json`：人工修改过的标签或其他相机的已有点云标签可能被保留，这些帧不代表已按本次任务重写。需要独立保存本次结果时，可使用下面的另存导出命令。

完整流程已经自动导出标签。如需将已有工作结果再导出到另一个独立标签目录：

```bash
./run.sh export-labels \
  --job "$LIDAR_DATA_ROOT/jobs/lmdb_sequence_01" \
  --output "$LIDAR_DATA_ROOT/exports/lmdb_sequence_01_labels"
```

界面用户应将 `--job` 替换为日志显示的实际工作目录。迁移或归档时，保存标准序列、标签和必要的工作文件；只复制 `annotations.json` 不会复制它引用的数据。

## 8. 失败恢复、更新与日常启动

任务失败时先打开 **实时日志** 查看最后的异常，同时检查工作目录的 `run_report.json` 和对应 `logs/<阶段>.log`。
发生在任务初始化之前的错误可能只有界面实时日志，尚未写入新的任务报告。

| 现象 | 处理方式 |
|---|---|
| SAM3 下载提示 401/403 | 检查官方访问申请与 Hugging Face 登录账号 |
| 远程依赖缺失或校验失败 | 运行 `python3 scripts/dependencies.py check`，按 [远程依赖说明](docs/remote_dependencies.md)重新获取 |
| 重算后处理提示配置已改变 | 选择“复用缓存继续执行”，先按当前配方更新模型结果 |
| 缺少有效时间戳、位姿或标定 | 核对输入格式与坐标约定；完整时序流程不能补造这些信息 |
| Qt xcb / 图形环境错误 | 检查桌面系统库，或使用命令行；详见 [部署说明](docs/source_deployment.md) |

输入、输出和配置不变时，可使用原命令或界面“复用缓存继续执行”继续任务。更换数据源或帧范围时，使用对应的独立任务目录，不要混入其他序列结果。

安装桌面入口：

```bash
./run.sh install-desktop
```

之后可从应用菜单启动，并固定到左侧启动栏。**点击 LiDAR Annotator 图标相当于执行 `run.sh gui`。** 移动源码目录后，重新执行安装命令更新入口。
更新代码后，先关闭已打开的窗口，再从图标或 `./run.sh gui` 重新启动；只点击仍在运行的图标可能仅切回旧窗口。

需要检查环境和代码时使用：

```bash
./run.sh doctor
./run.sh self-test
```

## 可选：先用独立样例跑通部署

样例数据不放在源码仓库，以 [GitHub Releases 附件](https://github.com/xuhan-rgb/lidar_annotator/releases/tag/v0.1.0) 提供。下载后，解压到外部数据目录：

```bash
mkdir -p "$LIDAR_DATA_ROOT/samples"
curl -fL https://github.com/xuhan-rgb/lidar_annotator/releases/download/v0.1.0/lidar-annotator-samples.tar.gz \
  -o /tmp/lidar-annotator-samples.tar.gz
curl -fL https://github.com/xuhan-rgb/lidar_annotator/releases/download/v0.1.0/lidar-annotator-samples.tar.gz.sha256 \
  -o /tmp/lidar-annotator-samples.tar.gz.sha256
(cd /tmp && sha256sum -c lidar-annotator-samples.tar.gz.sha256)
tar -xzf /tmp/lidar-annotator-samples.tar.gz \
  --strip-components=1 -C "$LIDAR_DATA_ROOT/samples"
./run.sh demo --dry-run
./run.sh demo --workers 2
```

`demo` 执行两帧 KITTI 的完整流程，工作结果在 `$LIDAR_DATA_ROOT/jobs/demo/`，标签在该样例序列的 `labels/` 下。样例只用于部署冒烟检查，不代表全量精度评估。
没有样例包也可以直接使用第 4 步准备的自有数据。

维护者通过 `python3 scripts/export_source.py --samples` 生成独立样例附件，并上传到 Release。源码直接通过 GitHub 仓库交付，不需要手工源码压缩包。

## 依赖与仓库内容

```text
源码目录/
  tools/         应用、流程编排与算法
  scripts/       安装、依赖与模型下载、自检、样例打包脚本
  third_party/   本地补丁、许可证与说明；没有完整上游源码
  configs/       可移植存储示例、依赖 commit 锁定清单、运行环境版本
  models/        模型来源、SHA256 清单与结构配置；没有模型权重
  assets/        自有应用图标；字体由部署脚本获取
  docs/          输入格式、部署及处理说明
```

第三方源码、字体、Python 环境和模型保存在 `~/.cache/lidar_annotator/`；启动会检查锁定资源的完整性。
获取和更新固定 commit、补丁的方法见 [远程依赖说明](docs/remote_dependencies.md)。许可证与来源见 [第三方材料说明](THIRD_PARTY_NOTICES.md)。
