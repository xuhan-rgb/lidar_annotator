# 数据输入与处理说明

软件支持两种输入：原始 LMDB 采集数据，以及已经整理好的 KITTI 风格序列。下面分别说明读取方式、处理顺序和保存位置。

## LMDB 输入：先解析，再导出 KITTI 与执行标注

选择包含 `index.lmdb/` 和 `shards/` 的采集根目录，“LMDB 导出目录”填写转换后标准数据的保存位置。

```text
原始 LMDB 采集目录/
├── index.lmdb/
└── shards/
    ├── shard_000000.lmdb/
    └── ...
```

首次执行顺序：

1. 只读解析 LMDB 索引与所选帧，提取图像、LiDAR 点云、标定、时间戳和已有位姿，写入独立工作目录的内部缓存。
2. 从内部缓存导出 KITTI 风格序列到填写的“LMDB 导出目录”：`image_2/`、`velodyne/`、`calib.txt`（标定变化时为 `calib/`）、`times.txt` 和有有效位姿时的 `poses.txt`；另保存 `metadata.npz`，保留来源和原始附加信息。
3. 标准数据导出成功后，按所选阶段执行 SAM 等模型推理、时序优化、校验和结果生成。后续算法使用内部缓存，不会再把刚导出的 KITTI 文件重新导入一遍。
4. 标签保存到“LMDB 导出目录”下的 `labels/`，工作缓存、日志与审核图库保存在独立工作目录。

因此，LMDB 模式会先完成解析和 KITTI 风格数据导出，再开始标注。导出 KITTI 只是数据格式转换，不等于已经运行 SAM，也不等于已经生成标签。原始 LMDB 不会被修改。图像、点云、标定、时间戳和位姿齐备时，转换后的目录也可以作为 KITTI 输入使用。

重复执行相同任务时，已有内部缓存和标准数据通过校验后复用，不必每次重新解析 LMDB；缺失的标准文件可以补齐。SAM 是否重新推理取决于执行方式和缓存有效性：“复用缓存继续执行”使用有效缓存，“复用 SAM，重算后处理”保留已有模型结果，“重新生成全部结果”清理工作缓存后重跑，包括 SAM。

## KITTI 输入：直接读取已有序列

如果已经有 `image_2/`、`velodyne/`、`calib.txt`、`times.txt`、`poses.txt`，选择 KITTI 输入并指定该序列根目录和相机通道即可，不需要先转换成 LMDB。程序将已有文件导入内部缓存，再执行所选标注阶段。

KITTI 模式的标签默认保存到所选原始序列下的 `labels/`；工作缓存、日志和审核图库仍放在独立工作目录。原始图片、点云和标定保持不变。下面的“最小结构”、标定与位姿约定及两帧示例均说明这种 KITTI 风格序列。

两种输入都不要求事先准备标签、mask、`profile.json` 或 `backend.json`。没有位姿时可以执行图像和单帧标注，完整时序优化需要有效位姿与时间戳。

## 标签自动保存在哪里

运行“生成标签与可视化”后，软件自动整理已有结果，无需单独保存：

```text
一个采集序列/
├── image_2/
├── velodyne/
├── calib.txt
├── times.txt
├── poses.txt
└── labels/
    ├── classes.json             # 类别 ID、名称、RGB 颜色与格式约定
    ├── image_2/
    │   ├── 000000.png           # 图像语义，uint8 类别 ID
    │   ├── 000000.json          # 2D 检测框、实例记录、目标 ID、审核状态
    │   └── 000000.npz           # 实例 mask、图像语义置信度
    └── lidar/
        ├── 000000.label         # 每个原始点的语义类别
        └── 000000.json          # 车辆 3D 框、行人位置、目标 ID、审核状态
```

多相机时在 `labels/` 下增加对应的 `image_3/` 等目录。标签文件名与对应原始文件一致；子集处理也不会从头重新命名。工作目录里的 `label_export.json` 记录本次实际导出位置、受保护帧和备份位置。“打开输出目录”在 KITTI 模式优先打开标签目录；LMDB 模式直接打开填写的“LMDB 导出目录”，该目录就是完整标准序列根目录。LMDB 正在读取或转换时，该按钮显示进度，不会打开内部 `raw/` 缓存目录。

- `.label` 使用小端 uint32，每个点一个数，与同帧 `.bin` 的点顺序和数量严格一致。保存原始 0–9 类别 ID，未知为 255；训练范围外的点写 255，不删除点。文件没有打包实例 ID，也不使用 KITTI/SemanticKITTI 的类别映射。
- 图像 NPZ 中 `semantic_confidence` 保存逐像素语义置信度；`masks_packed` 保存逐实例压缩 mask，使用 `mask_shape` 和 `mask_bitorder='little'` 解码。实例顺序对应同名 JSON 的 `instances[*].mask_index`。
- 2D JSON 中 `detections` 为 YOLO 候选框，`bbox_xyxy` 表示左上、右下像素坐标；`detector_class_id` 保留模型原类别，`class_id` 对应软件类别。`instances` 保留 SAM 实例及已有接受/忽略原因；不会把未确认候选自动视为正样本。
- LiDAR JSON 的 `vehicles[*].box_3d` 顺序为 `[x,y,z,length,width,height,yaw]`，米和弧度，坐标系为同帧 LiDAR。`persons` 保存行人位置和 ID，`supervision='xy'`、`box_3d_valid=false`，不构造行人尺寸或方向。
- 自动结果标记 `review_status='unreviewed'`、`manual_ground_truth=false`。标为已审核、或文件被人工修改后，再次导出会保护整帧对应标签；不会覆盖。自动结果变化时，上一版保存在工作目录 `label_versions/`。这里只整理已有算法结果，没有新增人工审核界面。
- 不做多相机融合：同帧 `labels/lidar/` 已由另一相机生成时，保留已有点云标签，并在导出报告标记 `protected_other_camera`；各相机的图像标签分别正常保存。
- 未运行相应标注阶段时，不会生成伪造的空标签；每帧 JSON 的 `available` 标明实际已有的标签类型。

桌面软件读取 LMDB 时，填写的“LMDB 导出目录”直接保存标准数据。例如填写 `/path/to/lidar_data/datasets/lmdb_import` 后：

```text
/path/to/lidar_data/datasets/lmdb_import/
├── image_2/
├── velodyne/
├── calib.txt
├── times.txt
├── poses.txt
├── metadata.npz
└── labels/
```

`raw/`、`review/`、`job.json`、日志和模型中间结果统一保存到外部数据根目录的 `jobs/`，不会混入标准数据。旧工作目录不自动迁移，也不通过旧路径别名查找；请使用新目录重新生成标签。

指定处理帧时，LMDB 标准数据仍保存到填写的导出目录，缓存按帧范围隔离；已有标准数据属于其他帧范围或来源时会拒绝覆盖，需要选择新的导出目录。同一来源的标准文件如果缺失，软件会校验现有文件并自动补齐，已有 `labels/` 保留；检测到原始导出文件被修改时不会覆盖。“重新生成全部结果”清理工作缓存，不删除已发布的标准数据和人工标签。原始 LMDB 始终只读。

已有任务也可以只整理标签，不运行模型：

```bash
python lidar_annotator.py export-labels --job /path/to/lidar_data/jobs/kitti_import/image_2
```

数据目录只读时，可以显式指定外部标签目录：

```bash
python lidar_annotator.py export-labels --job /path/to/lidar_data/jobs/kitti_import/image_2 \
  --output /path/to/writable/labels
```

## 最小结构

```text
一个采集序列/
├── image_2/              # 一个相机通道，六位数字帧号
│   ├── 000000.png
│   └── 000001.png
├── velodyne/             # 同帧 LiDAR 点云
│   ├── 000000.bin
│   └── 000001.bin
├── calib.txt             # 内参和 LiDAR → 相机外参
├── times.txt             # 一行一个时间戳，单位秒
└── poses.txt             # 一行一个 3×4 位姿，按行展开
```

点云 `.bin` 是小端 float32，每点 4 个数：`x y z intensity`，坐标单位米。图片与点云按同名数字帧号对应；时间戳和位姿的第 n 行对应帧号 n（从 0 开始）。可以只有一个相机目录，不必创建空相机目录。没有位姿时仍可查看投影、执行图像和单帧标注；时序优化需要有效位姿和时间戳。

这里采用 KITTI odometry 的序列组织方式。LMDB 转换示例只有实际存在的一路相机，是 KITTI 风格的单相机子集，不伪造左右双目相机或人工真值标签。

## 项目内可直接读取的两帧最小示例

目录：`/path/to/lidar_data/samples/examples/kitti_minimal_two_frames/`。该目录严格按上面的最小结构提供两张真实图片、两个同帧点云及 `calib.txt`、`times.txt`、`poses.txt`，不依赖 `metadata.npz`、标签或其他配置文件。数据取自项目已有的 `/path/to/lidar_data/samples/examples/kitti_zgc_first10/` 第 0、1 帧；图片、点云和标定原样复制，时间戳和位姿取前两行。这是格式与导入示例，不是人工真值标注集，也不代表两帧足以验证完整时序优化效果。

在软件中选择 KITTI 输入，选择 `/path/to/lidar_data/samples/examples/kitti_minimal_two_frames/` 为序列根目录，相机选 `image_2`，处理帧留空即可读取两帧。对应标注示例命令（会运行所选标注阶段）：

```bash
python lidar_annotator.py annotate \
  --source /path/to/lidar_data/samples/examples/kitti_minimal_two_frames \
  --output /path/to/lidar_data/jobs/kitti_minimal_example \
  --input-format kitti --camera image_2
```

只检查任务计划、不执行模型推理时，在上述命令末尾添加 `--dry-run`。`calib.txt` 的数值仅适用于附带数据；准备自己的数据时应换成自己的标定、时间戳和位姿。

## 标定和位姿约定

单相机示例：

```text
P2: fx 0 cx 0 0 fy cy 0 0 0 1 0
Tr: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```

`P2` 为 3×4 投影矩阵，作用于参考整流相机坐标；`Tr` 为 3×4 的 LiDAR 到参考相机变换。支持 `Tr_velo_to_cam` 别名及可选 `R0_rect` 整流矩阵。未整流的单相机数据应提供与实际图片匹配的针孔内参和外参，不能直接把有畸变的原图当作已整流图像。

### 怎样理解 LiDAR → 相机外参

外参描述的是：同一个空间点，怎样从“以雷达为参考”的坐标，换算成“以相机为参考”的坐标。雷达与相机安装在不同位置，朝向也可能不同，因此同一个点在两套坐标系里的数值通常不同。

例如，雷达测到一个“前方 10 米”的点，但相机安装在雷达前方、上方，朝向还有一些偏转，那么这个点在相机坐标里不一定恰好也是“正前方 10 米”。外参负责换算坐标，空间中的点本身没有移动。

外参由旋转 `R` 和平移 `t` 组成：

| 符号 | 含义 | 格式与单位 |
|---|---|---|
| `p_lidar` | 一个点在 LiDAR 坐标系中的 `[x, y, z]` | 3×1 列向量，单位米 |
| `R` | 把 LiDAR 坐标轴方向转换到相机坐标轴方向 | 3×3 旋转矩阵，无单位 |
| `t` | LiDAR 原点在目标相机坐标系中的位置 | 3×1 列向量，单位米 |
| `p_camera` | 同一个点在目标相机坐标系中的 `[x, y, z]` | 3×1 列向量，单位米 |

换算公式为：

```text
p_camera = R × p_lidar + t
```

`calib.txt` 中的 `Tr` 把 `R` 和 `t` 合并为一个 3×4 矩阵，按行展开保存为 12 个数：

```text
Tr: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz

展开后：
       旋转 R       平移 t
┌                        ┐
│ r00  r01  r02      tx   │
│ r10  r11  r12      ty   │
│ r20  r21  r22      tz   │
└                        ┘
```

前三列负责旋转，最后一列负责平移。这里的 `t` 不是“相机在雷达坐标系中的安装位置”。如果标定工具给的是反方向“相机 → LiDAR”，必须先对整个刚体变换求逆，不能只把平移改成负数。

### 外参和内参有什么区别

外参解决两个传感器之间的位置与朝向关系，内参解决相机三维坐标怎样投影到图片上：

```text
LiDAR 中的三维点
    ↓ 外参：旋转、平移
相机坐标中的三维点
    ↓ 内参：焦距、图像中心等，再做透视除法
图片上的像素位置
```

在上面的单相机 `P2` 示例中，`fx`、`fy` 是以像素为单位的焦距，`cx`、`cy` 是图像中心参数；`Tr` 保存外参。若提供 `R0_rect`，程序还会应用相机整流旋转；多相机情况下还需使用对应相机的投影和外参，不能将不同通道的标定混用。

示例里的外参数值只适用于附带数据。准备自己的数据时，需要使用实际雷达与相机的标定结果；安装位置或朝向改变后，应重新核查外参。

### poses.txt 与传感器外参的区别

`poses.txt` 使用 KITTI odometry 约定：当前参考相机坐标到首帧参考相机坐标的 3×4 刚体变换。软件通过标定把它转换成 LiDAR 参考系位姿，保留完整旋转和平移，不只保留 yaw。现有 LMDB 的位姿是 `[x,y,z,yaw]`，导出时先转成刚体矩阵，再转换到相机参考系。原始位姿另保存在转换元数据中。

`Tr` 描述雷达和相机之间的安装关系；`poses.txt` 描述采集设备逐帧运动后的位姿。两者表示不同的坐标变换，不能互相代替。

相机和 LiDAR 不同时刻采集时，序列按已配对的帧输入。转换不会自动做运动补偿或重新同步。

## 两帧数据的具体例子

下面是一个说明文件格式的假设示例：图像大小 1280×720，LiDAR 坐标为 X 向前、Y 向左、Z 向上，相机坐标为 X 向右、Y 向下、Z 向前，两个传感器原点重合。实际采集数据必须填写真实标定值，不能直接套用这些数值。

`calib.txt`：

```text
P2: 800 0 640 0 0 800 360 0 0 0 1 0
Tr: 0 -1 0 0 0 0 -1 0 1 0 0 0
```

`times.txt`：第一帧为 0 秒，第二帧为 0.1 秒。

```text
0.000000000
0.100000000
```

`poses.txt`：第一帧为单位变换，第二帧沿首帧相机 Z 轴（前方）移动 0.1 米，方向不变。

```text
1 0 0 0 0 1 0 0 0 0 1 0
1 0 0 0 0 1 0 0 0 0 1 0.1
```

同时准备 `image_2/000000.png`、`image_2/000001.png`，以及同名的两份 `velodyne/*.bin`。点云文件是二进制，不能直接把文本改名成 `.bin`。例如，下面代码把两个点写成一帧 KITTI 点云：

```python
import numpy as np

# 每行依次是 x、y、z（米）和 intensity
points = np.array([[5.0, 0.0, 0.0, 100.0],
                   [6.0, 1.0, 0.5, 80.0]], dtype='<f4')
points.tofile('velodyne/000000.bin')
```

该代码要求 `velodyne/` 已存在；实际数据应写入每一帧自己的全部原始点。

## 多相机

```text
一个采集序列/
├── image_2/              # 相机 2：000000.png、000001.png
├── image_3/              # 相机 3：000000.png、000001.png
├── velodyne/             # 共享：000000.bin、000001.bin
├── calib.txt             # P2、P3 与各自外参
├── times.txt
└── poses.txt
```

同一序列可以增加 `image_0/`、`image_1/`、`image_3/` 等目录。共享同一组 `velodyne/`、`times.txt` 和 `poses.txt`，`calib.txt` 中给各通道提供对应 `P0`、`P1`、`P2`、`P3`。对于朝向不同的独立相机，可额外写 `Tr_velo_to_cam_2`、`Tr_velo_to_cam_3`（兼容 `Tr_2`、`Tr_3`），分别表示 LiDAR 到该通道相机的外参，对应 `Pn` 使用 `[Kn|0]`。独立外参优先于共同 `Tr`；`poses.txt` 的参考仍为共同参考相机或显式通道 0 外参。没有共同参考外参时须提供 `Tr_velo_to_cam_0`，避免把选择的相机当成不同位姿参考。

软件在“相机通道”下拉框中选择本次使用的通道；默认优先 `image_2`。

一次运行处理一个选定通道，结果保存在输出目录下的相机子目录。不同通道的任务和缓存隔离；目前不执行多相机联合融合。当前 LMDB 转换仅输出其现有单相机，不增加其他相机。

## 前 10 帧示例

实际示例目录为 `/path/to/lidar_data/samples/examples/kitti_zgc_first10/`。重新转换时使用新的空输出目录：

```bash
python lidar_annotator.py export-kitti \
  --source /home/qwer/Downloads/zgc/lmdb_20260730T122821Z \
  --output /path/to/lidar_data/samples/examples/kitti_zgc_first10 \
  --count 10
```

只读取索引中最前面的 10 帧载荷。生成原图、完整原始点云、标定、时间戳和位姿，不运行模型，不生成标签。

转换目录还有一个可选 `metadata.npz` 扩展文件，保留 KITTI 标准字段容纳不下的原始信息：精确的 LiDAR/图像纳秒时间戳、原始帧号、点云额外列、原始 JPEG、原始位姿和来源记录。普通 KITTI 数据没有这个文件也能读取。PNG、BIN、标定和位姿文件是实际输入；元数据不会用旧图像或点云覆盖用户修改后的输入。

LMDB 原始读取命令保持可用，并自动保存 KITTI 副本：

```bash
python lidar_annotator.py lmdb --source /path/to/lmdb --output /path/to/lidar_data/jobs/raw --frames 0 1 2
# KITTI 副本：/path/to/lidar_data/jobs/raw/kitti/
```

完整标注流程 `annotate` 的 `--output` 直接指定标准序列目录，工作目录自动分开；也可以用 `--work-dir` 显式指定工作位置。独立低级 `lmdb` 导入命令仍保留旧的原始缓存导入用途。转换不修改原始 LMDB。

## 软件和命令行

在左侧“数据格式”选择“KITTI 序列（图像 / 点云 / 标定）”，选择序列目录和相机通道。继续使用“编辑类别与提示词…”配置标注，然后选择处理流程。

```bash
# 只验证输入并生成原始投影、联合索引，无模型推理
python lidar_annotator.py annotate --input-format kitti \
  --source /path/to/lidar_data/samples/examples/kitti_zgc_first10 --camera image_2 \
  --output /path/to/lidar_data/jobs/kitti_example/image_2 --stages render catalog

# 查看完整处理计划，不写数据、不运行模型
python lidar_annotator.py annotate --input-format kitti \
  --source /path/to/lidar_data/samples/examples/kitti_zgc_first10 --camera image_2 \
  --output /path/to/lidar_data/jobs/kitti_example_full/image_2 --dry-run
```

命令行不强制指定 `--input-format`，可根据目录自动识别；指定后会校验格式。桌面软件仅提供 LMDB 与 KITTI 两种数据入口。
