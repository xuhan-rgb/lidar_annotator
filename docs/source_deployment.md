# 从源码部署完整标注流程

源码版通过 Python 虚拟环境运行，不需要 `runtime/` 离线归档，也不在源码包内分发模型权重。
支持 Linux x86_64、Python 3.12 和 NVIDIA CUDA GPU。当前推理使用 CUDA bfloat16；原开发机器为 RTX 4090 24GB，其他显卡的完整流程耗时和显存尚未逐一验证。
只读预览、命令帮助和部分测试可在 CPU 上运行；完整模型推理不支持 CPU。

## 1. 获取源码与准备系统

从 GitHub 获取本项目源码，进入包含 `run_source.sh` 的根目录。源码仓库不包含第三方源码和字体二进制。
建议使用 Ubuntu 22.04/24.04，并自行安装可运行 CUDA 12.8 应用的 NVIDIA 驱动。
系统需要 Git、Python 3.12 及其 venv 支持；脚本不会修改系统 Python 或自动使用 sudo。

```bash
python3.12 --version
nvidia-smi
bash scripts/setup_env.sh
```

安装脚本创建 `~/.cache/lidar_annotator/venvs/default/`，先获取固定 GitHub commit 的源码与官方字体、应用校验过的补丁，再安装 PyTorch 2.10.0 / torchvision 0.25.0 的 CUDA 12.8 wheel 和应用依赖，最后执行 `pip check` 与推理模块导入检查。完整 commit、归档 SHA256 与补丁记录见 `configs/dependencies.lock.json`。
它不安装 ROS、Conda，也不复制开发者的整个环境。不需要系统 CUDA Toolkit。
需要图形界面时，系统还需具备正常桌面会话及 Qt xcb/OpenGL 系统库；Ubuntu 可安装 `libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1`。无桌面服务器使用命令行。

可指定 Python 和虚拟环境位置：

```bash
bash scripts/setup_env.sh --python /path/to/python3.12 --venv /path/to/venv
export LIDAR_PYTHON=/path/to/venv/bin/python
```

`run_source.sh` 默认使用用户缓存中的 `~/.cache/lidar_annotator/venvs/default/bin/python`；自定义环境时请在后续命令中保留 `LIDAR_PYTHON`。
`--torch-index` 可指定 PyTorch wheel 源，但脚本仍使用上述固定 torch/torchvision 版本，不能据此认为任意 CUDA 组合已验证。

## 2. 从官方来源获取模型

| 本地文件名 | 官方来源 | 获取方式 |
|---|---|---|
| `sam3.pt` | [Meta SAM3](https://huggingface.co/facebook/sam3) | 先申请访问并接受模型条款，再登录 Hugging Face |
| `wilddet3d.pt` | [Ai2 WildDet3D](https://huggingface.co/allenai/WildDet3D) | 脚本下载固定版本的 `wilddet3d_alldata_all_prompt_v1.0.pt` 并使用本地标准名称 |
| `yolo26x.pt` | [Ultralytics 官方发布](https://github.com/ultralytics/assets/releases/tag/v8.4.0) | 脚本从官方 release 下载 |

```bash
# 先在浏览器完成 SAM3 访问申请；登录操作由模型使用者执行。
~/.cache/lidar_annotator/venvs/default/bin/hf auth login

# 查看官方来源、固定版本、大小和 SHA256；不下载模型。
python3 scripts/models.py list

# 默认目录：~/.cache/lidar_annotator/models
bash run_source.sh models download
bash run_source.sh models check
```

也可以通过 `HF_TOKEN` 提供已获得授权的 Hugging Face token；不要写进脚本、配置或源码仓库。
下载使用固定模型版本，并按 `models/manifest.json` 的大小和 SHA256 验证；已有有效权重复用，下载失败不覆盖现有目标。Hugging Face 下载由官方客户端缓存及恢复。
三个权重总计约 8.3 GB（十进制）；Hugging Face 缓存与模型目录会各保存一份文件，安装与下载时需预留额外空间。
每个模型及第三方组件仍遵循其原始许可证，见 `THIRD_PARTY_NOTICES.md` 和官方模型页。

如已有从官网下载的权重，可直接放入同一目录，并使用标准文件名：

```bash
export LIDAR_MODEL_CACHE=/path/to/models
# 此目录中放 sam3.pt、wilddet3d.pt、yolo26x.pt。
bash run_source.sh models download   # 已验证文件不会重新下载；同时准备来源清单
bash run_source.sh models check
```

下载脚本的 `--directory` 只指定本次下载/检查位置；运行标注时请设置相同的 `LIDAR_MODEL_CACHE`。
当前提供的是现有 SAM3 + YOLO26x + WildDet3D 流程的部署，不包含任意模型架构替换。
几何优化不需要下载 OpenPCDet；WildDet3D 使用随源码提供的 LingBot 结构配置及其完整检测权重。

## 3. 检查环境与安装独立样例

源码包没有 `examples/` 或 `testdata/`。下载独立样例数据包后，先安装到外部数据目录：

```bash
export LIDAR_DATA_ROOT=/path/to/lidar_data
mkdir -p "$LIDAR_DATA_ROOT/samples"
tar -xzf lidar-annotator-samples.tar.gz --strip-components=1 -C "$LIDAR_DATA_ROOT/samples"
bash run_source.sh deployment-check --gpu --models
bash run_source.sh demo --dry-run
bash run_source.sh demo --workers 2
```

演示执行完整的两帧 KITTI 流程，结果写入 `$LIDAR_DATA_ROOT/jobs/demo`。样例仅用于部署冒烟检查，不用于精度评估。
也可直接指定自己的数据，安装样例并非运行标注的前提。

## 4. 处理自己的数据

KITTI 输入需要图像、点云、相机标定、时间戳和位姿。具体字段与坐标约定见 [KITTI 格式说明](kitti_input_format.md)。
LMDB 支持本项目的 `indexed_npz_lmdb_v1` 格式，见 [LMDB 格式说明](lmdb_input_format.md)，并非任意 LMDB 或 ROS bag。

```bash
# KITTI 序列，处理全部帧；可以使用 --camera image_2 选择相机。
bash run_source.sh annotate --source /path/to/kitti_sequence \
  --output /path/to/work_directory --workers 4

# LMDB：output 是导出的标准数据目录，工作缓存统一位于数据根目录的 jobs/。
bash run_source.sh annotate --source /path/to/lmdb_recording \
  --output /path/to/exported_sequence --workers 4

# 桌面界面使用同一流程。
bash run_source.sh gui
```

默认顺序：`semantic → instances → temporal → tracking → optimize → validate → render → catalog`。
不传 `--stages` 即执行全部阶段。GPU 模型顺序运行，`--workers` 主要控制 CPU 可视化。
自定义类别提示词使用 `--recipe /path/to/recipe.json`；已有有效缓存会复用。阶段失败保留日志及已完成产物，修复后使用相同命令继续。
输入需要有效标定和位姿，工具不会补造缺失的几何信息。

任务工作目录中的 `run_report.json`、`logs/` 记录执行状态；`review/` 保存审核产物，`annotations.json` 保存联合索引。
标准数据目录的 `labels/image_*/` 与 `labels/lidar/` 保存导出标签。界面及日志会打印具体路径。
详细流程及输出字段见 [完整流程说明](fresh_lmdb_workflow.md)。自动标注结果需要人工审核。

## 5. 发布与固定远程依赖

源码直接提交 GitHub，不需要手工源码压缩包。仓库中的 `third_party/` 只保留补丁、许可证与说明，自有图标保留在 `assets/icons/`。
第三方源码与字体的获取、缓存、完整 commit 及本地补丁见 [远程依赖说明](remote_dependencies.md)。

样例数据单独发布为 GitHub Release 附件：

```bash
python3 scripts/export_source.py --samples
# 上传 dist/lidar-annotator-samples.tar.gz 到 Release，不提交样例目录或权重。
```

源代码版本与样例发布版本可在 Release 说明中关联。数据、工作文件和标签需要独立备份。
项目自身未指定开源许可证，发布者需自行选择合适的项目许可证；第三方材料仍遵循各自许可证。

## 故障定位

- `401/403`：检查 SAM3 访问申请是否获批、是否登录正确账号。
- 权重校验失败：检查是否取得清单对应的官方版本；不要删除校验来混用不同权重。
- 缺少环境：检查 `~/.cache/lidar_annotator/venvs/default/bin/python` 或 `LIDAR_PYTHON`，不要用系统 Python 误启动推理。
- CUDA 检查失败：先检查 `nvidia-smi`，再查看 `deployment-check --gpu` 输出；完整推理要求 CUDA 和 bfloat16。
- Qt xcb 错误：安装桌面系统库或使用命令行；不应在服务器强行启动 GUI。
- 时序优化失败：查看该阶段日志，检查时间戳、标定与位姿；缺失位姿时仍可只读查看原始投影。

远程资源检查失败时，运行 `python3 scripts/dependencies.py check`。不要直接修改缓存中的代码；更新 commit 或补丁应修改锁定清单后重新部署。
