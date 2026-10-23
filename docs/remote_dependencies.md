# 远程依赖与固定版本

第三方源码与字体由 `scripts/dependencies.py` 从官方来源下载，保存到用户缓存。
源码仓库只保留锁定清单、本地补丁、许可证与自有图标，不保留上游完整源码或字体二进制。

## GitHub 源码 commit

完整 URL、40 位 commit、下载归档 SHA256、提取范围、补丁及修改前后文件 SHA256，均记录在 `configs/dependencies.lock.json`。

| 名称 | GitHub 仓库 | 固定 commit |
|---|---|---|
| sam3 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | `f6e51f59500a87c576c2df2323ce56b9fd7a12de` |
| wilddet3d | [allenai/WildDet3D](https://github.com/allenai/WildDet3D) | `1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a` |
| wild_sam | [weikaih04/sam3_for_wilddet3d](https://github.com/weikaih04/sam3_for_wilddet3d) | `159490bb13108749dd5ceff68f42cd1792e60641` |
| lingbot | [Robbyant/lingbot-depth](https://github.com/Robbyant/lingbot-depth) | `ff29de5bf433d1b1519cd851da00bc652668964a` |
| utils3d | [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d) | `83bfa5bda008803f6521ce9547f94023b8afeddd` |
| vis4d | [SysCV/vis4d](https://github.com/SysCV/vis4d) | `9cd052ed72a208773abc1f6f0cd9109d8ec97531` |
| msgpack | [msgpack/msgpack-python](https://github.com/msgpack/msgpack-python) | `6129789e9f6ebccdb19b23c2cd1dc9551e57fc53` |

WildDet3D 的 SAM3 分支和 LingBot-Depth 使用其上游子模块锁定的 commit。
msgpack 保留现有 1.0.3 的纯 Python 实现，以维持原有 LMDB 解码行为。
Vis4D 按 v1.0.0 对应的 commit 获取源码，仅使用本项目需要的算子；不安装其完整训练依赖。
termcolor 2.5.0、ml-collections 1.1.0 和 absl-py 2.5.1 从 PyPI 安装，版本在 `requirements/inference.txt` 中固定。

## 本地补丁

- `third_party/patches/sam3.patch`：保留现有视频/掩码提示、日志和按需导入 Decord 的修改。
- `third_party/patches/wilddet3d.patch`：保留本地 LingBot 结构配置加载以及 checkpoint 严格校验。

安装顺序是：下载固定 commit → 校验归档 SHA256 → 提取运行包 → 校验补丁输入文件 → `git apply` → 校验补丁输出文件。
没有使用 main/master 或 latest 作为下载版本。

## 字体和模型资源

Noto Sans CJK SC 2.004 从 `notofonts/noto-cjk` 的 `f8d157532fbfaeda587e826d4cd5b21a49186f7c` 获取。
DejaVu Sans 2.37 从官方 `version_2_37` release 获取；其标签对应 commit 为 `0eda8a319c08835009849583cd090bb5b141ce25`，发布附件另以 SHA256 锁定。
字体文件和许可证保存到依赖缓存的 `fonts/`，渲染代码不会回退到项目内副本。
DejaVu 使用官方发布的文件；与原系统字体的二进制文件不完全相同，未做界面截图对比。
自有 `assets/icons/lidar-annotator.svg` 继续保留在仓库。

模型权重仍由 `scripts/models.py` 从官方来源下载，HF revision 与 YOLO 发布标签、关联 commit 记录在 `models/sources.json`；权重内容由 `models/manifest.json` 的 SHA256 校验。
YOLO 的 GitHub commit 是资源仓库发布标签的来源记录，推理 Python 包版本单独固定在 requirements 中。

## 安装与检查

系统需要 Git 与 Python 3.12。部署脚本会安装 Python 依赖并获取锁定的远程资源：

```bash
bash scripts/setup_env.sh
bash run_source.sh deployment-check --gpu --models
```

也可以单独使用纯标准库的资源工具：

```bash
python3 scripts/dependencies.py list
python3 scripts/dependencies.py fetch
python3 scripts/dependencies.py check
python3 scripts/dependencies.py path
```

缓存默认位于 `~/.cache/lidar_annotator/dependencies/<锁定清单摘要>/`。下载文件按 SHA256 保存在相邻 `downloads/`，不会进入源码目录。
启动会检查锁定版本、项目补丁及缓存文件内容，包含非预期新增文件的缓存也会报错。已有正确缓存可离线复用。
`ready.json` 记录当前源码 commit 和已安装文件的校验值；`./run.sh doctor` 也会输出源码 commit。

## 更新版本

不要直接修改缓存中的源码。需要更新时，明确选择新的 commit，在锁定清单中更新 URL 和归档 SHA256，并重新生成、校验所需补丁及其修改前后文件 SHA256。
修改清单会产生新的缓存目录，旧版本不会被静默覆盖。随后运行资源获取、自检和独立样例全流程，确认后再提交到 GitHub。
如果只是缓存损坏，先检查 `dependencies.py check`，确认无须保留手工改动后移除该版本缓存，再重新获取。
