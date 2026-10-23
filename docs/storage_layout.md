# 独立存储布局

源码目录只保存自有代码、脚本、配置、第三方补丁及许可证、自有图标、文档与样例发布附件，不包含数据入口软链接。
本机数据根目录为 `/mnt/data/02-Datasets/lidar_annotator/`，由 `~/.config/lidar_annotator/storage.json` 指定。
其他电脑默认使用 `~/.local/share/lidar_annotator/`；`LIDAR_DATA_ROOT` 可覆盖数据根目录。

| 类别 | 位置 |
|---|---|
| 标准数据、生成标签 | `<data_root>/datasets/` |
| 工作目录、模型推理缓存、几何结果、日志、审核图 | `<data_root>/jobs/` |
| 样例数据 | `<data_root>/samples/` |
| 验证记录 | `<data_root>/verification/` |
| 历史实验 | `<data_root>/experiments/` |
| 旧结果归档 | `<data_root>/archives/` |
| 远程依赖源码和字体 | `~/.cache/lidar_annotator/dependencies/<锁定清单摘要>/` |
| 模型权重 | `~/.cache/lidar_annotator/models/` |
| 源码安装的 Python 环境 | `~/.cache/lidar_annotator/venvs/default/` |
| 现有离线 Python 环境 | `~/.cache/lidar_annotator/runtime/<版本>/` |
| 离线环境归档与系统库 | `~/.cache/lidar_annotator/offline-bundles/<版本>/` |
| Python 字节码 | `~/.cache/lidar_annotator/pycache/` |

`models/` 中仅有结构配置和官方来源、SHA256 清单。缓存中的模型与环境由部署脚本准备。
工作目录由数据源、导出目录和帧范围确定，统一位于 `jobs/`，不再扫描旧 `.lidar_annotator_work/` 或迁移旧任务。

## 新任务与历史记录

任务只通过当前配置的标准数据目录与工作目录读取产物。
原有源码数据入口、旧工作目录别名和自动迁移逻辑已移除；旧标签需要在当前路径下重新生成。
有效模型推理缓存可以独立复制并校验后复用，标签、几何结果和审核产物按新任务生成。
历史结果可保存在 `archives/`，不作为当前结果，也不保证旧绝对路径继续可用。

## 发布

源码通过 GitHub 仓库交付；上游依赖按锁定 commit 下载到缓存。样例数据作为独立 Release 附件，仓库不包含样例、结果、模型权重、字体二进制或数据目录软链接。
数据、标签和工作文件应独立备份。部署过程见 [源码部署说明](source_deployment.md)。
