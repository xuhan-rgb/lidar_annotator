# 第三方依赖说明

本目录只包含项目维护的补丁和许可证，不包含上游源码副本。

- `patches/`：相对于固定 GitHub commit 的本地修改。
- `licenses/`：相关项目和字体的许可证、版权说明。
- `../configs/dependencies.lock.json`：完整仓库地址、commit、SHA256 与补丁校验信息。

使用 `python3 scripts/dependencies.py fetch` 获取依赖到用户缓存。
详细版本、缓存位置及更新流程见 [远程依赖文档](../docs/remote_dependencies.md)。
