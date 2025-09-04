# Contributing

感谢你对本项目的关注！欢迎提交 Issue 与 Pull Request。为提高协作效率，请遵循以下流程：

- 使用模板提交问题与需求（Bug/Feature）。
- 在 PR 中简述变更动机、实现要点与测试方式。
- 遵循本仓库的代码风格（PEP8），保持最小化依赖与清晰结构。
- 对可能影响硬件的改动（如压力参数）请注明风险与验证方法。

开发建议：
- Python 3.8+
- 本地快速检查：`PYTHONPATH=src python -m robot_diag.cli --help`
- 运行一次全量检查：`PYTHONPATH=src python -m robot_diag.cli run --all --duration 10 --size-mb 64`

代码审查要求：
- 通过 CI（安装与基本运行）
- 文档更新（如 README 或模块说明）
