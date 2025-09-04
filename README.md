# Robot Hardware Diagnostics

工具集用于评估和排查机器人设备的关键硬件：USB、磁盘、内存、CPU、显卡（GPU）。

目标：
- 快速定位常见硬件故障与性能瓶颈
- 以标准库为主、零外部依赖（可选调用系统命令）
- 统一 CLI 输出与 JSON 报告，便于生产追溯

支持平台：
- Linux（Ubuntu 20.04/22.04 等，x86_64/ARM64，包括 Jetson、Intel NUC）
- Windows 10/11（x64/ARM64，部分能力依赖 PowerShell）
- macOS（Intel/Apple Silicon，能力适配）

包含能力：
- USB：
  - Linux：`lsusb`/sysfs 属性、电源/运行状态、udev/dmesg 事件监控
  - Windows：PowerShell `Get-PnpDevice` 枚举与轮询监控
  - macOS：`system_profiler SPUSBDataType` 枚举
- 磁盘：顺序读写吞吐评估（跨平台），设备/卷信息（Linux: `lsblk/df`；Windows: `Get-Volume`/`Get-PhysicalDisk`）
- 磁盘（增强）：随机 I/O 延迟/IOPS（可配置块大小），文件系统信息（Linux: `/proc/mounts`），Linux 下 USB 存储盘识别（`lsblk -OJ`）
- 内存：简单带宽与稳定性（纯 Python，跨平台）
- CPU：多核压力与吞吐、频率与温度（Linux 可读；其他平台降级）
- GPU：
  - NVIDIA：`nvidia-smi`（Linux/Windows）
  - Linux：`lspci` VGA 信息；macOS：`system_profiler`；Windows：`Win32_VideoController`

使用方法：

```bash
# Linux/macOS（仓库根目录）
PYTHONPATH=src python3 -m robot_diag.cli --help
PYTHONPATH=src python3 -m robot_diag.cli run --all --duration 30 --output reports/latest.json
PYTHONPATH=src python3 -m robot_diag.cli run --disk --disk-path /tmp --size-mb 512
PYTHONPATH=src python3 -m robot_diag.cli run --usb --duration 60

# Windows（PowerShell）
$env:PYTHONPATH="src"; python -m robot_diag.cli --help
$env:PYTHONPATH="src"; python -m robot_diag.cli run --all --duration 30 --output reports\latest.json
$env:PYTHONPATH="src"; python -m robot_diag.cli run --disk --disk-path C:\\Temp --size-mb 512
$env:PYTHONPATH="src"; python -m robot_diag.cli run --usb --duration 60
```

统计报告与评分：
- 运行后 JSON 中包含 `stats` 汇总：
  - `usb`: 设备数量、主机控制器（Linux）、版本与链路速率（Mb/s，Linux）
  - `cpu`: 总 ops/s、目标阈值、评分（0-100）
  - `mem`: 读/写带宽与评分（0-100）
  - `disk`: 顺序读/写与评分（0-100）
  - `gpu`: 平均利用率与评分（0-100，基于 `nvidia-smi` 采样）
  - `aggregate_score`: 加权总分（CPU 0.4、磁盘 0.3、内存 0.2、GPU 0.1）
- `evaluation` 中提供自适应阈值判定（不同平台/型号，Jetson/NUC 等）与失败项列表。
  - 随机 I/O 阈值（最小 IOPS、最大 p95 延迟）将自动评估（若启用随机 I/O 测试）。

USB 摄像头专测（UVC）：
- 需要安装 OpenCV（可选）。
- 命令：
  - Linux/macOS：`PYTHONPATH=src python3 -m robot_diag.cli run --usb --uvc --uvc-index 0 --uvc-duration 10 --uvc-target-fps 30`
  - Windows：`$env:PYTHONPATH="src"; python -m robot_diag.cli run --usb --uvc --uvc-index 0 --uvc-duration 10 --uvc-target-fps 30`
- 若未安装 OpenCV，Linux 上会退化为检测 `/dev/video*` 是否存在。

一键安装脚本：
- Unix：`bash scripts/install_unix.sh --venv .venv`
- Windows（PowerShell）：`powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1 -Venv .venv`

命令示例（带随机 I/O）：
- Linux/macOS：`PYTHONPATH=src python3 -m robot_diag.cli run --disk --duration 5 --size-mb 64 --disk-path /tmp --rand-block-kb 4`
- Windows：`$env:PYTHONPATH="src"; python -m robot_diag.cli run --disk --duration 5 --size-mb 64 --disk-path C:\\Temp --rand-block-kb 4`

注意事项与局限：
- USB 供电电压/电流原始数据通常难以通过通用接口直接读取；本工具通过 sysfs 状态（Linux）、设备枚举与掉线等间接指标辅助判断。
- 磁盘测试默认生成临时测试文件，避免覆盖业务数据；读写后自动清理。
- CPU/GPU 温度、频率等指标依赖平台接口（`/sys/class/thermal`、`nvidia-smi`、`system_profiler`、PowerShell/WMI），不可用时自动降级。
- Jetson 平台可通过 `nvidia-smi`（部分型号不支持）与系统工具补充；后续可加入 `tegrastats` 采样。

报告：
- 统一写入 `reports/` 目录，采用 JSON 格式，包含时间戳与各子测试结果与警示项。

目录结构：

```
.
├── README.md
├── .gitignore
├── pyproject.toml
└── src/
    └── robot_diag/
        ├── cli.py
        ├── utils.py
        ├── usb.py
        ├── disk.py
        ├── cpu.py
        ├── mem.py
        └── gpu.py
```

后续可拓展：
- 更精细的 USB 摄像头连通性测试（UVC 帧抓取）
- 文件系统与随机 I/O/延迟剖析
- 厂测工位一键产测脚本与合格判定阈值
