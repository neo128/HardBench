import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict

from . import usb, disk, mem, cpu, gpu
from .platforms import system_summary
from . import stats as stats_mod
from .thresholds import evaluate as eval_thresholds
from .utils import ensure_dir, now_ts, write_json


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "timestamp": now_ts(),
        "platform": system_summary(),
        "params": {
            "duration": args.duration,
            "size_mb": args.size_mb,
            "disk_path": args.disk_path,
        },
        "usb": {},
        "disk": {},
        "mem": {},
        "cpu": {},
        "gpu": {},
        "warnings": [],
    }

    if args.usb or args.all:
        results["usb"]["list"] = usb.list_devices()
        results["usb"]["power_attrs"] = usb.power_attrs()
        if args.duration > 0:
            results["usb"]["monitor"] = usb.monitor(duration_sec=args.duration)
        # Optional UVC camera test
        if getattr(args, "uvc", False) or args.all:
            try:
                from .usb_camera import uvc_test
                target_fps = getattr(args, "uvc_target_fps", 0) or None
                cam_idx = getattr(args, "uvc_index", 0)
                cam_dur = getattr(args, "uvc_duration", 10)
                results["usb"]["uvc_test"] = uvc_test(index=cam_idx, duration_sec=cam_dur, target_fps=target_fps)
            except Exception as e:
                results["usb"]["uvc_test"] = {"supported": False, "passed": False, "reason": str(e)}

    if args.disk or args.all:
        results["disk"]["overview"] = disk.disk_overview(args.disk_path)
        results["disk"]["fs_info"] = disk.fs_info(args.disk_path)
        results["disk"]["throughput"] = disk.throughput_test(args.disk_path, size_mb=args.size_mb)
        # Optional random I/O
        do_rand = getattr(args, "rand_io", True)
        if do_rand:
            results["disk"]["random_io"] = disk.random_io_test(
                args.disk_path, file_size_mb=min(64, max(8, args.size_mb)), duration_sec=max(2, min(10, args.duration)), block_kb=getattr(args, "rand_block_kb", 4)
            )
        # Linux USB storage overview
        results["disk"]["usb_storage"] = disk.usb_storage_overview()

    if args.mem or args.all:
        # memory bandwidth with optional mode override
        results["mem"]["bandwidth"] = mem.mem_bandwidth(size_mb=min(args.size_mb, 1024), passes=2, mode=getattr(args, "mem_mode", "auto"))

    if args.cpu or args.all:
        results["cpu"]["stress"] = cpu.stress(duration_sec=max(5, args.duration))

    if args.gpu or args.all:
        results["gpu"]["overview"] = gpu.gpu_overview()
        ns = gpu.nvidia_sample(duration_sec=max(0, min(args.duration, 30)))
        results["gpu"]["nvidia_sample"] = ns
        ts = gpu.tegrastats_sample(duration_sec=max(0, min(args.duration, 30)))
        results["gpu"]["tegrastats"] = ts

    # Basic warnings heuristics
    d = results.get("disk", {}).get("throughput", {})
    if d.get("write_MBps") and d.get("write_MBps") < 50:
        results["warnings"].append(f"Low disk write throughput: {d.get('write_MBps'):.1f} MB/s")
    if d.get("read_MBps") and d.get("read_MBps") < 50:
        results["warnings"].append(f"Low disk read throughput: {d.get('read_MBps'):.1f} MB/s")

    # Stats and evaluation
    results["stats"] = stats_mod.aggregate(results)
    results["evaluation"] = eval_thresholds(results)

    if args.output:
        ensure_dir(os.path.dirname(args.output) or ".")
        write_json(args.output, results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Robot Hardware Diagnostics CLI")
    sub = parser.add_subparsers(dest="cmd")

    runp = sub.add_parser("run", help="Run diagnostics")
    runp.add_argument("--all", action="store_true", help="Run all checks")
    runp.add_argument("--usb", action="store_true", help="Run USB checks")
    runp.add_argument("--disk", action="store_true", help="Run disk checks")
    runp.add_argument("--mem", action="store_true", help="Run memory checks")
    runp.add_argument("--cpu", action="store_true", help="Run CPU checks")
    runp.add_argument("--gpu", action="store_true", help="Run GPU checks")
    runp.add_argument("--duration", type=int, default=20, help="Duration for time-based tests (s)")
    runp.add_argument("--size-mb", type=int, default=256, help="Data size for disk/mem tests (MB)")
    runp.add_argument("--disk-path", type=str, default="/tmp", help="Directory for disk test file")
    runp.add_argument("--output", type=str, default="reports/latest.json", help="Path to write JSON report")
    # Random I/O options
    runp.add_argument("--no-rand-io", dest="rand_io", action="store_false", help="Disable random I/O test")
    runp.add_argument("--rand-block-kb", type=int, default=4, help="Random I/O block size (KB)")
    # Memory options
    runp.add_argument("--mem-mode", choices=["auto", "python", "numpy"], default="auto", help="Memory read mode: auto, python, or numpy")
    # UVC camera options
    runp.add_argument("--uvc", action="store_true", help="Run UVC camera test (OpenCV if available)")
    runp.add_argument("--uvc-index", type=int, default=0, help="Camera index for UVC test")
    runp.add_argument("--uvc-duration", type=int, default=10, help="Duration for UVC test (s)")
    runp.add_argument("--uvc-target-fps", type=int, default=0, help="Expected camera FPS (0 to ignore)")

    # report subcommand
    reportp = sub.add_parser("report", help="Generate Markdown from report JSON(s)")
    reportp.add_argument("--input", nargs="+", required=True, help="Input JSON report files")
    reportp.add_argument("--output", required=True, help="Output Markdown file path")

    args = parser.parse_args()
    if args.cmd == "run":
        results = run_all(args)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif args.cmd == "report":
        from .report_md import generate_md
        path = generate_md(args.input, args.output)
        print(f"Wrote markdown: {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
