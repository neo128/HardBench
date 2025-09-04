from typing import Any, Dict, List, Optional

from .utils import is_linux, is_macos, is_windows, read_sysfs
from . import usb as usb_mod
from .thresholds import _get_defaults


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def usb_stats() -> Dict[str, Any]:
    ls = usb_mod.list_devices()
    devices = ls.get("devices", []) if isinstance(ls, dict) else []
    info: Dict[str, Any] = {
        "device_count": len(devices),
        "host_controllers": None,
        "versions": [],
        "link_speeds_Mbps": [],
        "notes": None,
    }
    if is_linux():
        base = "/sys/bus/usb/devices"
        versions: List[str] = []
        speeds: List[float] = []
        host_controllers = 0
        try:
            import os
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if not os.path.isdir(p):
                    continue
                if d.startswith("usb") and d[3:].isdigit():
                    host_controllers += 1
                v = read_sysfs(os.path.join(p, "version"))
                s = read_sysfs(os.path.join(p, "speed"))
                if v:
                    versions.append(v)
                if s:
                    try:
                        speeds.append(float(s))
                    except Exception:
                        pass
        except Exception as e:
            info["notes"] = f"sysfs scan error: {e}"
        info["host_controllers"] = host_controllers
        info["versions"] = sorted(sorted(set(versions)))
        info["link_speeds_Mbps"] = sorted(speeds)
    else:
        info["host_controllers"] = None
        info["versions"] = []
        info["link_speeds_Mbps"] = []
        if is_windows():
            info["notes"] = "Windows: versions/speeds unavailable via default APIs"
        elif is_macos():
            info["notes"] = "macOS: versions/speeds not parsed from system_profiler"
    return info


def cpu_score(results: Dict[str, Any]) -> Dict[str, Any]:
    plat = results.get("platform", {})
    th = _get_defaults(plat)
    c = results.get("cpu", {}).get("stress", {})
    cores = c.get("cores", 1) or 1
    opsps = c.get("ops_per_sec", 0.0) or 0.0
    target = th["cpu_min_ops_per_sec_per_core"] * cores
    score = 0.0
    if target > 0:
        score = _clamp(100.0 * (opsps / (target * 1.5)))  # 150% of threshold ~= 100
    return {"cores": cores, "ops_per_sec": opsps, "target_ops": target, "score": round(score, 1)}


def mem_score(results: Dict[str, Any]) -> Dict[str, Any]:
    plat = results.get("platform", {})
    th = _get_defaults(plat)
    m = results.get("mem", {}).get("bandwidth", {})
    wr = float(m.get("fill_MBps", 0.0) or 0.0)
    rd = float(m.get("read_MBps", 0.0) or 0.0)
    wr_ratio = wr / max(1e-9, th["mem_min_write_MBps"]) if wr else 0.0
    rd_ratio = rd / max(1e-9, th["mem_min_read_MBps"]) if rd else 0.0
    ratio = min(wr_ratio, rd_ratio)
    score = _clamp(100.0 * (ratio / 1.2))  # 120% of threshold ~= 100
    return {"read_MBps": rd, "write_MBps": wr, "score": round(score, 1)}


def disk_score(results: Dict[str, Any]) -> Dict[str, Any]:
    plat = results.get("platform", {})
    th = _get_defaults(plat)
    d = results.get("disk", {}).get("throughput", {})
    wr = float(d.get("write_MBps", 0.0) or 0.0)
    rd = float(d.get("read_MBps", 0.0) or 0.0)
    wr_ratio = wr / max(1e-9, th["disk_min_write_MBps"]) if wr else 0.0
    rd_ratio = rd / max(1e-9, th["disk_min_read_MBps"]) if rd else 0.0
    ratio = min(wr_ratio, rd_ratio)
    score = _clamp(100.0 * (ratio / 1.2))
    return {"read_MBps": rd, "write_MBps": wr, "score": round(score, 1)}


def gpu_score(results: Dict[str, Any]) -> Dict[str, Any]:
    ns = results.get("gpu", {}).get("nvidia_sample", {})
    samples = ns.get("samples", []) if isinstance(ns, dict) else []
    avg_util = 0.0
    if samples:
        try:
            vals = []
            for s in samples:
                u = s.get("util")
                if u is None:
                    continue
                # util might be string without %
                if isinstance(u, str):
                    u = u.strip().strip("%")
                vals.append(float(u))
            if vals:
                avg_util = sum(vals) / len(vals)
        except Exception:
            avg_util = 0.0
    score = _clamp(avg_util)  # 0-100 util -> 0-100 score
    # GPU names for context
    names: List[str] = []
    ov = results.get("gpu", {}).get("overview", {})
    if isinstance(ov.get("nvidia_smi"), str):
        # parse first line name if present
        first = ov["nvidia_smi"].splitlines()[0] if ov["nvidia_smi"] else ""
        if first:
            names.append(first)
    vc = ov.get("video_controller")
    if isinstance(vc, str) and vc:
        names.append(vc)
    return {"avg_util": round(avg_util, 1), "gpu_names": names, "score": round(score, 1)}


def aggregate(results: Dict[str, Any]) -> Dict[str, Any]:
    usb = usb_stats()
    cpu = cpu_score(results)
    mem = mem_score(results)
    disk = disk_score(results)
    gpu = gpu_score(results)

    # Weighted aggregate (tunable)
    agg = 0.4 * cpu["score"] + 0.3 * disk["score"] + 0.2 * mem["score"] + 0.1 * gpu["score"]
    return {
        "usb": usb,
        "cpu": cpu,
        "mem": mem,
        "disk": disk,
        "gpu": gpu,
        "aggregate_score": round(agg, 1),
    }

