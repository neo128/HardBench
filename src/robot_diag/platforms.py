import os
from typing import Any, Dict, Optional

from .utils import arch, is_linux, is_macos, is_windows, linux_distro, read_sysfs


def detect_jetson() -> bool:
    if not is_linux():
        return False
    # Common hints: device-tree model contains NVIDIA Jetson; cpuinfo mentions tegra
    try:
        p = "/proc/device-tree/model"
        if os.path.exists(p):
            with open(p, "rb") as f:
                txt = f.read().decode(errors="ignore").lower()
                if "jetson" in txt or "nvidia" in txt:
                    return True
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo", "r") as f:
            if "tegra" in f.read().lower():
                return True
    except Exception:
        pass
    return False


def detect_intel_nuc() -> bool:
    if not is_linux():
        return False
    # Use DMI product name if available
    name = read_sysfs("/sys/class/dmi/id/product_name")
    if name and "nuc" in name.lower():
        return True
    return False


def system_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": "windows" if is_windows() else ("macos" if is_macos() else ("linux" if is_linux() else "unknown")),
        "arch": arch(),
    }
    if is_linux():
        info["distro"] = linux_distro()
        info["jetson"] = detect_jetson()
        info["intel_nuc"] = detect_intel_nuc()
    return info

