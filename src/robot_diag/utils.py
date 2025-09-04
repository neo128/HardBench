import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    """Run a shell command, return (code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def is_macos() -> bool:
    return platform.system().lower() == "darwin"

def is_windows() -> bool:
    return platform.system().lower() == "windows"


def read_sysfs(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def human_bytes(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"


def sample_thermal() -> Dict[str, Any]:
    """Best-effort thermal sampling (Linux)."""
    if not is_linux():
        return {"supported": False}
    base = "/sys/class/thermal"
    if not os.path.isdir(base):
        return {"supported": False}
    zones: List[Dict[str, Any]] = []
    for name in os.listdir(base):
        if not name.startswith("thermal_zone"):
            continue
        tpath = os.path.join(base, name, "temp")
        typath = os.path.join(base, name, "type")
        t = read_sysfs(tpath)
        ty = read_sysfs(typath)
        if t is None:
            continue
        try:
            temp_c = float(t) / 1000.0 if float(t) > 1000 else float(t)
        except Exception:
            continue
        zones.append({"zone": name, "type": ty, "temp_c": temp_c})
    return {"supported": True, "zones": zones}


def cpu_freqs() -> Dict[str, Any]:
    if not is_linux():
        return {"supported": False}
    result: Dict[str, Any] = {"supported": True, "cpus": []}
    cpu_dir = "/sys/devices/system/cpu"
    if not os.path.isdir(cpu_dir):
        return {"supported": False}
    for name in os.listdir(cpu_dir):
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        cur = read_sysfs(os.path.join(cpu_dir, name, "cpufreq", "scaling_cur_freq"))
        maxf = read_sysfs(os.path.join(cpu_dir, name, "cpufreq", "scaling_max_freq"))
        if cur:
            try:
                cur_mhz = float(cur) / 1000.0
            except Exception:
                cur_mhz = None
        else:
            cur_mhz = None
        result["cpus"].append({"cpu": name, "cur_mhz": cur_mhz, "max_khz": maxf})
    return result


def run_powershell(ps: str, timeout: int = 15) -> Tuple[int, str, str]:
    """Run a PowerShell command string on Windows."""
    exe = "powershell" if shutil.which("powershell") else "pwsh"
    if not exe:
        return 1, "", "no powershell"
    try:
        proc = subprocess.run(
            [exe, "-NoProfile", "-Command", ps],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def linux_distro() -> Optional[Dict[str, Any]]:
    if not is_linux():
        return None
    osr = "/etc/os-release"
    data: Dict[str, Any] = {}
    try:
        with open(osr, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k] = v.strip().strip('"')
        return data
    except Exception:
        return None


def arch() -> str:
    return platform.machine() or "unknown"
