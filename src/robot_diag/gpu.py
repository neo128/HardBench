from typing import Any, Dict, List

from .utils import is_linux, is_macos, is_windows, run_cmd, run_powershell, which


def gpu_overview() -> Dict[str, Any]:
    if is_linux():
        info: Dict[str, Any] = {"platform": "linux"}
        if which("nvidia-smi"):
            code, out, err = run_cmd([
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,clocks.gr,clocks.sm,clocks.mem,power.draw",
                "--format=csv,noheader,nounits",
            ], timeout=5)
            info["nvidia_smi"] = out or err
        # lspci for generic VGA adapters
        if which("lspci"):
            code, out, err = run_cmd(["lspci", "-nnk"])  # may be large
            if out:
                lines = [l for l in out.splitlines() if "VGA" in l or "3D controller" in l]
                info["lspci_vga"] = lines
        return info
    if is_macos():
        if which("system_profiler"):
            code, out, err = run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=10)
            return {"platform": "macos", "system_profiler": out or err}
    if is_windows():
        info: Dict[str, Any] = {"platform": "windows"}
        # NVIDIA
        if which("nvidia-smi"):
            code, out, err = run_cmd([
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,clocks.gr,clocks.mem,power.draw",
                "--format=csv,noheader,nounits",
            ], timeout=5)
            info["nvidia_smi"] = out or err
        # Generic controller info
        code, out, err = run_powershell(
            "Get-WmiObject Win32_VideoController | Select-Object Name, AdapterCompatibility, DriverVersion | ConvertTo-Json",
            timeout=20,
        )
        info["video_controller"] = out or err
        return info
    return {"platform": "unknown"}


def nvidia_sample(duration_sec: int = 10, interval: float = 1.0) -> Dict[str, Any]:
    if not which("nvidia-smi"):
        return {"supported": False}
    samples: List[Dict[str, Any]] = []
    import time
    end_t = time.time() + duration_sec
    while time.time() < end_t:
        code, out, err = run_cmd([
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,clocks.gr,clocks.mem,power.draw",
            "--format=csv,noheader,nounits",
        ])
        if out:
            for line in out.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if len(fields) >= 8:
                    samples.append({
                        "timestamp": fields[0],
                        "index": fields[1],
                        "name": fields[2],
                        "tempC": fields[3],
                        "util": fields[4],
                        "clock_gr": fields[5],
                        "clock_mem": fields[6],
                        "powerW": fields[7],
                    })
        time.sleep(interval)
    return {"supported": True, "samples": samples}


def tegrastats_sample(duration_sec: int = 10) -> Dict[str, Any]:
    if not is_linux() or not which("tegrastats"):
        return {"supported": False}
    import subprocess, time
    samples: List[str] = []
    try:
        proc = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        end_t = time.time() + duration_sec
        while time.time() < end_t:
            if proc.stdout is None:
                break
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            samples.append(line.strip())
        proc.terminate()
    except Exception as e:
        return {"supported": True, "error": str(e), "samples": samples}
    return {"supported": True, "samples": samples}
