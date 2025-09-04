import os
import time
from typing import Any, Dict, List

from .utils import is_linux, is_macos, is_windows, read_sysfs, run_cmd, run_powershell, which


def list_devices() -> Dict[str, Any]:
    if is_linux():
        if which("lsusb"):
            code, out, err = run_cmd(["lsusb"])
            devices = out.splitlines() if out else []
            return {"supported": True, "tool": "lsusb", "devices": devices, "error": err or None}
        # Fallback: enumerate sysfs
        base = "/sys/bus/usb/devices"
        if os.path.isdir(base):
            items = []
            for d in sorted(os.listdir(base)):
                vid = read_sysfs(os.path.join(base, d, "idVendor"))
                pid = read_sysfs(os.path.join(base, d, "idProduct"))
                prod = read_sysfs(os.path.join(base, d, "product"))
                manu = read_sysfs(os.path.join(base, d, "manufacturer"))
                if vid and pid:
                    items.append(f"{d} {vid}:{pid} {manu or ''} {prod or ''}".strip())
            return {"supported": True, "tool": "sysfs", "devices": items}
        return {"supported": False, "error": "No lsusb/sysfs"}
    if is_macos():
        if which("system_profiler"):
            code, out, err = run_cmd(["system_profiler", "SPUSBDataType"] , timeout=30)
            return {"supported": True, "tool": "system_profiler", "devices": out.splitlines(), "error": err or None}
    if is_windows():
        # Use PowerShell to list USB devices
        code, out, err = run_powershell(
            "Get-PnpDevice -Class USB | Select-Object InstanceId, FriendlyName, Status | ConvertTo-Json -Depth 2",
            timeout=20,
        )
        devices: List[str] = []
        if out:
            try:
                import json
                j = json.loads(out)
                if isinstance(j, dict):
                    j = [j]
                for d in j or []:
                    line = f"{d.get('InstanceId','')} {d.get('FriendlyName','')} {d.get('Status','')}".strip()
                    devices.append(line)
            except Exception:
                devices = out.splitlines()
        return {"supported": True, "tool": "powershell:Get-PnpDevice", "devices": devices, "error": err or None}
    return {"supported": False, "error": "Unsupported platform"}


def power_attrs() -> Dict[str, Any]:
    if not is_linux():
        # Detailed power runtime attributes only available on Linux sysfs in this tool
        return {"supported": False}
    base = "/sys/bus/usb/devices"
    if not os.path.isdir(base):
        return {"supported": False}
    result: List[Dict[str, Any]] = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if not os.path.isdir(p):
            continue
        attrs = {
            "device": d,
            "runtime_status": read_sysfs(os.path.join(p, "power", "runtime_status")),
            "autosuspend": read_sysfs(os.path.join(p, "power", "autosuspend")),
            "active_duration": read_sysfs(os.path.join(p, "power", "active_duration")),
            "connected_duration": read_sysfs(os.path.join(p, "power", "connected_duration")),
            "control": read_sysfs(os.path.join(p, "power", "control")),
            "manufacturer": read_sysfs(os.path.join(p, "manufacturer")),
            "product": read_sysfs(os.path.join(p, "product")),
            "idVendor": read_sysfs(os.path.join(p, "idVendor")),
            "idProduct": read_sysfs(os.path.join(p, "idProduct")),
        }
        result.append(attrs)
    return {"supported": True, "devices": result}


def monitor(duration_sec: int = 30) -> Dict[str, Any]:
    """Monitor USB hotplug/disconnect events for duration.
    Preference: udevadm -> dmesg -> polling lsusb.
    """
    end_t = time.time() + duration_sec
    events: List[str] = []

    # Try udevadm monitor
    if is_linux() and which("udevadm"):
        try:
            import subprocess

            proc = subprocess.Popen(
                ["udevadm", "monitor", "-k", "-s", "usb"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while time.time() < end_t:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    time.sleep(0.1)
                    continue
                if "add" in line or "remove" in line or "change" in line:
                    events.append(line.strip())
            proc.terminate()
            return {"supported": True, "source": "udevadm", "events": events}
        except Exception as e:
            events.append(f"udevadm failed: {e}")

    # Fallback: check dmesg incrementally
    if is_linux() and which("dmesg"):
        start_code, start_log, _ = run_cmd(["dmesg", "--time-format=iso"] , timeout=5)
        prev_len = len(start_log)
        while time.time() < end_t:
            code, out, _ = run_cmd(["dmesg", "--time-format=iso"])  # may require perms
            if code == 0 and out:
                # crude diff by substring length; then scan tail
                tail = out[prev_len:]
                if tail:
                    for line in tail.splitlines()[-100:]:
                        if "USB" in line or "usb" in line:
                            events.append(line.strip())
                prev_len = len(out)
            time.sleep(0.5)
        return {"supported": True, "source": "dmesg", "events": events}

    # Last resort: poll device lists and diff (cross-platform)
    before = list_devices()
    before_set = set(before.get("devices", [])) if before.get("devices") else set()
    while time.time() < end_t:
        cur = list_devices()
        cur_set = set(cur.get("devices", [])) if cur.get("devices") else set()
        added = cur_set - before_set
        removed = before_set - cur_set
        for a in added:
            events.append(f"add: {a}")
        for r in removed:
            events.append(f"remove: {r}")
        before_set = cur_set
        time.sleep(1.0)
    return {"supported": True, "source": "poll", "events": events}
