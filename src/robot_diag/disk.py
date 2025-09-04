import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .utils import human_bytes, is_windows, is_linux, run_cmd, run_powershell, which


def _device_info(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": path}
    if is_windows():
        # Derive drive letter from path
        drive = (path[:1] if ":" not in path else path[:1]).upper()
        if not drive.isalpha():
            drive = "C"
        # Volume info
        code, out, err = run_powershell(
            f"Get-Volume -DriveLetter {drive} | Select-Object DriveLetter, FileSystem, Size, SizeRemaining | ConvertTo-Json"
        )
        info["volume"] = out or err
        # Physical disk summary
        code, out, err = run_powershell(
            "Get-PhysicalDisk | Select-Object FriendlyName, MediaType, Size, SerialNumber | ConvertTo-Json"
        )
        info["physical_disks"] = out or err
    else:
        # Try lsblk
        if which("lsblk"):
            code, out, err = run_cmd(["lsblk", "-o", "NAME,MODEL,SIZE,TYPE,MOUNTPOINT" , "-J"])
            if code == 0 and out:
                info["lsblk"] = out
            else:
                info["lsblk_error"] = err
        # df for filesystem and device
        code, out, err = run_cmd(["df", "-h", path])
        info["df"] = out or err
    return info


def throughput_test(dir_path: str, size_mb: int = 256, block_kb: int = 1024) -> Dict[str, Any]:
    os.makedirs(dir_path, exist_ok=True)
    target = os.path.join(dir_path, ".diag_disk_test.tmp")
    size_bytes = size_mb * 1024 * 1024
    block_bytes = block_kb * 1024
    buf = bytearray(b"A" * block_bytes)
    result: Dict[str, Any] = {"path": target, "size_mb": size_mb, "block_kb": block_kb}

    # Write
    start = time.time()
    written = 0
    try:
        with open(target, "wb", buffering=0) as f:
            while written < size_bytes:
                n = min(block_bytes, size_bytes - written)
                f.write(buf[:n])
                written += n
            f.flush()
            os.fsync(f.fileno())
        write_time = max(1e-9, time.time() - start)
        result["write_bytes"] = written
        result["write_time_s"] = write_time
        result["write_MBps"] = (written / (1024 * 1024)) / write_time
    except Exception as e:
        result["write_error"] = str(e)

    # Read
    read_bytes = 0
    start = time.time()
    try:
        with open(target, "rb", buffering=0) as f:
            while True:
                chunk = f.read(block_bytes)
                if not chunk:
                    break
                read_bytes += len(chunk)
        read_time = max(1e-9, time.time() - start)
        result["read_bytes"] = read_bytes
        result["read_time_s"] = read_time
        result["read_MBps"] = (read_bytes / (1024 * 1024)) / read_time
    except Exception as e:
        result["read_error"] = str(e)

    # Cleanup
    try:
        os.remove(target)
    except Exception:
        pass

    return result


def disk_overview(path: str) -> Dict[str, Any]:
    return _device_info(path)


def fs_info(path: str) -> Dict[str, Any]:
    """Filesystem type and mount options. Best-effort per platform."""
    info: Dict[str, Any] = {"path": path}
    if is_linux():
        # parse /proc/mounts to find the mountpoint covering path
        mount: Optional[Tuple[str, str, str]] = None  # (device, mountpoint, fstype)
        try:
            with open("/proc/mounts", "r") as f:
                mounts: List[Tuple[str, str, str]] = []
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        mounts.append((parts[0], parts[1], parts[2]))
                # choose longest mountpoint prefix
                path_abs = os.path.abspath(path)
                best_len = -1
                for dev, mnt, fstype in mounts:
                    if path_abs.startswith(mnt) and len(mnt) > best_len:
                        best_len = len(mnt)
                        mount = (dev, mnt, fstype)
        except Exception as e:
            info["error"] = str(e)
        if mount:
            info["device"], info["mountpoint"], info["fstype"] = mount
        return info
    if is_windows():
        # Covered by Get-Volume in _device_info
        info["note"] = "Use disk_overview.volume for FileSystem"
        return info
    # macOS: minimal info (df -T not standard); rely on df string
    info["note"] = "Use disk_overview.df for FS details"
    return info


def random_io_test(dir_path: str, file_size_mb: int = 64, duration_sec: int = 5, block_kb: int = 4) -> Dict[str, Any]:
    """Random read/write I/O microbenchmark on a test file.
    Measures per-op latency and IOPS with fixed block size.
    """
    os.makedirs(dir_path, exist_ok=True)
    target = os.path.join(dir_path, ".diag_rand_io.tmp")
    size_bytes = file_size_mb * 1024 * 1024
    block_bytes = block_kb * 1024
    result: Dict[str, Any] = {
        "path": target,
        "file_size_mb": file_size_mb,
        "block_kb": block_kb,
        "duration_sec": duration_sec,
    }

    # Prepare file (sequential write once)
    try:
        with open(target, "wb", buffering=0) as f:
            buf = bytearray(os.urandom(block_bytes))
            written = 0
            while written < size_bytes:
                n = min(block_bytes, size_bytes - written)
                f.write(buf[:n])
                written += n
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        result["prepare_error"] = str(e)
        return result

    # Random R/W loop
    reads_lat: List[float] = []
    writes_lat: List[float] = []
    read_ops = write_ops = 0
    try:
        with open(target, "r+b", buffering=0) as f:
            end_t = time.time() + duration_sec
            while time.time() < end_t:
                off = random.randrange(0, size_bytes // block_bytes) * block_bytes
                # read
                t0 = time.time()
                f.seek(off)
                data = f.read(block_bytes)
                reads_lat.append(time.time() - t0)
                read_ops += 1
                # write (overwrite same block)
                t0 = time.time()
                f.seek(off)
                f.write(data or b"\x00" * block_bytes)
                writes_lat.append(time.time() - t0)
                write_ops += 1
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        result["randio_error"] = str(e)
    finally:
        try:
            os.remove(target)
        except Exception:
            pass

    def summarize(latencies: List[float]) -> Dict[str, Any]:
        if not latencies:
            return {"ops": 0}
        lat_ms = [x * 1000.0 for x in latencies]
        lat_ms.sort()
        p50 = lat_ms[len(lat_ms)//2]
        p95 = lat_ms[int(len(lat_ms) * 0.95) - 1]
        ops = len(latencies)
        iops = ops / max(1e-9, duration_sec)
        return {"ops": ops, "iops": iops, "p50_ms": p50, "p95_ms": p95}

    result["read"] = summarize(reads_lat)
    result["write"] = summarize(writes_lat)
    return result


def usb_storage_overview() -> Dict[str, Any]:
    """Linux: enumerate USB-backed block devices and mountpoints via lsblk.
    Other platforms: returns unsupported note.
    """
    if not is_linux():
        return {"supported": False, "note": "USB storage overview supported on Linux via lsblk"}
    if not which("lsblk"):
        return {"supported": False, "error": "lsblk not found"}
    code, out, err = run_cmd(["lsblk", "-OJ"])
    if code != 0 or not out:
        return {"supported": False, "error": err or "lsblk failed"}
    try:
        import json
        j = json.loads(out)
        items: List[Dict[str, Any]] = []
        def walk(children: List[Dict[str, Any]]):
            for c in children:
                name = c.get("name")
                tran = c.get("tran")
                type_ = c.get("type")
                model = c.get("model")
                size = c.get("size")
                mountpoint = c.get("mountpoint")
                if tran == "usb" and type_ in ("disk", "part"):
                    items.append({
                        "name": name, "model": model, "size": size, "type": type_, "mountpoint": mountpoint
                    })
                if c.get("children"):
                    walk(c["children"])
        walk(j.get("blockdevices", []))
        return {"supported": True, "devices": items}
    except Exception as e:
        return {"supported": False, "error": str(e)}
