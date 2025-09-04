import os
import time
from typing import Any, Dict


def mem_bandwidth(size_mb: int = 512, passes: int = 2) -> Dict[str, Any]:
    size_bytes = size_mb * 1024 * 1024
    buf = bytearray(os.urandom(1024 * 1024))  # 1MB seed
    dst = bytearray(size_bytes)
    result: Dict[str, Any] = {"size_mb": size_mb, "passes": passes}

    # Write/fill
    start = time.time()
    for _ in range(passes):
        mv = memoryview(dst)
        offset = 0
        while offset < size_bytes:
            n = min(len(buf), size_bytes - offset)
            mv[offset:offset+n] = buf[:n]
            offset += n
    fill_time = max(1e-9, time.time() - start)
    result["fill_time_s"] = fill_time
    result["fill_MBps"] = (size_mb * passes) / fill_time

    # Verify pattern read
    start = time.time()
    checksum = 0
    for _ in range(passes):
        # simple checksum to enforce read
        for b in dst:
            checksum = (checksum + b) & 0xFFFFFFFF
    read_time = max(1e-9, time.time() - start)
    result["read_time_s"] = read_time
    result["read_MBps"] = (size_mb * passes) / read_time
    result["checksum"] = checksum
    return result


def mem_stress(size_mb: int = 1024, duration_sec: int = 30) -> Dict[str, Any]:
    size_bytes = size_mb * 1024 * 1024
    start = time.time()
    dst = bytearray(size_bytes)
    iters = 0
    errors = 0
    while time.time() - start < duration_sec:
        pattern = (iters & 0xFF)
        dst[:] = bytes([pattern]) * len(dst)
        # basic verification on sample offsets
        for off in (0, len(dst)//3, (2*len(dst))//3, len(dst)-1):
            if dst[off] != pattern:
                errors += 1
        iters += 1
    return {"size_mb": size_mb, "duration_sec": duration_sec, "iterations": iters, "sample_errors": errors}

