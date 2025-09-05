from typing import Any, Dict, List


def _get_defaults(platform: Dict[str, Any]) -> Dict[str, Any]:
    osname = (platform or {}).get("os", "unknown")
    arch = (platform or {}).get("arch", "")
    jetson = bool((platform or {}).get("jetson"))
    nuc = bool((platform or {}).get("intel_nuc"))

    # Conservative defaults; can be overridden by env or args later
    disk_read = 70.0
    disk_write = 60.0
    mem_read = 2000.0  # MB/s (pure Python baseline, conservative)
    mem_write = 1500.0
    cpu_ops = 5_000  # ops/sec per core baseline for our micro workload

    if osname == "linux":
        # Linux robot targets often have better IO paths
        disk_read = 80.0
        disk_write = 70.0
    if jetson:
        # eMMC or SD; lower expectations
        disk_read = 45.0
        disk_write = 35.0
        mem_read = 1200.0
        mem_write = 900.0
        cpu_ops = 2500
    if nuc:
        disk_read = max(disk_read, 120.0)
        disk_write = max(disk_write, 100.0)
        mem_read = max(mem_read, 2500.0)
        mem_write = max(mem_write, 2000.0)
        cpu_ops = max(cpu_ops, 8000)

    return {
        "disk_min_read_MBps": disk_read,
        "disk_min_write_MBps": disk_write,
        "mem_min_read_MBps": mem_read,
        "mem_min_write_MBps": mem_write,
        "cpu_min_ops_per_sec_per_core": cpu_ops,
        # random I/O
        "rand_read_min_iops": 500.0,
        "rand_write_min_iops": 300.0,
        "rand_read_max_p95_ms": 20.0,
        "rand_write_max_p95_ms": 30.0,
    }


def evaluate(results: Dict[str, Any]) -> Dict[str, Any]:
    platform = results.get("platform", {})
    th = _get_defaults(platform)
    failures: List[str] = []

    # Disk seq
    d = results.get("disk", {}).get("throughput", {})
    if d:
        wr = d.get("write_MBps")
        rd = d.get("read_MBps")
        if isinstance(wr, (int, float)) and wr < th["disk_min_write_MBps"]:
            failures.append(f"Disk write throughput {wr:.1f} < {th['disk_min_write_MBps']:.1f} MB/s")
        if isinstance(rd, (int, float)) and rd < th["disk_min_read_MBps"]:
            failures.append(f"Disk read throughput {rd:.1f} < {th['disk_min_read_MBps']:.1f} MB/s")

    # Mem
    m = results.get("mem", {}).get("bandwidth", {})
    if m:
        w = m.get("fill_MBps")
        r = m.get("read_MBps")
        # adapt read threshold if using python mode
        eff_read_min = th["mem_min_read_MBps"]
        mode_eff = (m.get("mode_effective") or "").lower()
        if mode_eff == "python":
            eff_read_min = min(eff_read_min, 400.0)
        if isinstance(w, (int, float)) and w < th["mem_min_write_MBps"]:
            failures.append(f"Memory write bandwidth {w:.0f} < {th['mem_min_write_MBps']:.0f} MB/s")
        if isinstance(r, (int, float)) and r < eff_read_min:
            failures.append(f"Memory read bandwidth {r:.0f} < {eff_read_min:.0f} MB/s")

    # CPU
    c = results.get("cpu", {}).get("stress", {})
    if c:
        cores = c.get("cores", 1)
        opsps = c.get("ops_per_sec")
        if isinstance(opsps, (int, float)):
            per_core = opsps / max(1, cores)
            if per_core < th["cpu_min_ops_per_sec_per_core"]:
                failures.append(
                    f"CPU ops/s/core {per_core:.0f} < {th['cpu_min_ops_per_sec_per_core']:.0f}"
                )

    # USB camera test if present
    cam = results.get("usb", {}).get("uvc_test")
    if cam and not cam.get("passed", True):
        failures.append("UVC camera test failed: " + cam.get("reason", "unknown"))

    # Random I/O
    r = results.get("disk", {}).get("random_io")
    if r:
        rr = (r.get("read") or {})
        rw = (r.get("write") or {})
        if isinstance(rr.get("iops"), (int, float)) and rr["iops"] < th["rand_read_min_iops"]:
            failures.append(f"Rand read IOPS {rr['iops']:.0f} < {th['rand_read_min_iops']:.0f}")
        if isinstance(rw.get("iops"), (int, float)) and rw["iops"] < th["rand_write_min_iops"]:
            failures.append(f"Rand write IOPS {rw['iops']:.0f} < {th['rand_write_min_iops']:.0f}")
        if isinstance(rr.get("p95_ms"), (int, float)) and rr["p95_ms"] > th["rand_read_max_p95_ms"]:
            failures.append(f"Rand read p95 {rr['p95_ms']:.1f}ms > {th['rand_read_max_p95_ms']:.1f}ms")
        if isinstance(rw.get("p95_ms"), (int, float)) and rw["p95_ms"] > th["rand_write_max_p95_ms"]:
            failures.append(f"Rand write p95 {rw['p95_ms']:.1f}ms > {th['rand_write_max_p95_ms']:.1f}ms")

    passed = len(failures) == 0
    return {
        "thresholds": th,
        "passed": passed,
        "failures": failures,
    }
