from typing import Any, Dict, List
import json
import os


def _md_h2(text: str) -> str:
    return f"\n## {text}\n\n"


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    h = "| " + " | ".join(headers) + " |\n"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |\n"
    body = "".join("| " + " | ".join(str(c) if c is not None else "" for c in r) + " |\n" for r in rows)
    return h + sep + body + "\n"


def _num(x: Any, digits: int = 1) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def section_platform(r: Dict[str, Any]) -> str:
    p = r.get("platform", {})
    osname = p.get("os")
    arch = p.get("arch")
    distro = p.get("distro")
    tags = []
    if p.get("jetson"): tags.append("Jetson")
    if p.get("intel_nuc"): tags.append("Intel NUC")
    s = _md_h2("Platform")
    s += f"- OS: {osname}\n- Arch: {arch}\n"
    if distro: s += f"- Distro: {distro}\n"
    if tags: s += f"- Tags: {', '.join(tags)}\n"
    return s


def section_usb(r: Dict[str, Any]) -> str:
    u = r.get("usb", {})
    s = _md_h2("USB")
    # Device list (as lines)
    lst = u.get("list", {})
    devs = lst.get("devices") or []
    if isinstance(devs, list) and devs:
        rows = [[i, d] for i, d in enumerate(devs)]
        s += _md_table(["#", "Device"], rows)
    # Power attributes (Linux)
    p = u.get("power_attrs", {})
    if isinstance(p.get("devices"), list) and p.get("devices"):
        rows = []
        for d in p["devices"]:
            rows.append([
                d.get("device"),
                f"{(d.get('idVendor') or '')}:{(d.get('idProduct') or '')}",
                d.get("manufacturer"),
                d.get("product"),
                d.get("runtime_status"),
                d.get("control"),
            ])
        s += _md_table(["dev", "VID:PID", "Mfg", "Product", "Runtime", "PowerCtl"], rows)
    # UVC
    if u.get("uvc_test") is not None:
        t = u["uvc_test"]
        s += f"- UVC Test: {'PASS' if t.get('passed') else 'FAIL'}"
        meta = []
        if t.get("fps") is not None: meta.append(f"fps={_num(t.get('fps'),1)}")
        if t.get("frames") is not None: meta.append(f"frames={t.get('frames')}")
        if t.get("errors") is not None: meta.append(f"errors={t.get('errors')}")
        if t.get("reason"): meta.append(f"reason={t.get('reason')}")
        if meta: s += f" ({', '.join(meta)})"
        s += "\n"
    return s


def section_disk(r: Dict[str, Any]) -> str:
    d = r.get("disk", {})
    s = _md_h2("Disk")
    thr = d.get("throughput", {})
    if thr:
        rows = [[thr.get("path"), _num(thr.get("read_MBps")), _num(thr.get("write_MBps"))]]
        s += _md_table(["Path", "Read MB/s", "Write MB/s"], rows)
    rnd = d.get("random_io")
    if rnd:
        s += _md_table(
            ["Type", "IOPS", "p50 ms", "p95 ms"],
            [
                ["read", _num((rnd.get("read") or {}).get("iops")), _num((rnd.get("read") or {}).get("p50_ms")), _num((rnd.get("read") or {}).get("p95_ms"))],
                ["write", _num((rnd.get("write") or {}).get("iops")), _num((rnd.get("write") or {}).get("p50_ms")), _num((rnd.get("write") or {}).get("p95_ms"))],
            ],
        )
    usb = d.get("usb_storage")
    if usb and isinstance(usb.get("devices"), list) and usb["devices"]:
        rows = []
        for it in usb["devices"]:
            rows.append([it.get("name"), it.get("model"), it.get("size"), it.get("type"), it.get("mountpoint")])
        s += _md_table(["Dev", "Model", "Size", "Type", "Mount"], rows)
    return s


def section_mem_cpu_gpu(r: Dict[str, Any]) -> str:
    s = _md_h2("CPU / Memory / GPU")
    c = r.get("cpu", {}).get("stress", {})
    if c:
        per_core = None
        if c.get("ops_per_sec") and c.get("cores"):
            per_core = c["ops_per_sec"]/max(1,c["cores"])
        s += _md_table(["Cores", "Ops/s", "Ops/s/core"], [[c.get("cores"), _num(c.get("ops_per_sec"),0), _num(per_core,0) if per_core is not None else ""]])
    m = r.get("mem", {}).get("bandwidth", {})
    if m:
        s += _md_table(["Size(MB)", "Write MB/s", "Read MB/s"], [[m.get("size_mb"), _num(m.get("fill_MBps"),0), _num(m.get("read_MBps"),0)]])
    stats = r.get("stats", {}).get("gpu") or {}
    if stats:
        s += _md_table(["GPU Avg Util", "Score"], [[_num(stats.get("avg_util"),1), _num(stats.get("score"),1)]])
    return s


def section_stats_eval(r: Dict[str, Any]) -> str:
    s = _md_h2("Summary")
    st = r.get("stats", {})
    if st:
        s += _md_table(["CPU", "Disk", "Mem", "GPU", "Aggregate"], [[
            _num((st.get("cpu") or {}).get("score"),1),
            _num((st.get("disk") or {}).get("score"),1),
            _num((st.get("mem") or {}).get("score"),1),
            _num((st.get("gpu") or {}).get("score"),1),
            _num(st.get("aggregate_score"),1),
        ]])
    ev = r.get("evaluation", {})
    if ev:
        s += f"- Result: {'PASS' if ev.get('passed') else 'FAIL'}\n"
        fails = ev.get("failures") or []
        if fails:
            s += "- Failures:\n" + "\n".join(f"  - {f}" for f in fails) + "\n"
    return s


def to_markdown(reports: List[Dict[str, Any]]) -> str:
    out = ["# HardBench Report\n"]
    for i, r in enumerate(reports, start=1):
        ts = r.get("timestamp") or ""
        out.append(f"\n# Run {i} — {ts}\n")
        out.append(section_platform(r))
        out.append(section_usb(r))
        out.append(section_disk(r))
        out.append(section_mem_cpu_gpu(r))
        out.append(section_stats_eval(r))
    return "".join(out)


def generate_md(inputs: List[str], output: str) -> str:
    data: List[Dict[str, Any]] = []
    for path in inputs:
        with open(path, "r") as f:
            data.append(json.load(f))
    md = to_markdown(data)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(md)
    return output

