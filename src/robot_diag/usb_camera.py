import glob
import time
from typing import Any, Dict, Optional

from .utils import is_linux, is_macos, is_windows, which


def _opencv_uvc_test(index: int, duration_sec: int = 10, target_fps: Optional[int] = None) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as e:
        return {"supported": False, "error": f"opencv not available: {e}"}

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return {"supported": True, "passed": False, "reason": f"cannot open camera index {index}"}
    # Try to read properties
    fps_prop = cap.get(cv2.CAP_PROP_FPS) if hasattr(cv2, "CAP_PROP_FPS") else 0
    frames = 0
    start = time.time()
    last_ok = start
    errors = 0
    while time.time() - start < duration_sec:
        ok, frame = cap.read()
        if not ok or frame is None:
            errors += 1
            time.sleep(0.01)
            continue
        frames += 1
    cap.release()
    elapsed = max(1e-9, time.time() - start)
    fps = frames / elapsed

    # Basic checks
    if frames == 0:
        return {"supported": True, "passed": False, "reason": "no frames captured", "frames": frames, "fps": fps, "errors": errors}
    if target_fps and fps < target_fps * 0.5:
        return {"supported": True, "passed": False, "reason": f"fps {fps:.1f} below target {target_fps}", "frames": frames, "fps": fps, "errors": errors}
    return {"supported": True, "passed": True, "frames": frames, "fps": fps, "errors": errors, "camera_fps_prop": fps_prop}


def detect_devices() -> Dict[str, Any]:
    if is_linux():
        vids = sorted(glob.glob("/dev/video*"))
        return {"supported": True, "devices": vids}
    if is_macos():
        # macOS uses AVFoundation; device indices via OpenCV only
        return {"supported": True, "devices": ["index:0", "index:1"]}
    if is_windows():
        # OpenCV backend indexes; hard to enumerate without DirectShow
        return {"supported": True, "devices": ["index:0", "index:1"]}
    return {"supported": False}


def uvc_test(index: int = 0, duration_sec: int = 10, target_fps: Optional[int] = None) -> Dict[str, Any]:
    # Prefer OpenCV-based test for portability
    res = _opencv_uvc_test(index, duration_sec=duration_sec, target_fps=target_fps)
    if res.get("supported"):
        return res

    # Linux fallback: ensure device exists
    if is_linux():
        devs = detect_devices().get("devices", [])
        if not devs:
            return {"supported": True, "passed": False, "reason": "no /dev/video* devices"}
        return {"supported": True, "passed": True, "reason": "device present (no OpenCV)"}

    return {"supported": False, "passed": False, "reason": "no supported backend"}

