import json
import logging
import subprocess
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Literal
import re

import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image
import platform


def get_available_encoders():
    """
    获取当前系统中 ffmpeg 支持的视频编码器列表。
    """
    try:
        # 执行 ffmpeg -encoders 命令
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        output = result.stdout

        # # 调试：打印完整输出
        # print("ffmpeg output:")
        # print(output)

        encoders = set()
        in_encoders = False

        for line in output.splitlines():
            line = line.strip()

            # 检测到 Encoders: 行，开始解析编码器
            if line == "Encoders:":
                in_encoders = True
                continue

            if not in_encoders:
                continue

            # 匹配视频编码器行：V + 5个非空白字符 + 空格 + 编码器名称
            match = re.match(r"^\s*V\S{5}\s+(\S+)", line)
            if match:
                encoders.add(match.group(1))

        print(f"Get ffmpeg encoders: {encoders}")

        return encoders

    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg not found or failed to execute. "
            "Please ensure ffmpeg is installed and available in your PATH."
        )
    
# 缓存编码器列表，避免重复调用 ffmpeg
_AVAILABLE_ENCODERS = None

def _ensure_encoders_loaded():
    global _AVAILABLE_ENCODERS
    if _AVAILABLE_ENCODERS is None:
        _AVAILABLE_ENCODERS = get_available_encoders()

def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames
test = False
if test:
    def encode_video_frames(
        imgs_dir: Path | str,
        video_path: Path | str,
        fps: int,
        vcodec: Literal["libopenh264", "libx264", "h264_omx", "h264_v4l2m2m", "h264_vaapi", 
                    "h264_nvenc", "hevc_nvenc", "av1_nvenc"] = "h264_nvenc",  # 新增NVENC选项
        pix_fmt: str = "yuv420p",
        g: int | None = None,
        crf: int | None = 25,
        fast_decode: int = 0,
        log_level: Optional[str] = "error",
        overwrite: bool = False,
    ) -> None:
        """优化版视频编码函数，支持 NVENC 硬件编码器（NVIDIA Jetson 专用）"""
        print(f"当前帧率为 {fps}")
        # 确保编码器列表已加载
        _ensure_encoders_loaded()
        available_encoders = _AVAILABLE_ENCODERS

        # 获取系统架构信息（判断是否为 NVIDIA Jetson 设备）
        arch = platform.machine()
        is_aarch64 = arch == "aarch64"
        is_nvidia = "nvenc" in str(available_encoders)  # 检测是否有NVENC编码器

        # 用户指定的编码器是否可用
        if vcodec in available_encoders:
            # 在NVIDIA设备上优先确认NVENC可用性
            if is_nvidia and vcodec in ["h264_v4l2m2m", "h264_omx"]:
                warnings.warn(f"检测到NVIDIA设备，建议使用 {vcodec.replace('v4l2m2m', 'nvenc')} 获得更好性能")
        else:
            # 自动选择编码器（NVIDIA设备优先NVENC）
            if is_nvidia:
                # NVIDIA设备硬件编码器优先级：NVENC > v4l2m2m > vaapi > omx
                hardware_encoders = ["h264_nvenc", "hevc_nvenc", "av1_nvenc", 
                                    "h264_v4l2m2m", "h264_vaapi", "h264_omx"]
            elif is_aarch64:
                # 非NVIDIA aarch64设备（如其他ARM开发板）
                hardware_encoders = ["h264_v4l2m2m", "h264_vaapi", "h264_omx"]
            else:
                # x86设备
                hardware_encoders = ["h264_omx", "h264_vaapi", "h264_v4l2m2m"]
            
            # 筛选可用编码器
            supported_candidates = set(hardware_encoders + ["libx264", "libopenh264"]) & set(available_encoders)
            if not supported_candidates:
                raise ValueError("无可用编码器，请检查FFmpeg安装配置")

            # 选择最优编码器
            selected_vcodec = None
            for encoder in hardware_encoders + ["libx264", "libopenh264"]:
                if encoder in supported_candidates:
                    selected_vcodec = encoder
                    break

            # 提示编码器类型
            if selected_vcodec in hardware_encoders:
                print(f"使用硬件编码器: {selected_vcodec} (加速编码)")
            else:
                warnings.warn(f"使用软件编码器: {selected_vcodec}. 建议启用硬件加速")
            vcodec = selected_vcodec

        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        # 基础FFmpeg参数
        ffmpeg_args = OrderedDict([
            ("-f", "image2"),
            ("-r", str(fps)),
            ("-i", str(Path(imgs_dir) / "frame_%06d.jpg")),  # 输入图片路径格式
            ("-vf", f"format={pix_fmt},scale=trunc(iw/2)*2:trunc(ih/2)*2"),  # 确保宽高为偶数
            ("-vcodec", vcodec),
            ("-pix_fmt", pix_fmt),
        ])

        # 编码器特定参数配置
        # 1. NVENC编码器（h264_nvenc / hevc_nvenc / av1_nvenc）
        if vcodec in ["h264_nvenc", "hevc_nvenc", "av1_nvenc"]:
            # 关键帧间隔（默认2倍帧率，保证流畅度）
            if g is None:
                g = fps * 2
            ffmpeg_args["-g"] = str(g)
            
            # 质量控制（NVENC推荐用CRF，范围0-51，23-28为常用值）
            if crf is not None:
                # 根据编码器类型调整CRF范围
                if vcodec == "av1_nvenc":
                    crf = max(0, min(63, crf))  # AV1 CRF范围0-63
                else:
                    crf = max(0, min(51, crf))  # H.264/H.265 CRF范围0-51
                ffmpeg_args["-crf"] = str(crf)
            
            # 编码预设（平衡速度与质量）：fast < medium < slow < slowest
            # 实时场景用fast/medium，离线编码用slow
            ffmpeg_args["-preset"] = "medium"
            
            # 启用B帧（提升压缩效率，NVENC支持良好）
            ffmpeg_args["-bf"] = "2"  # 2个B帧
            ffmpeg_args["-refs"] = "3"  # 参考帧数
            
            # NVIDIA特有优化：低延迟模式（实时场景启用）
            if fast_decode:
                ffmpeg_args["-preset"] = "llhq"  # 低延迟高画质
                ffmpeg_args["-rc-lookahead"] = "16"  # 减少前瞻帧数
            
            # 编码器特性：H.264用main/high profile，HEVC用main
            if vcodec == "h264_nvenc":
                ffmpeg_args["-profile:v"] = "high"  # 支持更多特性
            elif vcodec == "hevc_nvenc":
                ffmpeg_args["-profile:v"] = "main"

        # 2. 其他硬件编码器（v4l2m2m / vaapi / omx）
        elif vcodec in ["h264_v4l2m2m", "h264_vaapi", "h264_omx"]:
            # 分辨率对齐（硬件要求）
            ffmpeg_args["-vf"] = "format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2"
            
            # 质量参数（v4l2m2m用-q:v，范围1-31）
            if crf is not None:
                q_value = max(1, min(31, crf))  # 映射CRF到v4l2m2m的质量范围
                ffmpeg_args["-q:v"] = str(q_value)
            
            # 关键帧间隔
            ffmpeg_args["-g"] = str(g) if g is not None else str(fps * 2)
            
            # 禁用B帧（部分硬件不支持）
            ffmpeg_args["-bf"] = "0"
            ffmpeg_args["-refs"] = "1"
            
            # VAAPI设备指定
            if vcodec == "h264_vaapi":
                ffmpeg_args["-vaapi_device"] = "/dev/dri/renderD128"

        # 3. 软件编码器（libx264等）
        else:
            if crf is not None:
                ffmpeg_args["-crf"] = str(crf)
            if g is not None:
                ffmpeg_args["-g"] = str(g)
            
            # 软件编码器优化
            if fast_decode:
                ffmpeg_args["-tune"] = "fastdecode"
            ffmpeg_args["-preset"] = "medium"

        # 通用优化参数
        ffmpeg_args["-threads"] = "0"  # 自动多线程
        ffmpeg_args["-movflags"] = "+faststart"  # 视频文件优化（快速开始播放）
        ffmpeg_args["-pix_fmt"] = pix_fmt  # 确保输出像素格式

        # 日志级别
        if log_level is not None:
            ffmpeg_args["-loglevel"] = log_level

        # 构建FFmpeg命令
        print("开始构建FFmpeg命令...")
        ffmpeg_cmd = ["ffmpeg"] + [item for pair in ffmpeg_args.items() for item in pair]
        print(f"构建命令: {' '.join(ffmpeg_cmd)}")

        # 覆盖输出文件
        if overwrite:
            ffmpeg_cmd.append("-y")
        ffmpeg_cmd.append(str(video_path))

        # 执行编码命令
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"视频编码成功: {video_path}")
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg 编码失败 (退出码 {e.returncode}):\n"
            error_msg += f"命令: {' '.join(e.cmd)}\n"
            error_msg += f"错误输出:\n{e.stderr}\n"
            raise RuntimeError(error_msg) from e

        # 验证文件生成
        if not video_path.exists():
            raise OSError(f"视频编码失败，文件未生成: {video_path}")
else:
    def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: Literal["libopenh264", "libx264"] = "libx264",
    pix_fmt: str = "yuv420p",
    g: int | None = 18,
    crf: int | None = 23,
    fast_decode: int = 1,
    log_level: Optional[str] = "error",
    overwrite: bool = False,
) -> None:
        """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""

        # 确保编码器列表已加载
        _ensure_encoders_loaded()

        # 获取当前支持的编码器列表
        available_encoders = _AVAILABLE_ENCODERS

        # 用户指定的编码器是否可用
        if vcodec in available_encoders:
            pass  # 正常使用指定的编码器
        else:
            # 从支持的两个编码器中选择一个可用的
            supported_candidates = {"libopenh264", "libx264"} & set(available_encoders)

            if not supported_candidates:
                raise ValueError(
                    "None of the supported encoders are available. "
                    "Please ensure at least one of 'libopenh264' or 'libx264' is supported by your ffmpeg installation."
                )

            # 优先选择 libx264，否则选择 libopenh264
            selected_vcodec = "libx264" if "libx264" in supported_candidates else "libopenh264"

            # 发出警告
            warnings.warn(
                f"vcodec '{vcodec}' not available. Automatically switched to '{selected_vcodec}'.",
                UserWarning
            )

            vcodec = selected_vcodec

        # 剩余逻辑不变（略去，与原函数一致）
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_args = OrderedDict(
            [
                ("-f", "image2"),
                ("-r", str(fps)),
                ("-i", str(imgs_dir / "frame_%06d.jpg")),
                ("-vcodec", vcodec),
                ("-pix_fmt", pix_fmt),
            ]
        )

        if g is not None:
            ffmpeg_args["-g"] = str(g)

        if crf is not None:
            ffmpeg_args["-crf"] = str(crf)

        if fast_decode:
            key = "-svtav1-params" if vcodec == "libsvtav1" else "-tune"
            value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
            ffmpeg_args[key] = value

        if log_level is not None:
            ffmpeg_args["-loglevel"] = str(log_level)

        ffmpeg_args = [item for pair in ffmpeg_args.items() for item in pair]
        if overwrite:
            ffmpeg_args.append("-y")

        ffmpeg_cmd = ["ffmpeg"] + ffmpeg_args + [str(video_path)]
        # redirect stdin to subprocess.DEVNULL to prevent reading random keyboard inputs from terminal
        subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)

        if not video_path.exists():
            raise OSError(
                f"Video encoding did not work. File not found: {video_path}. "
                f"Try running the command manually to debug: `{''.join(ffmpeg_cmd)}`"
            )

@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    ffprobe_audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,codec_name,bit_rate,sample_rate,bit_depth,channel_layout,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    audio_stream_info = info["streams"][0] if info.get("streams") else None
    if audio_stream_info is None:
        return {"has_audio": False}

    # Return the information, defaulting to None if no audio stream is present
    return {
        "has_audio": True,
        "audio.channels": audio_stream_info.get("channels", None),
        "audio.codec": audio_stream_info.get("codec_name", None),
        "audio.bit_rate": int(audio_stream_info["bit_rate"]) if audio_stream_info.get("bit_rate") else None,
        "audio.sample_rate": int(audio_stream_info["sample_rate"])
        if audio_stream_info.get("sample_rate")
        else None,
        "audio.bit_depth": audio_stream_info.get("bit_depth", None),
        "audio.channel_layout": audio_stream_info.get("channel_layout", None),
    }


def get_video_info(video_path: Path | str) -> dict:
    ffprobe_video_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height,codec_name,nb_frames,duration,pix_fmt",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream_info = info["streams"][0]

    # Calculate fps from r_frame_rate
    r_frame_rate = video_stream_info["r_frame_rate"]
    num, denom = map(int, r_frame_rate.split("/"))
    fps = num / denom

    pixel_channels = get_video_pixel_channels(video_stream_info["pix_fmt"])

    video_info = {
        "video.fps": fps,
        "video.height": video_stream_info["height"],
        "video.width": video_stream_info["width"],
        "video.channels": pixel_channels,
        "video.codec": video_stream_info["codec_name"],
        "video.pix_fmt": video_stream_info["pix_fmt"],
        "video.is_depth_map": False,
        **get_audio_info(video_path),
    }

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
