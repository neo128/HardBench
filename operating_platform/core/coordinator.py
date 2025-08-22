
import cv2
import json
import time
import draccus
import socketio
import requests
import traceback
import threading

from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pformat
from deepdiff import DeepDiff
from functools import cache
from termcolor import colored
from datetime import datetime
import subprocess

# from operating_platform.policy.config import PreTrainedConfig
from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots.utils import make_robot_from_config, Robot, busy_wait, safe_disconnect
from operating_platform.utils import parser
from operating_platform.utils.utils import has_method, init_logging, log_say
from operating_platform.utils.data_file import check_disk_space
from operating_platform.utils.constants import DOROBOT_DATASET
from operating_platform.dataset.dorobot_dataset import *

# from operating_platform.core._client import Coordinator
from operating_platform.core.daemon import Daemon
from operating_platform.core.record import Record, RecordConfig
import asyncio, aiohttp
DEFAULT_FPS = 25
file_local_path = None        
@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True

# def init_keyboard_listener():
#     # Allow to exit early while recording an episode or resetting the environment,
#     # by tapping the right arrow key '->'. This might require a sudo permission
#     # to allow your terminal to monitor keyboard events.
#     events = {}
#     events["exit_early"] = False
#     events["rerecord_episode"] = False
#     events["stop_recording"] = False

#     if is_headless():
#         logging.warning(
#             "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
#         )
#         listener = None
#         return listener, events

#     # Only import pynput if not in a headless environment
#     from pynput import keyboard

#     def on_press(key):
#         try:
#             if key == keyboard.Key.right:
#                 print("Right arrow key pressed. Exiting loop...")
#                 events["exit_early"] = True
#             elif key == keyboard.Key.left:
#                 print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
#                 events["rerecord_episode"] = True
#                 events["exit_early"] = True
#             elif key == keyboard.Key.esc:
#                 print("Escape key pressed. Stopping data recording...")
#                 events["stop_recording"] = True
#                 events["exit_early"] = True
#             elif key.char == 'q' or key.char == 'Q':  # 检测q键（不区分大小写）
#                 print("Q key pressed.")
#                 events["exit_early"] = True

#         except Exception as e:
#             print(f"Error handling key press: {e}")

#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()

#     return listener, events


def cameras_to_stream_json(cameras: dict[str, int]):
    """
    将摄像头字典转换为包含流信息的 JSON 字符串。
    
    参数:
        cameras (dict[str, int]): 摄像头名称到 ID 的映射
    
    返回:
        str: 格式化的 JSON 字符串
    """
    stream_list = [{"id": cam_id, "name": name} for name, cam_id in cameras.items()]
    # 修改depth
    result = {
        "total": len(stream_list),
        "streams": stream_list
    }
    return json.dumps(result)

class Coordinator:
    def __init__(self, daemon: Daemon, server_url="http://localhost:8088"):
        self.server_url = server_url
        # 1. 换成异步客户端
        self.sio = socketio.AsyncClient()
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=10)
        )
        self.daemon = daemon
        self.running = False
        self.heartbeat_interval = 2
        self.recording = False
        self.cameras = {"image_top": 1, "image_right": 2}

        # 2. 注册异步回调
        self.sio.on('HEARTBEAT_RESPONSE', self.__on_heartbeat_response_handle)
        self.sio.on('connect', self.__on_connect_handle)
        self.sio.on('disconnect', self.__on_disconnect_handle)
        self.sio.on('robot_command', self.__on_robot_command_handle)

        self.record = None
    
####################### Client Start/Stop ############################
    async def start(self):
        """启动客户端"""
        self.running = True
        await self.sio.connect(self.server_url)
        # 3. 用 asyncio 任务发心跳
        asyncio.create_task(self.send_heartbeat_loop())
        print("异步客户端已启动")

    
    async def stop(self):
        self.running = False
        await self.sio.disconnect()
        await self.session.close()
        print("异步客户端已停止")
    
####################### Client Handle ############################
    async def __on_heartbeat_response_handle(self, data):
        """心跳响应回调"""
        print("收到心跳响应:", data)
    
    async def __on_connect_handle(self):
        """连接成功回调"""
        print("成功连接到服务器")
        
        # # 初始化视频流列表
        # try:
        #     response = self.session.post(
        #         f"{self.server_url}/robot/stream_info",
        #         json = cameras_to_stream_json(self.cameras),
        #     )
        #     print("初始化视频流列表:", response.json())
        # except Exception as e:
        #     print(f"初始化视频流列表失败: {e}")
    
    async def __on_disconnect_handle(self):
        """断开连接回调"""
        print("与服务器断开连接")
    
    async def __on_robot_command_handle(self, data):
        """收到机器人命令回调"""
        print("收到服务器命令:", data)
        global file_local_path

        # 根据命令类型进行响应
        if data.get('cmd') == 'video_list':
            print("处理更新视频流命令...")
            response_data = cameras_to_stream_json(self.cameras)
            # 发送响应
            try:
                response = self.session.post(
                    f"{self.server_url}/robot/stream_info",
                    json = response_data,
                )
                print(f"已发送响应 [{data.get('cmd')}]: {response_data}")
            except Exception as e:
                print(f"发送响应失败 [{data.get('cmd')}]: {e}")
            
        elif data.get('cmd') == 'start_collection':
            print("处理开始采集命令...")
            if not check_disk_space(min_gb=2):  # 检查是否 ≥1GB
                print("存储空间不足,小于2GB,取消采集！")
                await self.send_response('start_collection', "存储空间不足,小于2GB")
            msg = data.get('msg')

            if self.recording == True:
                # self.send_response('start_collection', "fail")

                self.record.stop(save=False)
                self.recording = False


            self.recording = True

            task_id = msg.get('task_id')
            task_name = msg.get('task_name')
            task_data_id = msg.get('task_data_id')
            countdown_seconds = msg.get('countdown_seconds', 3) 
            repo_id=f"{task_name}_{task_id}"

            date_str = datetime.now().strftime("%Y%m%d")

            # 构建目标目录路径
            dataset_path = DOROBOT_DATASET
            target_dir = dataset_path / date_str / "user" / repo_id

            # 判断是否存在对应文件夹以决定是否启用恢复模式
            resume = False

            # 检查数据集目录是否存在
            if not dataset_path.exists():
                logging.info(f"Dataset directory '{dataset_path}' does not exist. Cannot resume.")
            else:
                # 检查目标文件夹是否存在且为目录
                if target_dir.exists() and target_dir.is_dir():
                    resume = True
                    logging.info(f"Found existing directory for repo_id '{repo_id}'. Resuming operation.")
                else:
                    logging.info(f"No directory found for repo_id '{repo_id}'. Starting fresh.")

            # resume 变量现在可用于后续逻辑
            print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")

            record_cfg = RecordConfig(fps=DEFAULT_FPS, repo_id=repo_id, resume=resume, root=target_dir)
            self.record = Record(fps=DEFAULT_FPS, robot=self.daemon.robot, daemon=self.daemon, record_cfg = record_cfg, record_cmd=msg)
            self.record.time = countdown_seconds
            self.record.start()

            # 发送响应
            await self.send_response('start_collection', "success")
        
        elif data.get('cmd') == 'finish_collection':
            # 模拟处理完成采集
            print("处理完成采集命令...")

            data = self.record.stop(save=True)
            self.recording = False
            file_local_path = data.get('file_message', {}).get('file_local_path')
            # 准备响应数据
            response_data = {
                "msg": "success",
                "data": data,
            }
            print("获取到路径：", file_local_path)
            # 发送响应
            await self.send_response('finish_collection', response_data['msg'], response_data)

        elif data.get('cmd') == 'discard_collection':
            # 模拟处理丢弃采集
            print("处理丢弃采集命令...")

            self.record.stop(save=False)
            self.recording = False

            # 发送响应
            await self.send_response('discard_collection', "success")

        elif data.get('cmd') == 'submit_collection':
            # 模拟处理提交采集
            print("处理提交采集命令...")
            time.sleep(0.01)  # 模拟处理时间
            
            # 发送响应
            await self.send_response('submit_collection', "success")
        elif data.get('cmd') == 'start_replay':
            print("开始进行replay")
            # 要跑的脚本路径和参数
            script_path = '/root/Operating-Platform/operating_platform/robot/robots/realman_v1/realman_replay.py'
            print(data)
            data_path =  file_local_path
            json_file_dir = os.path.join(data_path, "meta")
            os.makedirs(json_file_dir, exist_ok=True)
            json_file = os.path.join(json_file_dir, "op_dataid.jsonl")
            with open(json_file) as f:
                last_line = f.readlines()[-1]
            episode_index = json.loads(last_line)['episode_index']
            parquet_path = os.path.join(
            data_path, "data", "chunk-000",
            f"episode_{episode_index:06d}.parquet"
            )
            cmd = [
                'python', script_path,
                '--parquet.path', parquet_path
            ]
            
            response_data = {
                "msg": "success",
                "data": {"url":"www.baidu.com"},
            }
            await self.send_response('start_replay', response_data['msg'], response_data)
            subprocess.Popen(cmd, cwd=os.path.dirname(script_path))
            await self.send_response('end_replay', response_data['msg'], response_data)

####################### Client Send to Server ############################
    async def send_heartbeat_loop(self):
        while self.running:
            try:
                await self.sio.emit('HEARTBEAT')
            except Exception as e:
                print("发送心跳失败:", e)
            await asyncio.sleep(self.heartbeat_interval)

    # 发送回复请求
    async def send_response(self, cmd, msg, data=None):
        payload = {"cmd": cmd, "msg": msg}
        if data:
            payload.update(data)
        try:
            async with self.session.post(
                f"{self.server_url}/robot/response",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                print(f"已发送响应 [{cmd}]: {payload}")
        except Exception as e:
            print(f"发送响应失败 [{cmd}]: {e}")

####################### Robot API ############################
    def stream_info(self, info: dict[str, int]):
        self.cameras = info.copy()
        print(f"更新摄像头信息: {self.cameras}")

    async def update_stream_info_to_server(self):
        stream_info_data = cameras_to_stream_json(self.cameras)
        print(f"stream_info_data: {stream_info_data}")
        try:
            # 2. 异步post加await，确保请求发送
            async with self.session.post(
                f"{self.server_url}/robot/stream_info",
                json=stream_info_data,
                timeout=aiohttp.ClientTimeout(total=2)
            ) as response:
                if response.status == 200:
                    print("摄像头流信息已同步到服务器")
                else:
                    print(f"同步流信息失败: {response.status}")
        except Exception as e:
            print(f"同步流信息异常: {e}")

    def update_stream(self, name, frame):

        _, jpeg_frame = cv2.imencode('.jpg', frame, 
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        frame_data = jpeg_frame.tobytes()
        stream_id = self.cameras[name]
        # 不在浏览器界面显示深度信息
        if "depth" in name:
            return
        # Build URL
        url = f"{self.server_url}/robot/update_stream/{stream_id}"
        # Send POST request
        try:
            response = self.session.post(url, data=frame_data)
            if response.status_code != 200:
                print(f"Server returned error: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            
    async def update_stream_async(self, name, frame):
        if "depth" in name:
            return
        _, jpeg = cv2.imencode('.jpg', frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        url = f"{self.server_url}/robot/update_stream/{self.cameras[name]}"
        try:
            # 超时给短一点，丢几帧对视频流影响不大
            async with self.session.post(url, data=jpeg.tobytes(),
                                         timeout=aiohttp.ClientTimeout(total=0.2)) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    print(f"Server error {resp.status}: {txt}")
        except asyncio.TimeoutError:
            print("update_stream timeout")
        except Exception as e:
            print("update_stream exception:", e)

@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    # control: ControlConfig

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]
    

@parser.wrap()
def main(cfg: ControlPipelineConfig):
# 让事件循环跑 async_main
    asyncio.run(async_main(cfg))
    
async def async_main(cfg: ControlPipelineConfig):
    """原来的 async 主体"""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    daemon = Daemon(fps=DEFAULT_FPS)
    daemon.start(cfg.robot)

    coordinator = Coordinator(daemon)
    await coordinator.start()

    coordinator.stream_info(daemon.cameras_info)
    await coordinator.update_stream_info_to_server()

    try:
        while True:
            daemon.update()
            observation = daemon.get_observation()
            if observation is not None:
                tasks = []
                for key in observation:
                    if "image" in key and "depth" not in key:
                        img = cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR)
                        name = key[len("observation.images."):]
                        tasks.append(
                            coordinator.update_stream_async(name, img)
                        )
                if tasks:
                    # 并发地发；只等待 0.2 s，不阻塞主循环
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=0.2
                        )
                    except asyncio.TimeoutError:
                        pass
            else:
                print("observation is none")
            await asyncio.sleep(0)   # 让事件循环可以调度
    except KeyboardInterrupt:
        print("coordinator and daemon stop")
    finally:
        daemon.stop()
        await coordinator.stop()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
