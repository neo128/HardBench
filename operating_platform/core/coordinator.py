
import cv2
import json
import time
import draccus
import socketio
import requests
import traceback
import threading
import queue
import tempfile

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
from operating_platform.utils.utils import has_method, init_logging, log_say, get_current_git_branch, git_branch_log, get_container_ip_from_hosts
from operating_platform.utils.data_file import find_epindex_from_dataid_json

from operating_platform.utils.data_file import check_disk_space
from operating_platform.utils.constants import DOROBOT_DATASET
from operating_platform.dataset.dorobot_dataset import *
from operating_platform.dataset.visual.visual_dataset import visualize_dataset
from operating_platform.dataset.visual.visualize_dataset_html import visualize_dataset_html
# from operating_platform.core._client import Coordinator
from operating_platform.core.daemon import Daemon
from operating_platform.core.record import Record, RecordConfig
from operating_platform.core.replay import DatasetReplayConfig, ReplayConfig, replay

import asyncio, aiohttp
DEFAULT_FPS = 30
RERUN_WEB_PORT = 9195
RERUN_WS_PORT = 9285

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
        self.last_heartbeat_time = 0
        self.heartbeat_interval = 2
        self.recording = False
        self.replaying = False
        self.saveing = False

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
        global task_id
        global task_name
        global task_data_id
        global repo_id
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

            if self.replaying == True:
                await self.send_response('start_collection', "fail")
                print("Replay is running, cannot start collection.")
                return
            if self.recording == True:
                # self.send_response('start_collection', "fail")

                self.record.stop()
                self.record.discard()
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

            git_branch_name = get_current_git_branch()
            if "release" in git_branch_name:
                target_dir = dataset_path / date_str / "user" / repo_id
            elif "dev"  in git_branch_name:
                target_dir = dataset_path / date_str / "dev" / repo_id
            else:
                target_dir = dataset_path / date_str / "dev" / repo_id

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
            # 发送响应
            await self.send_response('start_collection', "success")
            # 开始采集倒计时
            print(f"开始采集倒计时{countdown_seconds}s...")
            time.sleep(countdown_seconds)

            # 开始采集
            self.record.start()

        
        elif data.get('cmd') == 'finish_collection':
            # 模拟处理完成采集
            print("处理完成采集命令...")
            if self.replaying == True:
                await self.send_response('finish_collection', "fail")
                print("Replay is running, cannot finish collection.")
                return
            
            if not self.saveing and self.record.save_data is None:
                # 如果不在保存状态，立即停止记录并保存
                self.saveing= True
                self.record.stop()
                self.record.save()
                self.recording = False
                self.saveing= False
            
            # 如果正在保存，循环等待直到 self.record.save_data 有数据
            while self.saveing:
                time.sleep(0.1)  # 避免CPU过载，适当延迟
            # 此时无论 saveing 状态如何，self.record.save_data 已有有效数据
            response_data = {
                "msg": "success",
                "data": self.record.save_data,
            }
            # 发送响应
            await self.send_response('finish_collection', response_data['msg'], response_data)

        elif data.get('cmd') == 'discard_collection':
            # 模拟处理丢弃采集
            print("处理丢弃采集命令...")

            if self.replaying == True:
                self.send_response('discard_collection', "fail")
                print("Replay is running, cannot discard collection.")
                return
            
            self.record.stop()
            self.record.discard()
            self.recording = False

            # 发送响应
            await self.send_response('discard_collection', "success")

        elif data.get('cmd') == 'submit_collection':
            # 模拟处理提交采集
            print("处理提交采集命令...")
            time.sleep(0.01)  # 模拟处理时间

            if self.replaying == True:
                await self.send_response('submit_collection', "fail")
                print("Replay is running, cannot submit collection.")
                return
            # 发送响应
            await self.send_response('submit_collection', "success")

        elif data.get('cmd') == 'start_replay':
            print("处理开始回放命令...")
            # msg拿不到信息
            # msg = data.get()
            if self.recording == True:
                await self.send_response('start_replay', "fail")
                print("Recording is running, cannot start replay.")
                return
            if self.replaying == True:
                await self.send_response('start_replay', "fail")
                print("Replay is already running.")
                return
            self.replaying = True
            # task_id = msg.get('task_id')
            # task_name = msg.get('task_name')
            # task_data_id = msg.get('task_data_id')
            # repo_id=f"{task_name}_{task_id}"

            date_str = datetime.now().strftime("%Y%m%d")

            # 构建目标目录路径
            dataset_path = DOROBOT_DATASET
            git_branch_name = get_current_git_branch()
            if "release" in git_branch_name:
                target_dir = dataset_path / date_str / "user" / repo_id
            elif "dev"  in git_branch_name:
                target_dir = dataset_path / date_str / "dev" / repo_id
            else:
                target_dir = dataset_path / date_str / "dev" / repo_id

            ep_index = find_epindex_from_dataid_json(target_dir, task_data_id)
            
            dataset = DoRobotDataset(repo_id, root=target_dir)
       
            print(f"开始回放数据集: {repo_id}, 目标目录: {target_dir}, 任务数据ID: {task_data_id}, 回放索引: {ep_index}")

            replay_dataset_cfg = DatasetReplayConfig(repo_id, ep_index, target_dir, fps=DEFAULT_FPS)
            replay_cfg = ReplayConfig(self.daemon.robot, replay_dataset_cfg)
            
            # 用于线程间通信的异常队列
            error_queue = queue.Queue()
            # 用于通知replay线程停止的事件
            stop_event = threading.Event()

            def visual_worker():
                """visual工作线程函数"""
                try:
                    # 主线程执行可视化（阻塞直到窗口关闭或超时）
                    visualize_dataset(
                        dataset,
                        mode="local",
                        episode_index=ep_index,
                        web_port=RERUN_WEB_PORT,
                        ws_port=RERUN_WS_PORT,
                        stop_event=stop_event  # 需要replay函数支持stop_event参数
                    )
                except Exception as e:
                    error_queue.put(e)
            # 创建并启动replay线程
            visual_thread = threading.Thread(
                target=visual_worker,
                name="VisualThread",
                daemon=True  # 设置为守护线程，主程序退出时自动终止
            )
            visual_thread.start()

            # 发送响应
            response_data = {
                "data": {
                    "url": f"http://localhost:{RERUN_WEB_PORT}/?url=ws://localhost:{RERUN_WS_PORT}",
                },
            }
            await self.send_response('start_replay', "success", response_data)

            try:
                replay(replay_cfg)

            finally:
                # 无论可视化是否正常结束，都通知replay线程停止
                stop_event.set()
                # 等待replay线程安全退出（设置合理超时）
                visual_thread.join(timeout=2.0)
                
                # 检查线程是否已退出
                if visual_thread.is_alive():
                    print("Warning: Visual thread did not exit cleanly")
                
                # 处理子线程异常
                try:
                    error = error_queue.get_nowait()
                    raise RuntimeError(f"Visual failed in thread: {str(error)}") from error
                except queue.Empty:
                    pass
            self.replaying = False
            print("="*20)
            print("Replay Complete Success!")
            print("="*20)
    
####################### Client Send to Server ############################
    async def send_heartbeat_loop(self):
        """定期发送心跳"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                try:
                    await self.sio.emit('HEARTBEAT')
                    self.last_heartbeat_time = current_time
                except Exception as e:
                    print(f"发送心跳失败: {e}")
            time.sleep(1)
            await self.sio.wait()


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