import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import ctypes
import platform
import sys

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from functools import cache

import threading
import cv2

import zmq


from operating_platform.robot.robots.utils import RobotDeviceNotConnectedError
from operating_platform.robot.robots.configs import RealmanRobotConfig
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig

from operating_platform.robot.robots.camera import Camera

# IPC Address
ipc_address = "ipc:///tmp/dora-zeromq"

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(ipc_address)
socket.setsockopt(zmq.RCVTIMEO, 300)  # 设置接收超时（毫秒）

running_server = True
recv_images = {}  # 缓存每个 event_id 的最新帧
recv_jointstats = {} 
recv_pose = {}
recv_gripper = {}
lock = threading.Lock()
def recv_server():
    """接收数据线程"""
    while running_server:
        try:

            message_parts = socket.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误
                
            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]
            if 'image' in event_id:
                # 解码图像
                img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)
                frame = img_array.reshape((480, 640, 3))  # 已经是 RGB 格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame is not None:
                    with lock:
                        # print(f"Received event_id = {event_id}")
                        recv_images[event_id] = frame

            if 'jointstate' in event_id:
                joint_array = np.frombuffer(buffer_bytes, dtype=np.float64)
                if joint_array is not None:
                    with lock:
                        # print(f"Received event_id = {event_id}")
                        # print(f"Received joint_array = {joint_array}")
                        recv_jointstats[event_id] = joint_array

            if 'pose' in event_id:
                pose_array = np.frombuffer(buffer_bytes, dtype=np.float64)
                if pose_array is not None:
                    with lock:
                        recv_pose[event_id] = pose_array

            if 'gripper' in event_id:
                gripper_array = np.frombuffer(buffer_bytes, dtype=np.float64)
                if gripper_array is not None:
                    with lock:
                        recv_gripper[event_id] = gripper_array

        except zmq.Again:
            # 接收超时，继续循环
            print(f"Received Timeout")
            continue
        except Exception as e:
            print("recv error:", e)
            break

class OpenCVCamera:
    def __init__(self, config: OpenCVCameraConfig):
        self.config = config
        self.camera_index = config.camera_index
        self.port = None

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height

        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> list[Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            cameras[key] = OpenCVCamera(cfg)
        else:
            raise ValueError(f"The camera type '{cfg.type}' is not valid.")

    return cameras

class RealmanManipulator:
    def __init__(self, config: RealmanRobotConfig):
        self.config = config
        self.robot_type = self.config.type
        # 需要修改
        self.follower_arms = {}
        self.follower_arms["left"] = self.config.left_leader_arm.motors
        self.follower_arms["right"] =self.config.right_leader_arm.motors

        self.cameras = make_cameras_from_configs(self.config.cameras)
        recv_thread = threading.Thread(target=recv_server,daemon=True)
        recv_thread.start()

        self.is_connected = False
        self.logs = {}
        self.frame_counter = 0  # 帧计数器

        
    def get_motor_names(self, arm: dict[str, dict]) -> list:
        return [f"{arm}_{motor}" for arm, motors in arm.items() for motor in motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft
    
    
    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.follower_arms)
        state_names = self.get_motor_names(self.follower_arms)
        print(f"期望的状态名字数量{state_names}")
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }
    
    def connect(self):
        timeout = 5  # 超时时间（秒）
        start_time = time.perf_counter()

        while True:
            # 检查是否已获取所有摄像头的图像
            if all(name in recv_images for name in self.cameras):
                break
            # 超时检测
            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("等待摄像头图像超时")

            # 可选：减少CPU占用
            time.sleep(0.01)
        
        start_time = time.perf_counter()
        while True:
            # 检查是否已获取所有机械臂的关节角度
            if any(
                any(name in key for key in recv_jointstats)
                for name in self.follower_arms
            ):
                break

            # 超时检测
            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("等待机械臂关节数据超时")

            # 可选：减少CPU占用
            time.sleep(0.01)

        start_time = time.perf_counter()
        while True:
            for key in recv_pose: 
                print(key)
            if any(
                any(name in key for key in recv_pose)
                for name in self.follower_arms
            ):
                break

            # 超时检测
            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("等待机械臂末端位姿超时")

            # 可选：减少CPU占用
            time.sleep(0.01)

        start_time = time.perf_counter()
        while True:
            if any(
                any(name in key for key in recv_gripper)
                for name in self.follower_arms
            ):
                break

            # 超时检测
            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("等待机械臂夹爪超时")

            # 可选：减少CPU占用
            time.sleep(0.01)

        self.is_connected = True

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}
    
    @property
    def has_camera(self):
        return len(self.cameras) > 0
    
    @property
    def num_cameras(self):
        return len(self.cameras)
    
    def teleop_step(
        self, record_data=False, 
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.frame_counter += 1

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Realman is not connected. You need to run `robot.connect()`."
            )
        if not record_data:
            return
        
        follower_joint = {}
        for name in self.follower_arms:
            for match_name in recv_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    joint_read = recv_jointstats[match_name]

                    byte_array[:6] = joint_read[:6]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now
        

        follower_pos = {}
        for name in self.follower_arms:
            for match_name in recv_pose:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    pose_read = recv_pose[match_name]

                    byte_array[:6] = pose_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_pos[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

        follower_gripper = {}
        for name in self.follower_arms:
            for match_name in recv_gripper:
                    now = time.perf_counter()

                    byte_array = np.zeros(1, dtype=np.float32)
                    gripper_read = recv_gripper[match_name]

                    byte_array[:1] = gripper_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_gripper[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_gripper_dt_s"] = time.perf_counter() - now

        
        #记录当前关节角度
        state = []
        for name in self.follower_arms:
            if name in follower_joint:
                state.append(follower_joint[name])
            if name in follower_pos:
                state.append(follower_pos[name])
            if name in follower_gripper:
                state.append(follower_gripper[name])
        state = torch.cat(state)

        #将关节目标位置添加到 action 列表中
        action = []
        for name in self.follower_arms:
            if name in follower_joint:
                action.append(follower_joint[name])
            if name in follower_pos:
                action.append(follower_pos[name])
            if name in follower_gripper:
                action.append(follower_gripper[name])
        action = torch.cat(action)

        images = {}
        for name in self.cameras:
            now = time.perf_counter()
            
            images[name] = recv_images[name]

            # images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - now

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        
        # print("end teleoperate record")

        return obs_dict, action_dict



    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Realman is not connected. You need to run `robot.connect()` before disconnecting."
            )
        self.is_connected = False
        running_server = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
