import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import ctypes
import platform
import sys
import zmq

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from functools import cache
from typing import Any

import threading
import cv2


from operating_platform.robot.robots.utils import RobotDeviceNotConnectedError
from operating_platform.robot.robots.configs import AlohaRobotConfig
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig

from operating_platform.robot.robots.camera import Camera


recv_images = {}
recv_master_jointstats = {}
recv_follower_jointstats = {}
recv_follower_pose = {}
lock = threading.Lock()  # 线程锁

# IPC Address
ipc_address = "ipc:///tmp/dora-zeromq"
ipc_address_piper = "ipc:///tmp/dorobot-piper"

context = zmq.Context()
running_recv_image_server = True
running_recv_piper_server = True

socket_image = context.socket(zmq.PAIR)
socket_image.connect(ipc_address)
socket_image.setsockopt(zmq.SNDHWM, 2000)
socket_image.setsockopt(zmq.SNDBUF, 2**25)
socket_image.setsockopt(zmq.SNDTIMEO, 2000)
socket_image.setsockopt(zmq.RCVTIMEO, 2000)  # 设置接收超时（毫秒）
socket_image.setsockopt(zmq.LINGER, 0)

socket_piper = context.socket(zmq.PAIR)
socket_piper.connect(ipc_address_piper)
socket_piper.setsockopt(zmq.SNDHWM, 2000)
socket_piper.setsockopt(zmq.SNDBUF, 2**25)
socket_piper.setsockopt(zmq.SNDTIMEO, 2000)
socket_piper.setsockopt(zmq.RCVTIMEO, 2000)  # 设置接收超时（毫秒）
socket_piper.setsockopt(zmq.LINGER, 0)

def piper_zmq_send(event_id, buffer, wait_time_s):
    buffer_bytes = buffer.tobytes()
    print(f"zmq send event_id:{event_id}, value:{buffer}")
    try:
        socket_piper.send_multipart([
            event_id.encode('utf-8'),
            buffer_bytes
        ], flags=zmq.NOBLOCK)
    except zmq.Again:
        pass
    time.sleep(wait_time_s)

def recv_img_server():
    """接收数据线程"""
    while running_recv_image_server:
        try:
            message_parts = socket_image.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]

            if 'image' in event_id:
                # 解码图像
                img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame is not None:
                    with lock:
                        recv_images[event_id] = frame

        except zmq.Again:
            print(f"Manipulator Receive Image Timeout")
            continue
        except Exception as e:
            print("Manipulator Receive Image Recv Error:", e)
            break

def recv_piper_server():
    """接收数据线程"""
    while running_recv_piper_server:
        try:
            message_parts = socket_piper.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]

            if 'jointstat' in event_id and 'master' in event_id:
                joint_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if joint_array is not None:
                    with lock:
                        recv_master_jointstats[event_id] = joint_array

            if 'jointstat' in event_id and 'follower' in event_id:
                joint_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if joint_array is not None:
                    with lock:
                        recv_follower_jointstats[event_id] = joint_array

            if 'endpose' in event_id and 'follower' in event_id:
                pose_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if pose_array is not None:
                    with lock:
                        recv_follower_pose[event_id] = pose_array

        except zmq.Again:
            print(f"Manipulator Receive Piper Timeout")
            continue
        except Exception as e:
            print("Manipulator Receive Piper Recv Error:", e)
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



class AlohaManipulator:
    def __init__(self, config: AlohaRobotConfig):
        self.config = config
        self.robot_type = self.config.type

        self.use_videos = self.config.use_videos

        self.follower_arms = {}
        self.follower_arms['right'] = self.config.right_leader_arm.motors
        self.follower_arms['left'] = self.config.left_leader_arm.motors

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.microphones = self.config.microphones

        self.recv_img_thread = threading.Thread(target=recv_img_server, daemon=True)
        self.recv_img_thread.start()

        self.recv_piper_thread = threading.Thread(target=recv_piper_server, daemon=True)
        self.recv_piper_thread.start()
        
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
                any(name in key for key in recv_follower_jointstats)
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
            if any(
                any(name in key for key in recv_follower_pose)
                for name in self.follower_arms
            ):
                break

            # 超时检测
            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("等待机械臂末端位姿超时")

            # 可选：减少CPU占用
            time.sleep(0.01)

        # start_time = time.perf_counter()
        # while True:
        #     if any(
        #         any(name in key for key in recv_follower_gripper)
        #         for name in self.follower_arms
        #     ):
        #         break

        #     # 超时检测
        #     if time.perf_counter() - start_time > timeout:
        #         raise TimeoutError("等待机械臂夹爪超时")

        #     # 可选：减少CPU占用
        #     time.sleep(0.01)

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
    
    def follower_record(self):
        follower_joint = {}
        for name in self.follower_arms:
            for match_name in recv_follower_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    joint_read = recv_follower_jointstats[match_name]

                    byte_array[:6] = joint_read[:6]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now

        follower_pos = {}
        for name in self.follower_arms:
            for match_name in recv_follower_pose:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    pose_read = recv_follower_pose[match_name]

                    byte_array[:6] = pose_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_pos[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

        follower_gripper = {}
        for name in self.follower_arms:
            for match_name in recv_follower_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(1, dtype=np.float32)
                    gripper_read = recv_follower_jointstats[match_name][6]

                    byte_array[0] = gripper_read
                    byte_array = np.round(byte_array, 3)
                    
                    follower_gripper[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_gripper_dt_s"] = time.perf_counter() - now
        
        return follower_joint, follower_pos, follower_gripper
    
    def master_record(self):
        master_joint = {}
        for name in self.follower_arms:
            for match_name in recv_master_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    joint_read = recv_master_jointstats[match_name]

                    byte_array[:6] = joint_read[:6]
                    byte_array = np.round(byte_array, 3)
                    
                    master_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_master_{name}_joint_dt_s"] = time.perf_counter() - now

        # master_pos = {}
        # for name in self.follower_arms:
        #     for match_name in recv_master_pose:
        #         if name in match_name:
        #             now = time.perf_counter()

        #             byte_array = np.zeros(6, dtype=np.float32)
        #             pose_read = recv_master_pose[match_name]

        #             byte_array[:6] = pose_read[:]
        #             byte_array = np.round(byte_array, 3)
                    
        #             master_pos[name] = torch.from_numpy(byte_array)

        #             self.logs[f"read_master_{name}_pos_dt_s"] = time.perf_counter() - now

        master_gripper = {}
        for name in self.follower_arms:
            for match_name in recv_master_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(1, dtype=np.float32)
                    gripper_read = recv_master_jointstats[match_name][6]

                    byte_array[0] = gripper_read
                    byte_array = np.round(byte_array, 3)
                    
                    master_gripper[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_master_{name}_gripper_dt_s"] = time.perf_counter() - now
        
        return master_joint, master_gripper

    def teleop_step(
        self, record_data=False, 
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.frame_counter += 1

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()`."
            )

        if not record_data:
            return

        follower_joint, follower_pos, follower_gripper = self.follower_record()

        # master_joint, master_pos, master_gripper = self.master_record()
        master_joint, master_gripper = self.master_record()

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
            if name in master_joint:
                action.append(master_joint[name])
            if name in follower_pos:
                action.append(follower_pos[name])
            if name in master_gripper:
                action.append(master_gripper[name])
        action = torch.cat(action)

        # Capture images from cameras
        
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



    # def capture_observation(self):
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             "KochRobot is not connected. You need to run `robot.connect()`."
    #         )

    #     #调用从臂api获取当前关节角度 
    #     for name in self.leader_arms:
    #         now = time.perf_counter()
    #         self.pDll.Get_Joint_Degree(self.nSocket,self.joint_obs_read)  
    #         #夹爪通信获取当前夹爪开合度
    #         #   giper_read=ctypes.c_int()
    #         #   self.pDll.Get_Read_Holding_Registers(self.nSocket,1,40005,1,ctypes.byref(giper_read))
    #         #   #八位数组存储关节和夹爪数据
    #         self.joint_obs_present[:7]=self.joint_obs_read[:]
    #         #   self.joint_obs_present[7]=giper_read.value
    #         if self.gipflag_send==1:
    #             self.joint_obs_present[7]=100
    #         elif self.gipflag_send==0:
    #             self.joint_obs_present[7]=10
    #         # self.joint_obs_present = np.zeros(8)  # 创建一个包含八个0的 NumPy 数组
    #         self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

    #     # Create state by concatenating follower current position
    #     #上传当前机械臂状态
    #     state = []
    #     self.joint_obs_present = np.round(self.joint_obs_present, 2)
    #     joint_array_np = np.array( self.joint_obs_present)
    #     state = np.array([joint_array_np], dtype=np.float32)
    #     state = np.concatenate(state, dtype=np.float32)

    #     # Capture images from cameras
    #     images = {}
    #     for name in self.cameras:
    #         now = time.perf_counter()
    #         images[name] = self.cameras[name].async_read()
    #         self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
    #         self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - now

    #     # Populate output dictionnaries and format to pytorch
    #     obs_dict = {}
    #     obs_dict["observation.state"] = torch.from_numpy(state)
    #     for name in self.cameras:
    #         # Convert to pytorch format: channel first and float32 in [0,1]
    #         img = torch.from_numpy(images[name])
    #         img = img.type(torch.float32) / 255
    #         img = img.permute(2, 0, 1).contiguous()
    #         obs_dict[f"observation.images.{name}"] = img
    #     return obs_dict    def capture_observation(self):

    # def capture_observation(self):
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             "KochRobot is not connected. You need to run `robot.connect()`."
    #         )

    #     follower_pos = {}
    #     for name in self.follower_arms:
    #         now = time.perf_counter()
    #         eight_byte_array = np.zeros(8, dtype=np.float32)
    #         joint_obs_read = self.follower_arms[name].async_read_joint_degree()

    #         #夹爪通信获取当前夹爪开合度
    #         # giper_read=ctypes.c_int()
    #         # self.pDll.Get_Read_Holding_Registers(self.nSocket,1,40000,1,ctypes.byref(giper_read))
    #         #   #八位数组存储关节和夹爪数据
    #         eight_byte_array[:7] = joint_obs_read[:]
    #         # self.joint_obs_present[7]=giper_read.value
    #         eight_byte_array[7] = self.follower_arms[name].old_grasp
    #         # self.joint_obs_present = np.zeros(8)  # 创建一个包含八个0的 NumPy 数组
    #         eight_byte_array = np.round(eight_byte_array, 2)
    #         follower_pos[name] = torch.from_numpy(eight_byte_array)
    #         self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

    #     # Create state by concatenating follower current position
    #     #上传当前机械臂状态
    #     state = []
    #     for name in self.follower_arms:
    #         if name in follower_pos:
    #             state.append(follower_pos[name])    
    #     state = torch.cat(state)

    #     # Capture images from cameras
    #     images = {}
    #     for name in self.cameras:
    #         now = time.perf_counter()
    #         images[name] = self.cameras[name].async_read()
    #         images[name] = torch.from_numpy(images[name])
    #         self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
    #         self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - now

    #     # Populate output dictionnaries and format to pytorch
    #     obs_dict = {}
    #     obs_dict["observation.state"] = state
    #     for name in self.cameras:
    #         obs_dict[f"observation.images.{name}"] = images[name]
    #     return obs_dict




    def send_action(self, action: dict[str, Any]):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        # from_idx = 0
        # to_idx = 6
        # arm_action_dim = 13
        # arm_index = 0
        # action_sent = []
        for name in self.follower_arms:
            goal_joint = [ val for key, val in action.items() if name in key and "joint" in key]
            goal_gripper = [ val for key, val in action.items() if name in key and "gripper" in key]

            # goal_joint = action[(arm_index*arm_action_dim+from_idx):(arm_index*arm_action_dim+to_idx)]
            # goal_gripper = action[arm_index*arm_action_dim + 12]
            # arm_index += 1
            goal_joint_numpy = np.array([t.item() for t in goal_joint], dtype=np.float32)
            goal_gripper_numpy = np.array([t.item() for t in goal_gripper], dtype=np.float32)
            position = np.concatenate([goal_joint_numpy, goal_gripper_numpy], axis=0)

            piper_zmq_send(f"action_joint_{name}", position, wait_time_s=0.01)
            # piper_zmq_send(f"action_gripper_{name}", goal_gripper_numpy, wait_time_s=0.01)

            # action_sent.append(goal_joint)

        # return torch.cat(action_sent)

        # from_idx = 0
        # to_idx = 8
        # index = 0
        # action_sent = []
        # for name in self.follower_arms:

        #     goal_pos = action[index*8+from_idx:index*8+to_idx]
        #     index+=1

        #     for i in range(7):
        #         self.follower_arms[name].joint_send[i] = max(self.follower_arms[name].joint_n_limit[i], min(self.follower_arms[name].joint_p_limit[i], goal_pos[i]))
            
            
        #     self.follower_arms[name].movej_canfd(self.follower_arms[name].joint_send)
        #     # if (goal_pos[7]<50):
        #     #     # ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, int(follower_goal_pos_array[7]), 1, 1)
        #     #     ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, 0 , 1, 1)
        #     #     self.gipflag_send=0
        #     # #状态为闭合，且需要张开夹爪
        #     # if (goal_pos[7]>=50):
        #     #     # ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, int(follower_goal_pos_array[7]), 1, 1)
        #     #     ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, 100, 1, 1)
        #     #     self.gipflag_send=1
        #     gripper_value = max(0, min(100, int(goal_pos[7])))

        #     self.frame_counter += 1

        #     self.follower_arms[name].old_grasp = gripper_value
        #     if self.frame_counter % 5 == 0:
        #         self.frame_counter = 0
        #         self.follower_arms[name].write_single_register(gripper_value)
        #     action_sent.append(goal_pos)

        # return torch.cat(action_sent)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )
        

        self.is_connected = False
        
        global running_recv_image_server
        global running_recv_piper_server

        running_recv_image_server = False
        running_recv_piper_server = False

        self.recv_img_thread.join()
        self.recv_piper_thread.join()

        socket_image.close()
        socket_piper.close()

        context.term()
        

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
