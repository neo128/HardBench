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
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_peripheral_read_write_params_t 
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
recv_lift_height = {}
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
                if 'depth' not in event_id:
                    img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)
                    frame = img_array.reshape((480, 640, 3))  # 已经是 RGB 格式 
                    if frame is not None:
                        with lock:
                            # print(f"Received event_id = {event_id}")
                            recv_images[event_id] = frame
                elif 'depth' in event_id: 
                    depth_array = np.frombuffer(buffer_bytes, dtype=np.uint16)
                    depth_frame = depth_array.reshape((480, 640))  # 已经是 RGB 格式
                    if depth_frame is not None:
                        # 排除无效值（可选，根据传感器特性，例如0或65535可能是无效深度）
                        valid_mask = (depth_frame > 0) & (depth_frame < 65535)
                        if np.any(valid_mask):
                            # 仅对有效区域归一化，避免无效值干扰
                            valid_depth = depth_frame[valid_mask]
                            min_val = valid_depth.min()
                            max_val = valid_depth.max()
                        else:
                            # 若全无效，强制范围为0~65535
                            min_val, max_val = 0, 65535

                        depth_16bit = cv2.normalize(
                        depth_frame,
                        None,
                        alpha=0,
                        beta=65535,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_16U  # 转换为16位无符号整数
                        )
                        rgb_depth = cv2.cvtColor(depth_16bit, cv2.COLOR_GRAY2RGB)  # 扩展维度到(480,640,3)以实现保存
                        # 存储归一化后的图像用于显示
                        with lock:
                            # print(f"Received event_id = {event_id}")
                            recv_images[event_id] = rgb_depth

            if 'jointstate' in event_id:
                joint_array = np.frombuffer(buffer_bytes, dtype=np.float64)
                if joint_array is not None:
                    with lock:
                        # print(f"Received event_id = {event_id}")
                        # print(f"Received {event_id}joint_array = {joint_array}")
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
            if 'lift_height' in event_id:
                lift_height = np.frombuffer(buffer_bytes, dtype=np.int64)
                if lift_height is not None:
                    with lock:
                        recv_lift_height[event_id] = lift_height
           

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
class RM75Arm:
    def __init__(self, ip, port, start_pos, joint_p_limit, joint_n_limit, fps, is_second):
        if is_second:
            self.arm =  RoboticArm()
        else:
            self.arm = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
        self.ip = ip
        self.port = int(port)
        self.start_pos = start_pos
        self.joint_p_limit = joint_p_limit
        self.joint_n_limit = joint_n_limit
        self.fps = fps
        self.lock = threading.Lock()  # 线程锁确保操作安全
        float_joint = ctypes.c_float*7
        self.joint_send=float_joint()

        self.handle = self.arm.rm_create_robot_arm(self.ip, self.port)
        if not self.handle:
            raise ConnectionError(f"无法连接到 {self.ip}:{self.port}")
    def movej_cmd(self, joint, speed=80):
        """
        关节空间运动
        arm_side: 'left' 或 'right' 指定手臂
        """
        with self.lock:
            # 选择对应的手臂
            result = self.arm.rm_movej(joint, speed, 0, 0, 0)
            if result != 0:
                print(f"ip为{self.ip}臂运动命令发送失败，错误码: {result}")
                return False
            return True
    
    def movej_canfd(self, joint):
        """通过CANFD发送关节运动命令"""
        with self.lock:
            result = self.arm.rm_movej_canfd(joint, False, 0) # 设置为低跟随模式
            if result != 0:
                print(f"ip为{self.ip}的臂CANFD运动命令失败，错误码: {result}")
                return False
            return True
    
    def set_gripper(self, value):
        """控制夹爪"""
        value = int(value)
        try:
            with self.lock:
                result = self.arm.rm_set_gripper_position(value, False, 1)
            if result != 0:
                print(f"ip为{self.ip}的臂夹爪控制失败，错误码: {result}")
                return False
            return True
        except Exception as e:
            print(f"ip为{self.ip}的臂夹爪控制异常: {e}")
            return False
    def set_lift(self, arm_side, value):
        """控制升降机"""

        value = int(value)

        try:
            with self.lock:
                result = self.arm.rm_set_lift_height(50, value, 0)
            if result != 0:
                print(f"升降机控制失败，错误码: {result}")
                return False
            return True
        except Exception as e:
            print(f"升降机控制异常: {e}")
            return False    
    def get_joint_dergree(self):
        """获取当前关节角度"""
        with self.lock:
            _, joint_pos = self.arm.rm_get_joint_degree()
        return np.array(joint_pos)
    
    def get_gripper_value(self, arm_side):
        """获取当前夹爪角度"""
        with self.lock:
            _, gripper_pos = self.arm.rm_get_gripper_state()
        return np.array(gripper_pos['actpos'])    
    def get_lift_height(self, arm_side):
        """获取当前升降机高度"""
        with self.lock:
            _num, lift_read = self.arm.rm_get_lift_state()
        return np.array(lift_read['pos'])    
    def stop(self):
        """停止双臂运动"""
        with self.lock:
            self.arm.rm_set_arm_stop()
        print("双臂已停止运动")
    
    def disconnect(self):
        """断开双臂连接"""
        try:
            with self.lock:
                self.arm.rm_close_modbus_mode(1)
            print("双臂已断开连接")
        except Exception as e:
            print(f"断开连接失败: {e}")
class RealmanManipulator:
    def __init__(self, config: RealmanRobotConfig):
        self.config = config
        self.robot_type = self.config.type

        self.follower_arms = {}
        self.leader_arms = {}
        self.leader_arms['right'] = self.config.right_leader_arm.motors
        self.leader_arms['left'] = self.config.left_leader_arm.motors
        self.follower_arms["left"] = RM75Arm(
            ip = self.config.left_arm_config['ip'], 
            port = self.config.left_arm_config['port'],
            start_pos = self.config.left_arm_config['start_pose'],
            joint_p_limit = self.config.left_arm_config['joint_p_limit'],
            joint_n_limit = self.config.left_arm_config['joint_n_limit'],
            fps = self.config.left_arm_config['fps'],
            is_second = False, 
        )
        self.follower_arms["right"] = RM75Arm(
            ip = self.config.right_arm_config['ip'], 
            port = self.config.left_arm_config['port'],
            start_pos = self.config.right_arm_config['start_pose'],
            joint_p_limit = self.config.right_arm_config['joint_p_limit'],
            joint_n_limit = self.config.right_arm_config['joint_n_limit'],
            fps = self.config.right_arm_config['fps'],
            is_second = True, 
        )
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.microphones = self.config.microphones

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
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        print(f"期望的状态名字数量{state_names}")
        return {
            "action": {
                "dtype": "float64",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float64",
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

                    byte_array = np.zeros(7, dtype=np.float64)
                    joint_read = recv_jointstats[match_name]

                    byte_array[:7] = joint_read[:7]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now
        

        follower_pos = {}
        for name in self.follower_arms:
            for match_name in recv_pose:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(7, dtype=np.float64)
                    pose_read = recv_pose[match_name]

                    byte_array[:7] = pose_read[:7]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_pos[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

        follower_gripper = {}
        for name in self.follower_arms:
            for match_name in recv_gripper:
                    now = time.perf_counter()

                    byte_array = np.zeros(1, dtype=np.float64)
                    gripper_read = recv_gripper[match_name]

                    byte_array[:1] = gripper_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_gripper[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_gripper_dt_s"] = time.perf_counter() - now

        #记录当前关节角度 为30维:(7+7+1)*2
        state = []
        for name in self.follower_arms:
            if name in follower_joint:
                state.append(follower_joint[name])
            if name in follower_pos:
                state.append(follower_pos[name])
            if name in follower_gripper:
                state.append(follower_gripper[name])
        # 单独添加升降机高度
        state.append(torch.from_numpy(recv_lift_height['lift_height'].copy()))
        
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
        # 单独添加升降机高度
        action.append(torch.from_numpy(recv_lift_height['lift_height'].copy()))
        
        action = torch.cat(action)
        images = {}
        for name in self.cameras:
            now = time.perf_counter()
            images[name] = recv_images[name]

            # images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy((images[name]).copy())
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


    def send_action(self, action: torch.Tensor):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )
        from_idx = 0
        to_idx = 7
        index = 0
        pos_num = 7
        # 0-7
        # 
        action_sent = []
        for name in self.follower_arms:
            if index != 1:
                goal_pos = action[index*8+from_idx:index*8+to_idx]
                gripper_pos = action[index*8+from_idx+pos_num]
                goal_pos = torch.cat((goal_pos, torch.tensor([gripper_pos])))
            else:
                goal_pos = action[index*8+pos_num:index*8+pos_num+to_idx]
                gripper_pos = action[index*8+pos_num+pos_num+to_idx]
                goal_pos = torch.cat((goal_pos, torch.tensor([gripper_pos])))
            index+=1
            for i in range(7):
                # joint_send[i] = max(self.follower_arms[name].joint_n_limit[i], min(self.follower_arms[name].joint_p_limit[i], goal_pos[i]))
                self.follower_arms[name].joint_send[i] = goal_pos[i]
            
            self.follower_arms[name].movej_canfd(self.follower_arms[name].joint_send)
            # if (goal_pos[7]<50):
            #     # ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, int(follower_goal_pos_array[7]), 1, 1)
            #     ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, 0 , 1, 1)
            #     self.gipflag_send=0
            # #状态为闭合，且需要张开夹爪
            # if (goal_pos[7]>=50):
            #     # ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, int(follower_goal_pos_array[7]), 1, 1)
            #     ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, 100, 1, 1)
            #     self.gipflag_send=1
            
            gripper_value = goal_pos[7]
            # 后续添加夹爪控制条件
            self.follower_arms[name].set_gripper(gripper_value)
            self.frame_counter += 1

            action_sent.append(goal_pos)

        return torch.cat(action_sent)
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