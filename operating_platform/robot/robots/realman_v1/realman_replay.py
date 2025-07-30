import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from draccus import wrap
from typing import Union, List
import math

# 机械臂控制库
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_peripheral_read_write_params_t

# --------------------------
# 机械臂配置参数
# --------------------------
fps = 20  # 帧率设置
left_ip = "169.254.128.18"
right_ip = "169.254.128.19"
port = 8080

# 关节限位配置
joint_p_limit_str = "90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 180.0"  # 包含第七关节
joint_n_limit_str = "-90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -180.0"

# 转换为列表
joint_p_limit = [float(x) for x in joint_p_limit_str.split(',')]
joint_n_limit = [float(x) for x in joint_n_limit_str.split(',')]

# --------------------------
# 机械臂控制类
# --------------------------
class RealmanDualArm:
    def __init__(self, left_ip, right_ip, port, joint_p_limit, joint_n_limit):  
        """初始化双臂机械臂连接"""
        # 初始化左右臂
        self.arm_left = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
        self.arm_right = RoboticArm()
        
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.port = port
        self.joint_p_limit = joint_p_limit
        self.joint_n_limit = joint_n_limit
        self.is_running = True  # 运行状态标志
        self.lock = threading.Lock()  # 线程锁确保操作安全

        # 连接左臂
        self.handle_left = self.arm_left.rm_create_robot_arm(self.left_ip, self.port)
        if not self.handle_left:
            raise ConnectionError(f"左臂无法连接到 {self.left_ip}:{self.port}")
        print(f"左臂已连接: {self.left_ip}:{self.port}，句柄: {self.handle_left.id}")
        
        # 连接右臂
        self.handle_right = self.arm_right.rm_create_robot_arm(self.right_ip, self.port)
        if not self.handle_right:
            raise ConnectionError(f"右臂无法连接到 {self.right_ip}:{self.port}")
        print(f"右臂已连接: {self.right_ip}:{self.port}，句柄: {self.handle_right.id}")
        
        # 初始化末端执行器
        self._init_end_effectors()
        self.is_connected = True

    def _init_end_effectors(self):
        """初始化左右臂末端执行器（夹爪）"""
        # 左臂夹爪初始化
        try:
            # self.arm_left.rm_set_tool_voltage(24)
            # self.arm_left.rm_set_modbus_mode(1, 115200, 5)
            # self.peripheral_left = rm_peripheral_read_write_params_t(1, 40000, 1, 1)
            # self.arm_left.rm_write_single_register(self.peripheral_left, 100)
            print("左臂末端执行器初始化完成")
        except Exception as e:
            print(f"左臂末端执行器初始化失败: {e}")
        
        # 右臂夹爪初始化
        try:
            # self.arm_right.rm_set_tool_voltage(24)
            # self.arm_right.rm_set_modbus_mode(1, 115200, 5)
            # self.peripheral_right = rm_peripheral_read_write_params_t(1, 40000, 1, 1)
            # self.arm_right.rm_write_single_register(self.peripheral_right, 100)
            print("右臂末端执行器初始化完成")
        except Exception as e:
            print(f"右臂末端执行器初始化失败: {e}")
    
    def movej_cmd(self, arm_side, joint, speed=30):
        """
        关节空间运动
        arm_side: 'left' 或 'right' 指定手臂
        """
        with self.lock:
            # 选择对应的手臂
            arm = self.arm_left if arm_side == 'left' else self.arm_right
            
            
            result = arm.rm_movej(joint, speed, 0, 0, 0)
            if result != 0:
                print(f"{arm_side}臂运动命令发送失败，错误码: {result}")
                return False
            return True
    
    def movej_canfd(self, arm_side, joint):
        """通过CANFD发送关节运动命令"""
        arm = self.arm_left if arm_side == 'left' else self.arm_right
        
        # 关节角度限位检查

        
        with self.lock:
            result = arm.rm_movej_canfd(joint, True, 0)
            if result != 0:
                print(f"{arm_side}臂CANFD运动命令失败，错误码: {result}")
                return False
            return True
    
    def set_gripper(self, arm_side, value):
        """控制夹爪"""
        value = int(value)
        try:
            with self.lock:
                if arm_side == 'left':
                    result = self.arm_left.rm_set_gripper_position(value, 0, 3)
                else:
                    result = self.arm_right.rm_set_gripper_position(value, 0, 3)
                # print(f"返回的状态码为{result}, 夹爪的控制量为{value}")
            if result != 0:
                print(f"{arm_side}臂夹爪控制失败，错误码: {result}")
                return False
            return True
        except Exception as e:
            print(f"{arm_side}臂夹爪控制异常: {e}")
            return False
    
    def get_joint_dergree(self, arm_side):
        """获取当前关节角度"""
        with self.lock:
            if arm_side == 'left':
                _, joint_pos = self.arm_left.rm_get_joint_degree()
            else:
                _, joint_pos = self.arm_right.rm_get_joint_degree()
        return np.array(joint_pos)
    
    def get_gripper_value(self, arm_side):
        """获取当前夹爪角度"""
        with self.lock:
            if arm_side == 'left':
                _, gripper_pos = self.arm_left.rm_get_gripper_state()
            else:
                _, gripper_pos = self.arm_right.rm_get_gripper_state()
        return np.array(gripper_pos['actpos'])    
    
    def stop(self):
        """停止双臂运动"""
        with self.lock:
            self.arm_left.rm_set_arm_stop()
            self.arm_right.rm_set_arm_stop()
        print("双臂已停止运动")
    
    def disconnect(self):
        """断开双臂连接"""
        try:
            with self.lock:
                self.arm_left.rm_close_modbus_mode(1)
                self.arm_right.rm_close_modbus_mode(1)
            self.is_connected = False
            print("双臂已断开连接")
        except Exception as e:
            print(f"断开连接失败: {e}")


# --------------------------
# 工具函数
# --------------------------
def degrees_to_radians(degrees):  
    """将度转换为弧度"""  
    return [degree * (math.pi / 180) for degree in degrees]


# --------------------------
# 控制线程函数
# --------------------------
def arm_control_thread(
    dual_arm: RealmanDualArm, 
    action_data: List, 
    replay_speed: float, 
    use_canfd: bool, 
    debug: bool
):
    """单线程控制双臂运动"""
    try:
        start_time = time.time()
        print("开始重放双臂动作...")
        
        for idx, action_value in enumerate(action_data):
            if not dual_arm.is_running:
                break
                
            try:
                # 解析动作数据（根据用户定义的格式）
                # left: joint 0-6, pose:7-12, gripper:13
                # right: joint:14-20, pose:21-26, gripper:27
                joint_left = action_value[0:7]
                gripper_left = float(action_value[13])
                joint_right = action_value[14:21]
                gripper_right = float(action_value[27])

                # 调试信息
                if debug and idx % 10 == 0:
                    print(f"第 {idx} 帧 - 左臂关节: {joint_left[:3]}... 右臂关节: {joint_right[:3]}...")

                # 发送运动命令
                if use_canfd:
                    dual_arm.movej_canfd('left', joint_left)
                    dual_arm.movej_canfd('right', joint_right)
                else:
                    dual_arm.movej_cmd('left', joint_left)
                    dual_arm.movej_cmd('right', joint_right)

                # 计算并打印误差（仅当获取到当前位置时）
                try:
                    current_left = dual_arm.get_joint_dergree('left')
                    current_right = dual_arm.get_joint_dergree('right')
                    current_left_gripper = dual_arm.get_gripper_value('left')
                    current_right_gripper = dual_arm.get_gripper_value('right')
                    left_error = np.abs(np.array(joint_left) - current_left)
                    right_error = np.abs(np.array(joint_right) - current_right)
                    gripper_left_error = np.abs(gripper_left - current_left_gripper)
                    gripper_right_error = np.abs(gripper_right - current_right_gripper)

                    print(f"第 {idx} 帧 - 左臂误差: {left_error.mean():.2f}° 右臂误差: {right_error.mean():.2f}° 左夹爪误差：{gripper_left_error} 右夹爪误差：{gripper_right_error}")
                except Exception as e:
                    print(f"计算误差失败: {e}")

                # 控制夹爪
                dual_arm.set_gripper('left', gripper_left)
                dual_arm.set_gripper('right', gripper_right)

                # 第一帧等待3秒，确保机械臂准备就绪
                if idx == 0:
                    time.sleep(3)

                # 控制重放速度
                expected_time = idx / (fps * replay_speed)
                elapsed_time = time.time() - start_time
                if elapsed_time < expected_time:
                    time.sleep(expected_time - elapsed_time)

            except Exception as e:
                print(f"处理第 {idx} 帧时出错: {e}")
                user_input = input("是否继续? (y/n) ").lower()
                if user_input != 'y':
                    dual_arm.is_running = False
                    break

        print("重放完成!")

    except Exception as e:
        print(f"线程异常: {e}")
    finally:
        dual_arm.stop()
        dual_arm.disconnect()


# --------------------------
# 配置类定义
# --------------------------
@dataclass
class ParquetConfig:
    path: Union[str, Path]  # 本地 Parquet 文件路径
    episode: Union[int, None] = None  # 筛选特定 episode
    max_frames: Union[int, None] = None  # 限制最大帧数

@dataclass
class ReplayConfig:
    parquet: ParquetConfig  # Parquet 读取配置
    debug: bool = False  # 调试日志开关
    replay_speed: float = 1.0  # 重放速度倍率
    use_canfd: bool = False  # 是否使用CANFD通信


# --------------------------
# 核心重放逻辑
# --------------------------
@wrap()
def replay_realman_arm(cfg: ReplayConfig):
    """双臂重放控制主函数"""
    logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO)
    logging.info("重放配置:\n" + pformat(cfg))

    dual_arm = None
    control_thread = None

    try:
        # 1. 读取Parquet文件
        parquet_path = Path(cfg.parquet.path).resolve()
        action_data = load_parquet(parquet_path, cfg.parquet)
        action_list = [row for row in action_data["action"]]
        print(f"成功加载动作数据: {len(action_list)} 帧")

        # 2. 初始化双臂机械臂
        dual_arm = RealmanDualArm(
            left_ip=left_ip,
            right_ip=right_ip,
            port=port,
            joint_p_limit=joint_p_limit,
            joint_n_limit=joint_n_limit
        )

        # 3. 创建并启动控制线程
        control_thread = threading.Thread(
            target=arm_control_thread,
            args=(dual_arm, action_list, cfg.replay_speed, cfg.use_canfd, cfg.debug),
            name="DualArmControlThread"
        )

        print("启动双臂控制线程...")
        control_thread.start()
        control_thread.join()

    except Exception as e:
        print(f"主程序错误: {e}")
        if dual_arm:
            dual_arm.is_running = False
    finally:
        # 确保线程正确关闭
        if control_thread and control_thread.is_alive():
            control_thread.join(timeout=5)
        print("所有重放进程已结束")


# --------------------------
# Parquet 读取工具函数
# --------------------------
def load_parquet(parquet_path: Path, parquet_cfg: ParquetConfig):
    """读取并处理Parquet文件"""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet文件不存在: {parquet_path}")

    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        logging.info(f"加载Parquet文件: {parquet_path} (共 {len(df)} 帧)")
    except Exception as e:
        logging.warning(f"pyarrow读取失败，尝试pandas: {e}")
        df = pd.read_parquet(parquet_path)

    # 筛选特定episode
    if parquet_cfg.episode is not None:
        if "episode" not in df.columns:
            raise ValueError("Parquet文件中没有'episode'列，无法筛选")
        df = df[df["episode"] == parquet_cfg.episode].copy()
        if len(df) == 0:
            raise ValueError(f"未找到episode={parquet_cfg.episode}的数据")
        logging.info(f"筛选后剩余 {len(df)} 帧 (episode={parquet_cfg.episode})")

    # 限制最大帧数
    if parquet_cfg.max_frames and parquet_cfg.max_frames < len(df):
        df = df.iloc[:parquet_cfg.max_frames]
        logging.info(f"限制最大帧数为: {parquet_cfg.max_frames}")

    return df


# --------------------------
# 入口函数
# --------------------------
if __name__ == "__main__":
    replay_realman_arm()
# python realman_replay.py --parquet.path "/home/rm/DoRobot/dataset/20250729/user/倒水_Pour the waterCopy0725testCopy_226/data/chunk-000/episode_000000.parquet"