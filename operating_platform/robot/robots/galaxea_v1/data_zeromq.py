#!/usr/bin/python3
import rclpy
from rclpy.node import Node as ROS2Node
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Float32MultiArray
import zmq
import threading
import numpy as np
import time
import json
from collections import defaultdict

# IPC Address
ipc_address_joint = "ipc:///tmp/ros2-zeromq-galaxea-joint"
ipc_address_image = "ipc:///tmp/dora-zeromq-galaxea-image"

class ROS2BridgeNode(ROS2Node):
    def __init__(self):
        super().__init__('ros2_zeromq_bridge')
        
        # ROS 2 订阅与发布
        self.subscription_arm_left = self.create_subscription(
            JointState,
            '/hdas/feedback_arm_left',
            self.ros2_to_zeromq_arm_left_callback,
            10)
        
        self.subscription_arm_right = self.create_subscription(
            JointState,
            '/hdas/feedback_arm_right',
            self.ros2_to_zeromq_arm_right_callback,
            10)
        
        self.subscription_image_top_left = self.create_subscription(
            CompressedImage,
            '/hdas/camera_head/left_raw/image_raw_color/compressed',
            self.ros2_to_zeromq_images_top_left_callback,
            10)
        
        self.subscription_image_top_right = self.create_subscription(
            CompressedImage,
            '/hdas/camera_head/right_raw/image_raw_color/compressed',
            self.ros2_to_zeromq_images_top_right_callback,
            10)
        
        self.subscription_image_wrist_left = self.create_subscription(
            CompressedImage,
            '/hdas/camera_wrist_left/color/image_raw/compressed',
            self.ros2_to_zeromq_images_wrist_left_callback,
            10)
        
        self.subscription_image_wrist_right = self.create_subscription(
            CompressedImage,
            '/hdas/camera_wrist_right/color/image_raw/compressed',
            self.ros2_to_zeromq_images_wrist_right_callback,
            10)
        self.publisher_arm_left = self.create_publisher(
            JointState,
            '/motion_target/target_joint_state_arm_left',
            10)
        self.publisher_gripper_left = self.create_publisher(
            JointState,
            '/motion_target/target_position_gripper_left',
            10)
        
        # ZeroMQ 初始化
        self.galaxea_context = zmq.Context()
        self.socket_joint = self.galaxea_context.socket(zmq.PAIR)
        self.socket_joint.bind(ipc_address_joint)
        self.socket_joint.setsockopt(zmq.SNDHWM, 2000)
        self.socket_joint.setsockopt(zmq.SNDBUF, 2**25)
        self.socket_joint.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket_joint.setsockopt(zmq.LINGER, 0)
        
        self.socket_image = self.galaxea_context.socket(zmq.PAIR)
        self.socket_image.bind(ipc_address_image)
        self.socket_image.setsockopt(zmq.SNDHWM, 2000)
        self.socket_image.setsockopt(zmq.SNDBUF, 2**25)
        self.socket_image.setsockopt(zmq.SNDTIMEO, 2000)
        self.socket_image.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket_image.setsockopt(zmq.LINGER, 0)
        # 运行标志
        self.running = True
        
        # 启动 ZeroMQ 接收线程
        self.zeromq_thread = threading.Thread(target=self.zeromq_receive_loop)
        self.zeromq_thread.daemon = True
        self.zeromq_thread.start()

        # 缓存左右臂的数据（字典结构：{timestamp: {"left": pos+vel, "right": pos+vel, "timestamp_ns": arrival_time}}）
        self.arm_data_cache = defaultdict(dict)
        self.MAX_CACHE_SIZE = 1000
        self.MAX_DATA_AGE_NS = 10e9  # 100ms超时
        self.timestamp_tolerance_ns = timestamp_tolerance_ns = 10_000_000 # 可配置的时间戳容差

        # 添加锁用于线程同步
        self._cache_lock = threading.Lock()

         # 频率控制相关变量

        self.last_process_time_left = 0
        self.last_process_time_right = 0
        self.min_interval_ns = 1e9 / 1  # 30Hz 对应的最小间隔时间(纳秒)

    # def ros2_to_zeromq_arm_left_callback(self, msg):
    #     """ROS 2 -> ZeroMQ"""
    #     try:
    #         data = {
    #             "timestamp": {
    #                 "sec": msg.header.stamp.sec,
    #                 "nsec": msg.header.stamp.nanosec  # 可选：更高精度
    #             },
    #         }
    #         # 序列化为 JSON 字符串并编码为 bytes
    #         #json_data = json.dumps(data).encode('utf-8')
    #         position =  msg.position 
    #         velocity = msg.velocity  
    #         # 转换为 float32 的 NumPy 数组
    #         pos_array = np.array(position, dtype=np.float32)  # shape=(N,)
    #         vel_array = np.array(velocity, dtype=np.float32)   # shape=(N,)
    #         combined_array = np.concatenate((pos_array, vel_array))  # shape=(2N,)

    #         self.socket_joint.send_multipart([
    #             b"state_joint_arm_left",
    #             combined_array.tobytes()
    #         ], flags=zmq.NOBLOCK)
    #     except Exception as e:
    #         self.get_logger().error(f"ROS2->ZeroMQ error: {e}")

    # def ros2_to_zeromq_arm_right_callback(self, msg):
    #     """ROS 2 -> ZeroMQ"""
    #     try:
    #         data = {
    #             "timestamp": {
    #                 "sec": msg.header.stamp.sec,
    #                 "nsec": msg.header.stamp.nanosec  # 可选：更高精度
    #             },
    #         }
    #         # 序列化为 JSON 字符串并编码为 bytes
    #         #json_data = json.dumps(data).encode('utf-8')
    #         position =  msg.position 
    #         velocity = msg.velocity  
    #         # 转换为 float32 的 NumPy 数组
    #         pos_array = np.array(position, dtype=np.float32)  # shape=(N,)
    #         vel_array = np.array(velocity, dtype=np.float32)   # shape=(N,)
    #         combined_array = np.concatenate((pos_array, vel_array))  # shape=(2N,)
    #         self.socket_joint.send_multipart([
    #             b"state_joint_arm_right",
    #             combined_array.tobytes()
    #         ], flags=zmq.NOBLOCK)
    #     except Exception as e:
    #         self.get_logger().error(f"ROS2->ZeroMQ error: {e}")

    def _convert_to_ns(self, sec, nanosec):
        """将ROS时间戳转换为纳秒级整数"""
        return sec * 1_000_000_000 + nanosec
 
    def ros2_to_zeromq_arm_left_callback(self, msg):
        current_time_ns = time.time_ns()
        if (current_time_ns - self.last_process_time_left) < self.min_interval_ns:
            return  # 跳过处理，保持30Hz频率
        self.last_process_time_left = current_time_ns
        try:
            # 转换为纳秒级时间戳（作为缓存键）
            timestamp_ns = self._convert_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            arrival_time_ns = time.time_ns()
            
            # 存储数据
            position = np.array(msg.position, dtype=np.float32)
            velocity = np.array(msg.velocity, dtype=np.float32)
            with self._cache_lock:
                self.arm_data_cache[timestamp_ns]["left"] = np.concatenate((position, velocity))
                self.arm_data_cache[timestamp_ns]["timestamp_ns"] = arrival_time_ns
                # 尝试匹配发送
                self._try_send_and_cleanup(timestamp_ns)
        except Exception as e:
            self.get_logger().error(f"Left arm ROS2->ZeroMQ error: {e}")
 
    def ros2_to_zeromq_arm_right_callback(self, msg):
        current_time_ns = time.time_ns()
        if (current_time_ns - self.last_process_time_right) < self.min_interval_ns:
            return  # 跳过处理，保持30Hz频率
        self.last_process_time_right = current_time_ns
        try:
            timestamp_ns = self._convert_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            arrival_time_ns = time.time_ns()
            
            position = np.array(msg.position, dtype=np.float32)
            velocity = np.array(msg.velocity, dtype=np.float32)
            with self._cache_lock:
                self.arm_data_cache[timestamp_ns]["right"] = np.concatenate((position, velocity))
                self.arm_data_cache[timestamp_ns]["timestamp_ns"] = arrival_time_ns
                self._try_send_and_cleanup(timestamp_ns)
            
        except Exception as e:
            self.get_logger().error(f"Right arm ROS2->ZeroMQ error: {e}")
 
    def _try_send_and_cleanup(self, current_timestamp_ns):
        try:
            # 1. 查找匹配的时间戳（在容差范围内）
            matched_timestamp = None
            left_matched_timestamp = None
            right_matched_timestamp = None
            for cached_timestamp in list(self.arm_data_cache.keys()):
                if abs(cached_timestamp - current_timestamp_ns) <= self.timestamp_tolerance_ns:
                    matched_timestamp = cached_timestamp
                    if "left" in self.arm_data_cache[matched_timestamp]:
                        left_matched_timestamp = matched_timestamp
                    else:
                        right_matched_timestamp = matched_timestamp
                    if left_matched_timestamp and right_matched_timestamp:
                        print(left_matched_timestamp,right_matched_timestamp)
                        break
 
            # 2. 如果左右臂数据均已到达，则发送
            if left_matched_timestamp and right_matched_timestamp:
                self._send_matched_data(left_matched_timestamp,right_matched_timestamp)
            
            # 3. 清理超时数据
            self._cleanup_expired_data()
            
            # 4. 限制缓存大小
            if len(self.arm_data_cache) > self.MAX_CACHE_SIZE:
                self.arm_data_cache.popitem(last=False)
 
        except Exception as e:
            self.get_logger().error(f"Send/Cleanup error: {e}")
 
    def _send_matched_data(self, left_matched_timestamp,right_matched_timestamp):
        try:
            left_data = self.arm_data_cache[left_matched_timestamp]["left"]
            right_data = self.arm_data_cache[right_matched_timestamp]["right"]
            if len(left_data) != len(right_data):
                return
            merged_data = np.concatenate((left_data, right_data))
            print(merged_data)
            
            # # 重新解析原始时间戳（用于发送）
            # sec = timestamp_ns // 1_000_000_000
            # nanosec = timestamp_ns % 1_000_000_000
            # # np.array([sec, nanosec], dtype=np.uint32).tobytes(),
            self.socket_joint.send_multipart([
                b"state_joint_arms_merged",
                merged_data.tobytes()
            ], flags=zmq.NOBLOCK)
            del self.arm_data_cache[left_matched_timestamp]
            del self.arm_data_cache[right_matched_timestamp]
 
        except Exception as e:
            self.get_logger().error(f"Failed to send matched data: {e}")
 
    def _cleanup_expired_data(self):
        current_time_ns = time.time_ns()
        expired_timestamps = [
            ts for ts, data in self.arm_data_cache.items()
            if (current_time_ns - data["timestamp_ns"]) > self.MAX_DATA_AGE_NS
        ]
        
        for ts in expired_timestamps:
            sec = ts // 1_000_000_000
            nanosec = ts % 1_000_000_000
            self.get_logger().warn(
                f"Dropping expired data (age: {(current_time_ns - self.arm_data_cache[ts]['timestamp_ns'])/1e6:.2f}ms) "
                f"at timestamp sec={sec}, nsec={nanosec}"
            )
            del self.arm_data_cache[ts]

    def ros2_to_zeromq_images_top_left_callback(self, msg):
        """ROS 2 -> ZeroMQ"""
        try:
            event_id = "image_top_left"
            buffer_bytes = msg.data
            meta = {
                "encoding":"JPEG",
                "width": 640,
                "height": 480
            }
            meta_bytes = json.dumps(meta).encode('utf-8')
            # self.socket_image.send_multipart([
            #     event_id.encode('utf-8'),
            #     buffer_bytes,
            #     meta_bytes,
            # ], flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().error(f"ROS2->ZeroMQ error: {e}")

    def ros2_to_zeromq_images_top_right_callback(self, msg):
        """ROS 2 -> ZeroMQ"""
        try:
            event_id = "image_top_right"
            buffer_bytes = msg.data
            meta = {
                "encoding":"JPEG",
                "width": 640,
                "height": 480
            }
            meta_bytes = json.dumps(meta).encode('utf-8')
            # self.socket_image.send_multipart([
            #     event_id.encode('utf-8'),
            #     buffer_bytes,
            #     meta_bytes,
            # ], flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().error(f"ROS2->ZeroMQ error: {e}")

    def ros2_to_zeromq_images_wrist_left_callback(self, msg):
        """ROS 2 -> ZeroMQ"""
        try:
            event_id = "image_wrist_left"
            buffer_bytes = msg.data
            meta = {
                "encoding":"JPEG",
                "width": 640,
                "height": 480
            }
            meta_bytes = json.dumps(meta).encode('utf-8')
            # self.socket_image.send_multipart([
            #     event_id.encode('utf-8'),
            #     buffer_bytes,
            #     meta_bytes,
            # ], flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().error(f"ROS2->ZeroMQ error: {e}")

    def ros2_to_zeromq_images_wrist_right_callback(self, msg):
        """ROS 2 -> ZeroMQ"""
        try:
            event_id = "image_wrist_right"
            buffer_bytes = msg.data
            meta = {
                "encoding":"JPEG",
                "width": 640,
                "height": 480
            }
            meta_bytes = json.dumps(meta).encode('utf-8')
            # self.socket_image.send_multipart([
            #     event_id.encode('utf-8'),
            #     buffer_bytes,
            #     meta_bytes,
            # ], flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().error(f"ROS2->ZeroMQ error: {e}")

    def zeromq_receive_loop(self):
        """ZeroMQ -> ROS 2 (独立线程)"""
        while self.running:
            try:
                # 使用 poll 设置超时，避免完全阻塞
                if self.socket_joint.poll(timeout=100):  # 100ms 超时
                    event_id, buffer_bytes = self.socket_joint.recv_multipart()
                    event_id = event_id.decode('utf-8')
                    
                    if 'action_joint' in event_id:
                        array = np.frombuffer(buffer_bytes, dtype=np.float32)
                        msg = Float32MultiArray()
                        msg.data = array.tolist()
                        # 使用线程安全的方式发布消息
                        self.publisher_arm_left.publish(msg)
            except zmq.Again:
                continue  # 超时后继续循环
            except Exception as e:
                self.get_logger().error(f"ZeroMQ->ROS2 error: {e}")
                time.sleep(1)  # 出错后稍作等待

    def destroy(self):
        """清理资源"""
        self.running = False
        if self.zeromq_thread.is_alive():
            self.zeromq_thread.join()
        self.socket_joint.close()
        self.socket_image.close()
        self.galaxea_context.term()
        super().destroy_node()

def main():
    rclpy.init()
    node = ROS2BridgeNode()
    
    try:
        rclpy.spin(node)  # 运行节点
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()

if __name__ == "__main__":
    main()