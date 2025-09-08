import zmq
import threading
import pyarrow as pa
import time
import json
import rclpy
from rclpy.node import Node as ROS2Node
from std_msgs.msg import Float32MultiArray
from dora import Node
import numpy as np
import queue

# IPC 地址
ipc_address_image = "ipc:///tmp/dora-zeromq-ruantong-image"
ipc_address_joint = "ipc:///tmp/ros2-zeromq-ruantong-joint"

# 全局变量
context = zmq.Context()
socket_image = context.socket(zmq.PAIR)
socket_image.bind(ipc_address_image)
socket_image.setsockopt(zmq.SNDHWM, 2000)
socket_image.setsockopt(zmq.SNDBUF, 2**25)
socket_image.setsockopt(zmq.SNDTIMEO, 2000)
socket_image.setsockopt(zmq.RCVTIMEO, 2000)
socket_image.setsockopt(zmq.LINGER, 0)

socket_joint = context.socket(zmq.PAIR)
socket_joint.bind(ipc_address_joint)
socket_joint.setsockopt(zmq.SNDHWM, 2000)
socket_joint.setsockopt(zmq.SNDBUF, 2**25)
socket_joint.setsockopt(zmq.SNDTIMEO, 2000)
socket_joint.setsockopt(zmq.RCVTIMEO, 2000)
socket_joint.setsockopt(zmq.LINGER, 0)

running_server = True
output_queue = queue.Queue()

class ROS2BridgeNode(ROS2Node):
    def __init__(self):
        super().__init__('dora_ros2_bridge')
        # 订阅 robot_actions 话题
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'robot_actions',
            self.listener_callback,
            10)
        # 发布 robot_control 话题
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'robot_control',
            10)
        self.get_logger().info("ROS2节点初始化完成")

    def listener_callback(self, msg):
        try:
            array = np.array(msg.data, dtype=np.float32)
            self.get_logger().info(f"收到ROS2消息: {array}")  # 添加日志
            socket_joint.send_multipart([
                b"action_joint",
                array.tobytes()
            ])
        except Exception as e:
            self.get_logger().error(f"ROS2->ZeroMQ error: {e}")

def dora_node_thread(dora_node):
    """Dora 节点处理线程"""
    try:
        for event in dora_node:
            if event["type"] == "INPUT":
                event_id = event["id"]
                # 处理 Dora 输入数据
                if isinstance(event["value"], pa.Buffer):
                    buffer_bytes = event["value"].to_pybytes()
                else:
                    buffer_bytes = np.array(event["value"]).tobytes()
                
                # 发送到 ZeroMQ (图像数据)
                if "image" in event_id:
                    try:
                        socket_image.send_multipart([
                            event_id.encode('utf-8'),
                            buffer_bytes,
                            json.dumps(event.get("metadata", {})).encode('utf-8')
                        ], flags=zmq.NOBLOCK)
                    except zmq.Again:
                        print("Dora -> ZeroMQ: 发送超时")
            
            elif event["type"] == "STOP":
                break
    except Exception as e:
        print(f"Dora 节点错误: {e}")
    finally:
        global running_server
        running_server = False

def recv_zeromq_thread():
    """接收 ZeroMQ 数据并转发到 ROS2"""
    while running_server:
        try:
            message_parts = socket_joint.recv_multipart(flags=zmq.NOBLOCK)
            if message_parts and len(message_parts) >= 2:
                event_id = message_parts[0].decode('utf-8')
                buffer_bytes = message_parts[1]
                array = np.frombuffer(buffer_bytes, dtype=np.float32)
                
                if 'action_joint' in event_id:
                    output_queue.put(("action_joint", array))
        except zmq.Again:
            time.sleep(0.01)
        except Exception as e:
            print(f"ZeroMQ 接收错误: {e}")
            break

def ros2_publisher_thread(ros2_node):
    """ROS2 发布线程"""
    while rclpy.ok() and running_server:
        try:
            while not output_queue.empty():
                port, array = output_queue.get_nowait()
                msg = Float32MultiArray()
                msg.data = array.tolist()
                ros2_node.publisher.publish(msg)
                ros2_node.get_logger().info(f"发布ROS2消息: {array}")  # 添加日志
            time.sleep(0.01)
        except Exception as e:
            print(f"ROS2 发布错误: {e}")
            break

def ros2_spin(ros2_node):
    """ROS2 spin线程"""
    try:
        rclpy.spin(ros2_node)
    except Exception as e:
        ros2_node.get_logger().error(f"ROS2 spin错误: {e}")
    finally:
        global running_server
        running_server = False

if __name__ == "__main__":
    # 初始化 ROS2（必须在主线程）
    rclpy.init()
    ros2_node = ROS2BridgeNode()

    # 启动线程
    ros2_pub_thread = threading.Thread(
        target=ros2_publisher_thread,
        args=(ros2_node,),
        daemon=True
    )
    ros2_pub_thread.start()

    zeromq_recv_thread = threading.Thread(target=recv_zeromq_thread, daemon=True)
    zeromq_recv_thread.start()

    ros2_spin_thread = threading.Thread(
        target=ros2_spin,
        args=(ros2_node,),
        daemon=True
    )
    ros2_spin_thread.start()

    # 启动 Dora 节点线程
    dora_node = Node()
    dora_thread = threading.Thread(target=dora_node_thread, args=(dora_node,))
    dora_thread.start()

    try:
        # 主线程等待退出信号
        while rclpy.ok() and running_server:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("收到终止信号，正在清理...")
    finally:
        # 清理资源
        running_server = False
        dora_thread.join()
        zeromq_recv_thread.join()
        ros2_pub_thread.join()
        ros2_spin_thread.join()
        ros2_node.destroy_node()
        rclpy.shutdown()
        socket_image.close()
        socket_joint.close()
        context.term()