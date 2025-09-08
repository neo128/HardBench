#!/usr/bin/python3
import rclpy
from rclpy.node import Node as ROS2Node
from std_msgs.msg import Float32MultiArray
import zmq
import threading
import numpy as np
import time

# IPC Address
ipc_address_joint = "ipc:///tmp/ros2-zeromq-ruantong-joint"

class ROS2BridgeNode(ROS2Node):
    def __init__(self):
        super().__init__('ros2_zeromq_bridge')
        
        # ROS 2 订阅与发布
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'robot_actions',
            self.ros2_to_zeromq_callback,
            10)
        
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'robot_control',
            10)
        
        # ZeroMQ 初始化
        self.joint_context = zmq.Context()
        self.socket_joint = self.joint_context.socket(zmq.PAIR)
        self.socket_joint.bind(ipc_address_joint)
        self.socket_joint.setsockopt(zmq.SNDHWM, 2000)
        self.socket_joint.setsockopt(zmq.SNDBUF, 2**25)
        self.socket_joint.setsockopt(zmq.LINGER, 0)
        
        # 运行标志
        self.running = True
        
        # 启动 ZeroMQ 接收线程
        self.zeromq_thread = threading.Thread(target=self.zeromq_receive_loop)
        self.zeromq_thread.daemon = True
        self.zeromq_thread.start()

    def ros2_to_zeromq_callback(self, msg):
        """ROS 2 -> ZeroMQ"""
        try:
            array = np.array(msg.data, dtype=np.float32)
            print(array)
            self.socket_joint.send_multipart([
                b"action_joint",
                array.tobytes()
            ])
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
                        self.publisher.publish(msg)
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
        self.joint_context.term()
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