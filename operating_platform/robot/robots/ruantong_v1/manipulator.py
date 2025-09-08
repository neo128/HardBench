import threading
import zmq


ipc_address_image = "ipc:///tmp/dora-zeromq-ruantong-image"
ipc_address_joint = "ipc:///tmp/ros2-zeromq-ruantong-joint"

recv_images = {}
recv_joint = {}
lock = threading.Lock()  # 线程锁

running_recv_image_server = True
running_recv_joint_server = True

zmq_context = zmq.Context()

socket_image = zmq_context.socket(zmq.PAIR)
socket_image.connect(ipc_address_image)
socket_image.setsockopt(zmq.RCVTIMEO, 2000)

socket_joint = zmq_context.socket(zmq.PAIR)
socket_joint.connect(ipc_address_joint)
socket_joint.setsockopt(zmq.RCVTIMEO, 2000)

def recv_joint_server():
    """接收数据线程"""
    while running_recv_joint_server:
        try:
            message_parts = socket_joint.recv_multipart()
            print(message_parts)
            if len(message_parts) < 2:
                continue  # 协议错误
        except zmq.Again:
            print(f"ruantong Joint Received Timeout")
            continue
        except Exception as e:
            print("recv joint error:", e)
            break

if __name__ == '__main__':
    recv_joint_server()