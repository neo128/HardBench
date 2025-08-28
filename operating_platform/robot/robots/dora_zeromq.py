import zmq
import threading
import pyarrow as pa
import time
from dora import Node
import numpy as np
import queue

node = Node()

# IPC Address
ipc_address = "ipc:///tmp/dora-zeromq"

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind(ipc_address)
socket.setsockopt(zmq.SNDHWM, 2000)
socket.setsockopt(zmq.SNDBUF, 2**25)
socket.setsockopt(zmq.RCVTIMEO, 300)
running_server = True

# 创建线程安全队列 (在全局作用域)
output_queue = queue.Queue()


# def recv_server():
#     while running_server:
#         try:
#             # 接收 multipart 消息
#             message_parts = socket.recv_multipart(flags=zmq.NOBLOCK)
#             if message_parts:
#                 if len(message_parts) < 2:
#                     continue  # 协议错误

#                 event_id = message_parts[0].decode('utf-8')
#                 buffer_bytes = message_parts[1]

#                 print(f"zmq recv event_id:{event_id}")

#                 array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()

#                 print(f"zmq recv array:{array}")

#                 if 'action_joint_right' in event_id:
#                     # joint_array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()

#                     node.send_output("action_joint_right", pa.array(array, type=pa.float32()))
#                     print(f"send event_id : <{event_id}>  succese")
#                 elif 'action_joint_left' in event_id:
#                     # joint_array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()
#                     node.send_output("action_joint_left", pa.array(array, type=pa.float32()))
#                     print(f"send event_id : <{event_id}>  succese")
#                 elif 'action_gripper_right' in event_id:
#                     # gripper_array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()
#                     node.send_output("action_gripper_right", pa.array(array, type=pa.float32()))
#                     print(f"send event_id : <{event_id}>  succese")
#                 elif 'action_gripper_left' in event_id:
#                     # gripper_array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()
#                     node.send_output("action_gripper_left", pa.array(array, type=pa.float32()))
#                     print(f"send event_id : <{event_id}>  succese")

#         except zmq.Again:
#             time.sleep(0.01)  # 避免忙等
#         except Exception as e:
#             print("recv error:", e)
#             break

def recv_server():
    while running_server:
        try:
            message_parts = socket.recv_multipart(flags=zmq.NOBLOCK)
            if message_parts and len(message_parts) >= 2:
                event_id = message_parts[0].decode('utf-8')
                buffer_bytes = message_parts[1]
                
                # print(f"zmq recv event_id:{event_id}")
                array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()
                # print(f"zmq recv array:{array}")

                # ✅ 仅将数据放入队列，不再操作 node
                if 'action_joint_right' in event_id:
                    output_queue.put(("action_joint_right", array))
                elif 'action_joint_left' in event_id:
                    output_queue.put(("action_joint_left", array))
                elif 'action_gripper_right' in event_id:
                    output_queue.put(("action_gripper_right", array))
                elif 'action_gripper_left' in event_id:
                    output_queue.put(("action_gripper_left", array))
                    
        except zmq.Again:
            time.sleep(0.01)
        except Exception as e:
            print("recv error:", e)
            break



if __name__ == "__main__":


    server_thread = threading.Thread(target=recv_server)
    server_thread.start()

    try:
        for event in node:
            # 1. 先处理队列中的待发送数据
            while not output_queue.empty():
                try:
                    port, array = output_queue.get_nowait()
                    # ✅ 此时在主线程，安全操作 node
                    node.send_output(port, pa.array(array, type=pa.float32()))
                    # print(f"MAIN THREAD: send event_id : <{port}> succese")
                    # print(f"MAIN THREAD: send array : <{array}> succese")
                except queue.Empty:
                    break

            if event["type"] == "INPUT":
                event_id = event["id"]
                buffer_bytes = event["value"].to_numpy().tobytes()
                            
                # 处理接收到的数据
                # print(f"Send event: {event_id}")
                # print(f"Buffer size: {len(buffer_bytes)} bytes")

                try:
                    socket.send_multipart([
                        event_id.encode('utf-8'),
                        buffer_bytes
                    ], flags=zmq.NOBLOCK)
                except zmq.Again:
                    # print("Socket would block, skipping send this frame...")
                    # continue
                    # continue
                    pass
                # time.sleep(0.001)
                
            elif event["type"] == "STOP":
                break
    
    finally:
        # Close server 
        running_server = False
        server_thread.join()

        # Close zmq
        socket.close()
        context.term()
