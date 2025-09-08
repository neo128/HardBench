import zmq
import threading
import pyarrow as pa
import time
from dora import Node
import numpy as np
import queue
import json


# IPC Address
ipc_address_image = "ipc:///tmp/dora-zeromq-so101-image"
ipc_address_joint = "ipc:///tmp/dora-zeromq-so101-joint"

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

# 创建线程安全队列 (在全局作用域)
output_queue = queue.Queue()

def recv_server():
    while running_server:
        try:
            message_parts = socket_joint.recv_multipart()
            if message_parts and len(message_parts) >= 2:
                event_id = message_parts[0].decode('utf-8')
                buffer_bytes = message_parts[1]
                
                array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()

                if 'action_joint' in event_id:
                    output_queue.put(("action_joint", array))
                    
        except zmq.Again:
            print(f"Dora ZeroMQ Received Timeout")
            time.sleep(0.01)
            continue
            
        except Exception as e:
            print("recv error:", e)
            break



if __name__ == "__main__":
    node = Node()

    server_thread = threading.Thread(target=recv_server)
    server_thread.start()

    try:
        for event in node:
            while not output_queue.empty():
                try:
                    port, array = output_queue.get_nowait()
                    node.send_output(port, pa.array(array, type=pa.float32()))
                    # print(f"MAIN THREAD: send event_id : <{port}> succese")
                    # print(f"MAIN THREAD: send array : <{array}> succese")
                except queue.Empty:
                    break

            if event["type"] == "INPUT":
                event_id = event["id"]
                buffer_bytes = event["value"].to_numpy().tobytes()
                meta_bytes = json.dumps(event["metadata"]).encode('utf-8')
                            
                # 处理接收到的数据
                # print(f"Send event: {event_id}")
                # print(f"Buffer size: {len(buffer_bytes)} bytes")

                if "image" in event_id:
                    try:
                        socket_image.send_multipart([
                            event_id.encode('utf-8'),
                            buffer_bytes,
                            meta_bytes,
                        ], flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                else:
                    try:
                        socket_joint.send_multipart([
                            event_id.encode('utf-8'),
                            buffer_bytes
                        ], flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                
            elif event["type"] == "STOP":
                break
    
    finally:
        # Close server 
        running_server = False
        server_thread.join()

        # Close zmq
        socket_image.close()
        socket_joint.close()

        context.term()
