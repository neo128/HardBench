import json
import time
import pyarrow as pa
from dora import Node
import requests

# 你的 JSON 字符串
config_str = '''
{
    "device_name": "睿尔曼双臂轮式机器人V1.1",
    "device_body": "睿尔曼",
    "specifications": {
        "end_type": "",
        "fps": 20,
        "camera": {
            "number": 3,
            "information": [
                {
                    "name": "cam_high",
                    "chinese_name": "头部摄像头",
                    "type": "Intel RealSense D435",
                    "width": 640,
                    "height": 480,
                    "is_connect": false
                },
                {
                    "name": "cam_left_wrist",
                    "chinese_name": "左腕摄像头",
                    "type": "Intel RealSense D435",
                    "width": 640,
                    "height": 480,
                    "is_connect": false
                },
                {
                    "name": "cam_right_wrist",
                    "chinese_name": "右腕摄像头",
                    "type": "Intel RealSense D435",
                    "width": 640,
                    "height": 480,
                    "is_connect": false
                }
            ]
        },
        "piper": {
            "number": 2,
            "information": [
                {
                    "name": "piper_left",
                    "type": "RM75-6FB",
                    "start_pose": [
                        -90.0,
                        90.0,
                        90.0,
                        -90.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    "joint_p_limit": [
                        169.0,
                        102.0,
                        169.0,
                        52.0,
                        169.0,
                        117.0,
                        169.0
                    ],
                    "joint_n_limit": [
                        -169.0,
                        -102.0,
                        -169.0,
                        -167.0,
                        -169.0,
                        -87.0,
                        -169.0
                    ],
                    "is_connect": false
                },
                {
                    "name": "piper_right",
                    "type": "RM75-6FB",
                    "start_pose": [
                        -90.0,
                        90.0,
                        90.0,
                        -90.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    "joint_p_limit": [
                        169.0,
                        102.0,
                        169.0,
                        52.0,
                        169.0,
                        117.0,
                        169.0
                    ],
                    "joint_n_limit": [
                        -169.0,
                        -102.0,
                        -169.0,
                        -167.0,
                        -169.0,
                        -87.0,
                        -169.0
                    ],
                    "is_connect": false
                }
            ]
        }
    }
}
'''

# 将 JSON 字符串解析为 JSON 对象
config_json = json.loads(config_str)
session = requests.Session()
# 设置默认请求头，确保Content-Type为application/json
session.headers.update({"Content-Type": "application/json"})

# 定义一个集合，用于跟踪哪些节点已经返回了信息
received_nodes = set()

# 定义所有需要接收信息的节点
all_nodes = {
    "hw_camera_top_single",
    "hw_camera_left_single",
    "hw_camera_right_single",
    "hw_arm_left_single",
    "hw_arm_right_single",
}
server_url="http://localhost:8088"

def main():
    node = Node()
    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "tick":
                metadata = event["metadata"]
                metadata["timestamp"] = time.time_ns()
                node.send_output(
                    "get_hw_info_single",
                    pa.array([b"True"]),
                    metadata,
                )
            elif event["id"] in all_nodes:
                # 更新对应的 is_connect 状态
                if event["id"] == "hw_camera_top_single":
                    config_json["specifications"]["camera"]["information"][0]["is_connect"] = True
                elif event["id"] == "hw_camera_left_single":
                    config_json["specifications"]["camera"]["information"][1]["is_connect"] = True
                elif event["id"] == "hw_camera_right_single":
                    config_json["specifications"]["camera"]["information"][2]["is_connect"] = True
                elif event["id"] == "hw_arm_left_single":
                    config_json["specifications"]["piper"]["information"][0]["is_connect"] = True
                elif event["id"] == "hw_arm_right_single":
                    config_json["specifications"]["piper"]["information"][1]["is_connect"] = True
                
                # 将当前节点加入已接收集合
                received_nodes.add(event["id"])
                
                # 检查是否所有节点都已返回信息
                if received_nodes == all_nodes:
                    # 所有节点都已返回信息，发送 config_json
                    url = f"{server_url}/robot/update_machine_information"
                    try:
                        # 发送请求时会使用session中设置的默认头
                        response = session.post(url, json=config_json)
                        if response.status_code != 200:
                            print(f"Server returned error: {response.status_code}, {response.text}")
                        else:
                            print("Data sent successfully")
                    except requests.exceptions.RequestException as e:
                        print(f"Request failed: {e}")
                    # 重置已接收节点集合，以便重新开始
                    received_nodes.clear()
        elif event["type"] == "ERROR":
            raise RuntimeError(event["error"])

if __name__ == "__main__":
    main()
