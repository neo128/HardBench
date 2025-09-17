import numpy as np
from dora import Node

node = Node()
counter = 0

for event in node:
    if event["type"] == "INPUT" and event["id"] == "tick":
        # 模拟RGB图像 (640x480, 3通道)
        rgb_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # 模拟深度图 (单通道)
        depth_image = np.random.rand(480, 640).astype(np.float32) * 10.0

        # 转换为 Python 原生字节流
        rgb_bytes = rgb_image.tobytes()  # 返回 `bytes` 对象
        depth_bytes = depth_image.tobytes()

        # 直接发送字节流
        node.send_output("image", rgb_bytes)  # 直接传 `bytes`
        node.send_output("image_depth", depth_bytes)
        counter += 1