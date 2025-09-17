import os
import time
import numpy as np
import cv2
import pyarrow as pa
import pyzed.sl as sl
from dora import Node

RUNNER_CI = True if os.getenv("CI") == "true" else False

def main():

    #获得环境变量值
    flip = os.getenv("FLIP","")
    device_serial = os.getenv("DEVICE_SERIAL","")
    image_hight = int(os.getenv("IMAGE_HEIGHT","480"))
    image_width = int(os.getenv("IMAGE_WIDTH","640"))
    encoding = os.getenv("ENCODING","bgr8")

    #zed相机初始化
    zed = sl.Camera()
    init_params = sl.InitParameters()  
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30  
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if device_serial:  
        init_params.set_from_serial_number(int(device_serial))

    err = zed.open(init_params)  
    if err != sl.ERROR_CODE.SUCCESS:  
        raise ConnectionError(f"ZED camera failed to open: {err}")
    
    camera_info = zed.get_camera_information()

    left_image = sl.Mat()
    right_image = sl.Mat()

    #dora节点与事件循环主函数
    node = Node()  
    start_time = time.time()
    pa.array([])

    for event in node:
        if RUNNER_CI and time.time() - start_time > 10:  
            break  
        event_type = event["type"]
        if event_type == "INPUT":  
            event_id = event["id"]      
            if event_id == "tick":   
                if zed.grab() == sl.ERROR_CODE.SUCCESS:  
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)
                    zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                    left_frame = np.asanyarray(left_image.get_data())
                    right_frame = np.asanyarray(right_image.get_data()) 
                    left_frame = cv2.resize(left_frame, (640, 480))  
                    right_frame = cv2.resize(right_frame, (640, 480))
                    #旋转
                    if flip == "VERTICAL":  
                        left_frame = cv2.flip(left_frame, 0)  
                        right_frame = cv2.flip(right_frame, 0)  
                    elif flip == "HORIZONTAL":  
                        left_frame = cv2.flip(left_frame, 1)  
                        right_frame = cv2.flip(right_frame, 1)  
                    elif flip == "BOTH":  
                        left_frame = cv2.flip(left_frame, -1)  
                        right_frame = cv2.flip(right_frame, -1)
                    #参数
                    left_calibration = camera_info.camera_configuration.calibration_parameters.left_cam  
                    right_calibration = camera_info.camera_configuration.calibration_parameters.right_cam

                    #左
                    left_metadata = event["metadata"].copy()  
                    left_metadata["encoding"] = encoding  
                    left_metadata["width"] = int(left_frame.shape[1])  
                    left_metadata["height"] = int(left_frame.shape[0])

                    if encoding == "bgr8":  
                        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGBA2BGR)  
                    elif encoding == "rgb8":  
                        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGBA2RGB)  
                    elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:  
                        ret, left_frame = cv2.imencode("." + encoding, left_frame)  
                        if not ret:  
                            print("Error encoding left image...")  
                            continue  
                    left_storage = pa.array(left_frame.ravel())  
                    left_metadata["resolution"] = [int(left_calibration.cx), int(left_calibration.cy)]  
                    left_metadata["focal_length"] = [int(left_calibration.fx), int(left_calibration.fy)]  
                    left_metadata["timestamp"] = time.time_ns()  
                      
                    node.send_output("left_image", left_storage, left_metadata)

                    #右
                    right_metadata = event["metadata"].copy()  
                    right_metadata["encoding"] = encoding  
                    right_metadata["width"] = int(right_frame.shape[1])  
                    right_metadata["height"] = int(right_frame.shape[0])

                    if encoding == "bgr8":  
                        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGBA2BGR)  
                    elif encoding == "rgb8":  
                        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGBA2RGB)  
                    elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:  
                        ret, right_frame = cv2.imencode("." + encoding, right_frame)  
                        if not ret:  
                            print("Error encoding right image...")  
                            continue  
                      
                    right_storage = pa.array(right_frame.ravel())  
                    right_metadata["resolution"] = [int(right_calibration.cx), int(right_calibration.cy)]  
                    right_metadata["focal_length"] = [int(right_calibration.fx), int(right_calibration.fy)]  
                    right_metadata["timestamp"] = time.time_ns()  
                      
                    node.send_output("right_image", right_storage, right_metadata)
            
        elif event_type == "ERROR":  
            raise RuntimeError(event["error"])  
          
        if event_type == "STOP":  
            break  

    zed.close()

if __name__ == "__main__":
    main()