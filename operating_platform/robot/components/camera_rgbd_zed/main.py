import os
import time
import numpy as np
import cv2
import pyarrow as pa
import pyzed.sl as sl
from dora import Node
from enum import Enum

RUNNER_CI = True if os.getenv("CI") == "true" else False

class CaptureMode(Enum):
    LEFT_AND_RIGHT = 1  
    LEFT_AND_DEPTH = 2  
    LEFT_AND_DEPTH_16 = 3 

def encode_frame(frame, encoding):    
    """统一的图像编码函数"""    
    if encoding == "bgr8":    
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 从RGB转BGR  
    elif encoding == "rgb8":    
        return frame  # 已经是RGB格式  
    elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:    
        # OpenCV的imencode期望BGR格式  
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        ret, encoded_frame = cv2.imencode("." + encoding, bgr_frame)    
        if not ret:    
            return None    
        return encoded_frame    
    else:    
        return frame

def main():

    #获得环境变量值
    flip = os.getenv("FLIP","")
    device_serial = os.getenv("DEVICE_SERIAL","")
    image_hight = int(os.getenv("IMAGE_HEIGHT","480"))
    image_width = int(os.getenv("IMAGE_WIDTH","640"))
    encoding = os.getenv("ENCODING","bgr8")
    capture_mode = int(os.getenv("CAPTURE_MODE", "1"))

    app_mode = CaptureMode(capture_mode)

    #zed相机初始化
    zed = sl.Camera()
    init_params = sl.InitParameters()  
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 30  
    
    if app_mode == CaptureMode.LEFT_AND_RIGHT:  
        init_params.depth_mode = sl.DEPTH_MODE.NONE  
    else:  
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA 

    if device_serial:  
        init_params.set_from_serial_number(int(device_serial))

    err = zed.open(init_params)  
    if err != sl.ERROR_CODE.SUCCESS:  
        raise ConnectionError(f"ZED camera failed to open: {err}")
    
    camera_info = zed.get_camera_information()

    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat() 

    #dora节点与事件循环主函数
    node = Node()  
    start_time = time.time()
    pa.array([])
    
    print(f"ZED Camera initialized with mode: {app_mode.name}")

    for event in node:
        if RUNNER_CI and time.time() - start_time > 10:  
            break  
        event_type = event["type"]
        if event_type == "INPUT":  
            event_id = event["id"]      
            if event_id == "tick":   
                if zed.grab() == sl.ERROR_CODE.SUCCESS:  
                    
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)

                    if app_mode == CaptureMode.LEFT_AND_RIGHT:    
                        zed.retrieve_image(right_image, sl.VIEW.RIGHT)    
                        right_frame = np.asanyarray(right_image.get_data())  
                        right_frame = cv2.resize(right_frame, (image_width, image_hight))  
                    elif app_mode == CaptureMode.LEFT_AND_DEPTH:    
                        zed.retrieve_image(right_image, sl.VIEW.DEPTH)     
                        right_frame = np.asanyarray(right_image.get_data())  
                        right_frame = cv2.resize(right_frame, (image_width, image_hight))  
                    elif app_mode == CaptureMode.LEFT_AND_DEPTH_16:    
                        zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)  
                        depth_data = np.asanyarray(depth_image.get_data())  
                        depth_data = cv2.resize(depth_data, (image_width, image_hight))   
                        right_frame = depth_data  

                    left_frame = np.asanyarray(left_image.get_data())
                    left_frame = cv2.resize(left_frame, (image_width, image_hight))  
                    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGBA2RGB)
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
                    left_metadata["capture_mode"] = app_mode.name
                    
                    left_frame_encoded = encode_frame(left_frame, encoding)  
                    if left_frame_encoded is None:  
                        print("Error encoding left image...")  
                        continue 
                     
                    left_storage = pa.array(left_frame.ravel())  
                    left_metadata["resolution"] = [int(left_calibration.cx), int(left_calibration.cy)]  
                    left_metadata["focal_length"] = [int(left_calibration.fx), int(left_calibration.fy)]  
                    left_metadata["timestamp"] = time.time_ns()  
                      
                    node.send_output("left_image", left_storage, left_metadata)

                    #右
                    right_metadata = event["metadata"].copy()  
  
                    right_metadata["width"] = int(right_frame.shape[1])  
                    right_metadata["height"] = int(right_frame.shape[0])
                    right_metadata["capture_mode"] = app_mode.name
                      
                    if app_mode == CaptureMode.LEFT_AND_RIGHT:    
                        output_name = "right_image"    
                        right_metadata["encoding"] = encoding    
                        right_metadata["data_type"] = "right_camera"    
                        right_frame_encoded = encode_frame(right_frame, encoding)  
                          
                    elif app_mode == CaptureMode.LEFT_AND_DEPTH:  
                        output_name = "depth_image"  
                        right_metadata["encoding"] = encoding  
                        right_metadata["data_type"] = "depth_view"  
                        right_frame_encoded = encode_frame(right_frame, encoding)  
                          
                    elif app_mode == CaptureMode.LEFT_AND_DEPTH_16:  
                        output_name = "depth_data"  
                        right_metadata["encoding"] = "16UC1"  # 16位单通道  
                        right_metadata["data_type"] = "depth_16bit"  
                        right_metadata["depth_unit"] = "millimeter"  
                        depth_cleaned = np.nan_to_num(right_frame, nan=0.0, posinf=65535, neginf=0.0)  
                        depth_cleaned = np.clip(depth_cleaned, 0, 65535)
 
                        right_frame_encoded = depth_cleaned.astype(np.uint16)  

                    
                    if right_frame_encoded is None:  
                        print(f"Error encoding {output_name}...")  
                        continue

                    right_storage = pa.array(right_frame_encoded.ravel())    
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