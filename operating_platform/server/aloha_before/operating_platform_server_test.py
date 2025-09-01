coding = "utf-8"

from gevent import monkey
monkey.patch_all()

from flask import Flask, jsonify, Response, request, session
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import os
import logging
import datetime
from flask_socketio import SocketIO, emit
import schedule
import requests
import json
import upload_to_nas
import uuid

MACHINE_ID_FILE = './machine_id.txt'
MACHINE_ID_PATH = '/home/machine/.config/baai_platform/baai_server_machine_id'

class VideoStream:
    def __init__(self, stream_id, stream_name):
        self.stream_id = stream_id
        self.name = stream_name
        self.running = False
        self.frame_buffers = [None, None]  # 双缓冲
        self.buffer_index = 0
        self.lock = threading.Lock()
         
        

    def start(self):
        """启动视频流（仅标记为运行）"""
        if self.running:
            print(f"已经启动视频流")
            return True
        self.running = True
        return True
    
    def stop(self):
        """停止视频流"""
        self.running = False

    def update_frame(self, frame_data):
        """接收外部帧数据并更新当前帧"""
        if not self.running:
            return
        
        # 解码图像
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        
        # 压缩图像（可选）
        img = cv2.resize(img, (640, 480))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, jpeg = cv2.imencode('.jpg', img, encode_param)
        compressed_frame = jpeg.tobytes()

        with self.lock:
            self.buffer_index = 1 - self.buffer_index
            self.frame_buffers[self.buffer_index] = compressed_frame

    def get_frame(self):
        if not self.running:
            return self.generate_blank_frame()
        with self.lock:
            return self.frame_buffers[self.buffer_index]
    
    @staticmethod 
    def generate_blank_frame():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank)
        return jpeg.tobytes()

 
def get_machine_id():
    if os.path.exists(MACHINE_ID_FILE):
        with open(MACHINE_ID_FILE, 'r') as f:
            return f.read().strip()
    else:
        return "BAAI_AX_AL_001"


class FlaskServer:
    def __init__(self):
        # 初始化Flask应用
        self.app = Flask(__name__)
        self.app.secret_key = 'agilex'  # 暂时密钥
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)

        #self.web = "http://120.92.116.59:80/api"
        #self.web = "http://172.16.17.253:8080/api"
        self.web = "http://ei2rmd.baai.ac.cn/api"
        self.session = requests.Session() 
        self.token = None

        # 初始化日志
        self.init_logging()
        
        # 初始化实例变量
        self.robot_sid = None
        self.video_list = {}
        self.video_timestamp = time.time()
        self.video_streams = {}
        self.stream_status = {}
        self.frame_lock = threading.Lock()
        self.upload_lock = threading.Lock()  # 用于保护 self.upload_nas_flag 的访问
        self.init_streams_flag = False
        self.task_steps = {}
        self.upload_thread = threading.Thread(target=self.time_job, daemon=True)
        self.upload_nas_flag = False
        
        # 响应模板
        self.response_start_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_finish_collection = {
            "timestamp": time.time(),
            "msg": None,
            "data": None
        }
        self.response_submit_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_discard_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        
        # 注册路由
        self.register_routes()


    # ---------------------------------生成设备ID---------------------------------------------
    def get_address(self):
        # """获取 MAC 地址（格式：A1B2C3D4E5F6）"""
        # mac = uuid.getnode()
        # mac_hex = '{:012X}'.format(mac)  # 12 位大写十六进制
        # return mac_hex
        """生成随机唯一标识（不依赖硬件）"""
        return uuid.uuid4().hex.upper()[:12]
    
    def generate_machine_id(self):
        """生成机器 ID（MAC地址_aloha）"""
        mac = self.get_address()
        return f"{mac}_aloha"  # 格式：A1B2C3D4E5F6_aloha
    
    def save_machine_id(self,machine_id):
        """保存机器 ID 到"""
        try:
            os.makedirs(os.path.dirname(MACHINE_ID_PATH), exist_ok=True)  # 确保目录存在
            with open(MACHINE_ID_PATH, "w") as f:
                f.write(machine_id)
            logging.info(f"机器 ID 已保存到 {MACHINE_ID_PATH}")
            return True
        except Exception as e:
            logging.error(f"保存机器 ID 失败: {e}")
            return False
    
    def load_machine_id(self):
        """读取机器 ID"""
        if not os.path.exists(MACHINE_ID_PATH):
            logging.info("未找到机器 ID 文件，将生成新 ID")
            return None
        try:
            with open(MACHINE_ID_PATH, "r") as f:
                machine_id = f.read().strip()
                logging.info(f"已加载机器 ID: {machine_id}")
                return machine_id
        except Exception as e:
            logging.error(f"读取机器 ID 失败: {e}")
            return None
    
    def get_or_create_machine_id(self):
        """主逻辑：检查是否存在，不存在则生成并保存"""
        # 1. 尝试读取现有 ID
        machine_id = self.load_machine_id()
        if machine_id:
            return machine_id
        try:
            # 2. 不存在则生成新 ID
            logging.info("未找到机器 ID，正在生成...")
            machine_id = self.generate_machine_id()
            logging.info(f"生成的机器 ID: {machine_id}")
        
            # 3. 保存到用户目录
            if self.save_machine_id(machine_id):
                return machine_id
            else:
                logging.warning("警告：无法保存机器 ID")
                return get_machine_id()
        except Exception as e:
            logging.error(f"未知错误，无法生成机器 ID: {e}")
            return get_machine_id()
    
    # --------------------------------定时任务-------------------------------------------------
    def login(self):
        """发送登录请求"""
        logging.info("[API Request] login - 开始登录云平台")
        url = f"{self.web}/login"
        
        data = {
            "username": "eai_data_collect",
            "password": "eai_collect@2025"
        }
        
        try:
            response = self.session.post(url, json=data)
            logging.info(f"[API Response] login - 状态码: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                logging.info("[API Response] login - 登录成功")
                self.token = response_data["token"]
                return True
            else:
                logging.error(f"[API Response] login - 登录失败，状态码: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logging.error(f"[API Error] login - 请求异常: {str(e)}")
            return False
        
    def make_request_with_token(self, path, data):
        """发送带有 token 和请求体的请求"""
        if not self.token:
            logging.warning("[API Warning] make_request_with_token - 未登录，无法发送请求")
            return None
    
        url = f"{self.web}/{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
        try:
            logging.info(f"[API Request] make_request_with_token - 路径: {path}, 数据: {data}")
            if data:
                response = self.session.post(url, headers=headers, json=data)
            else:
                response = self.session.get(url, headers=headers)
            
            logging.info(f"[API Response] make_request_with_token - 状态码: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                logging.info(f"[API Response] make_request_with_token - 成功: {response_data}")
                return response_data
            else:
                logging.error(f"[API Response] make_request_with_token - 失败，状态码: {response.status_code}, 响应: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"[API Error] make_request_with_token - 请求异常: {str(e)}")
            return None
 
    def local_to_nas(self):
        logging.info(f"[Task] local_to_nas - 任务执行开始于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.login():
            upload_to_nas.upload()
            with self.upload_lock:
                self.upload_nas_flag = False
            logging.info("[Task] local_to_nas - 任务执行完成")
        else:
            with self.upload_lock:
                self.upload_nas_flag = False
            logging.error("[Task] local_to_nas - 任务执行失败，登录不成功")

    def time_job(self):
        schedule.every().day.at("14:00").do(self.local_to_nas)
        logging.info("[Task] time_job - 定时任务已启动，每天23:00执行...")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logging.info("[Task] time_job - 定时任务已停止")
    
    def init_logging(self):
        """初始化日志配置"""
        now = datetime.datetime.now()
        file_name = "./log/" + now.strftime("%Y.%m.%d.%H.%M") + ".log"
        log_dir = "./log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logging.basicConfig(
            filename=file_name,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode="a"
        )
        # 添加控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(console_handler)
    
    def register_routes(self):
        """注册所有路由"""
        # 系统信息
        self.app.add_url_rule('/api/info', 'system_info', self.system_info, methods=['GET'])
        
        # 视频流管理
        self.app.add_url_rule('/api/stream_info', 'get_streams', self.get_streams, methods=['GET'])
        self.app.add_url_rule('/api/start_stream', 'start_stream', self.start_stream, methods=['POST'])
        self.app.add_url_rule('/api/get_stream/<stream_id>', 'stream_video', self.stream_video, methods=['GET'])
        self.app.add_url_rule('/api/stop_stream/<stream_id>', 'stop_stream', self.stop_stream, methods=['POST'])
        
        # 采集控制
        self.app.add_url_rule('/api/start_collection', 'start_collection', self.start_collection, methods=['POST'])
        self.app.add_url_rule('/api/finish_collection', 'finish_collection', self.finish_collection, methods=['POST'])
        self.app.add_url_rule('/api/discard_collection', 'discard_collection', self.discard_collection, methods=['POST'])
        self.app.add_url_rule('/api/submit_collection', 'submit_collection', self.submit_collection, methods=['POST'])

        # 手动上传nas
        self.app.add_url_rule('/api/manual_upload_nas', 'manual_upload_nas', self.manual_upload_nas, methods=['POST']) 

        # nas上传反馈
        self.app.add_url_rule('/api/upload_start', 'upload_start', self.upload_start, methods=['POST'])
        self.app.add_url_rule('/api/upload_finish', 'upload_finish', self.upload_finish, methods=['POST'])
        self.app.add_url_rule('/api/upload_fail', 'upload_fail', self.upload_fail, methods=['POST'])
        self.app.add_url_rule('/api/upload_process', 'upload_process', self.upload_process, methods=['POST'])
       
        
        # 机器人接口
        self.app.add_url_rule('/robot/update_stream/<stream_id>', 'update_frame', self.update_frame, methods=['POST'])
        self.app.add_url_rule('/robot/stream_info', 'robot_get_video_list', self.robot_get_video_list, methods=['POST'])
        self.app.add_url_rule('/robot/response', 'robot_response', self.robot_response, methods=['POST'])
        self.app.add_url_rule('/robot/get_task_steps', 'get_task_steps', self.get_task_steps, methods=['GET'])
        
        # WebSocket事件
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.on_event('HEARTBEAT', self.handle_heartbeat)
        self.socketio.on_event('disconnect', self.handle_disconnect)
    
    def send_message_to_robot(self, sid, message):
        """向特定机器人客户端发送消息"""
        logging.info(f"[WebSocket] send_message_to_robot - 发送消息到 {sid}: {message}")
        self.socketio.emit('robot_command', message, room=sid, namespace='/')

    
    # ---------------------- WebSocket 事件处理 ----------------------
    def handle_connect(self):
        self.robot_sid = request.sid
        logging.info(f"[WebSocket] handle_connect - 客户端连接: {self.robot_sid}")
    
    def handle_heartbeat(self):
        """响应心跳包"""
        logging.debug("[WebSocket] handle_heartbeat - 收到心跳包")
        emit('HEARTBEAT_RESPONSE', {'server': 'alive'})
    
    def handle_disconnect(self):
        logging.info(f"[WebSocket] handle_disconnect - 客户端断开连接: {self.robot_sid}")
    
    # ---------------------- 路由处理方法 ----------------------
    def system_info(self):
        """获取系统信息"""
        logging.info("[API] system_info - 获取系统信息请求")
        try:
            active_count = sum(1 for s in self.stream_status.values() if s["active"])
            
            response_data = {
                "status": "running",
                "streams_active": active_count,
                "total_streams": len(self.stream_status),
                "timestamp": time.time(),
                "streams": self.stream_status
            }
            logging.info("[API] system_info - 返回系统信息")
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] system_info - 异常: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    def get_streams(self):
        try:
            logging.info("[API] get_streams - 获取视频流列表请求")
            response_data = {
                "code": 200,
                "data": self.video_list,
                "msg": "success"
            }
            logging.info("[API] get_streams - 返回视频流列表")
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] get_streams - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def start_stream(self):
        try:
            logging.info("[API] start_stream - 启动视频流请求")
            data = request.get_json()
            logging.debug(f"[API] start_stream - 请求数据: {data}")
            
            stream_id = data.get('stream_id')
            if stream_id not in self.video_streams:
                logging.warning(f"[API] start_stream - 无效的视频流ID: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "无效的视频流ID"
                }
                return jsonify(response_data), 200
                
            success = self.video_streams[stream_id].start()
            if success:
                self.stream_status[stream_id]["active"] = True
                logging.info(f"[API] start_stream - 成功启动视频流: {stream_id}")
                response_data = {
                    "code": 200,
                    "data": {},
                    "msg": "success"
                }
                return jsonify(response_data), 200
            else:
                logging.error(f"[API] start_stream - 启动视频流失败: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "启动视频流失败"
                }
                return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] start_stream - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def stream_video(self, stream_id):
        try:
            logging.info(f"[API] stream_video - 获取视频流请求: {stream_id}")
            
            try:
                stream_id = int(stream_id)
            except ValueError:
                logging.warning(f"[API] stream_video - 无效的流ID: {stream_id}")
                response_data = {
                    "code": 400,
                    "data": {},
                    "msg": "无效的流ID,必须为数字"
                }
                return jsonify(response_data), 200
            
            if stream_id not in self.video_streams:
                logging.warning(f"[API] stream_video - 视频流不存在: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "视频流不存在"
                }
                return jsonify(response_data), 200
            
            if not self.video_streams[stream_id].running:
                logging.warning(f"[API] stream_video - 视频流未开启: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "视频流未开启"
                }
                return jsonify(response_data), 200
            
            def generate():
                max_retries = 10
                retry_count = 0
                try:
                    while True:
                        frame = self.video_streams[stream_id].get_frame()
                        if frame:
                            retry_count = 0
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        else:
                            retry_count += 1
                            if retry_count >= max_retries:
                                logging.warning(f"[API] stream_video - 超过最大重试次数: {stream_id}")
                                break
                        time.sleep(0.03)
                except GeneratorExit:
                    logging.info(f"[API] stream_video - 客户端断开连接: {stream_id}")
            
            logging.info(f"[API] stream_video - 开始传输视频流: {stream_id}")
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            logging.error(f"[API Error] stream_video - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def stop_stream(self, stream_id):
        try:
            logging.info(f"[API] stop_stream - 停止视频流请求: {stream_id}")
            
            try:
                stream_id = int(stream_id)
            except ValueError:
                logging.warning(f"[API] stop_stream - 无效的流ID: {stream_id}")
                response_data = {
                    "code": 400,
                    "data": {},
                    "msg": "无效的流ID,必须为数字"
                }
                return jsonify(response_data), 200
            
            if stream_id not in self.video_streams:
                logging.warning(f"[API] stop_stream - 视频流不存在: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "视频流不存在"
                }
                return jsonify(response_data), 200
                
            self.video_streams[stream_id].stop()
            self.stream_status[stream_id]["active"] = False
            logging.info(f"[API] stop_stream - 成功停止视频流: {stream_id}")
            
            response_data = {
                "code": 200,
                "data": {},
                "msg": "stopped"
            }
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] stop_stream - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def init_streams(self):
        """初始化视频流"""
        logging.info("[API] init_streams - 初始化视频流")
        if 'streams' in self.video_list:
            for stream in self.video_list['streams']:
                self.video_streams[stream['id']] = VideoStream(stream['id'], stream['name'])
                self.stream_status[stream['id']] = {
                    "name": str(stream['name']),
                    "active": False
                }
        logging.debug(f"[API] init_streams - 初始化完成: {self.video_list}")
    
    def start_collection(self):
        try:
            logging.info("[API] start_collection - 开始采集请求")
            data = request.get_json()
            logging.debug(f"[API] start_collection - 请求数据: {data}")
            
            data['machine_id'] = get_machine_id()
            self.task_steps = data
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'start_collection', 'msg': data})
            
            while True:
                if 0 < self.response_start_collection["timestamp"] - now_time < 8:
                    if self.response_start_collection['msg'] == "success":
                        logging.info("[API] start_collection - 采集开始成功")
                        response_data = {
                            "code": 200,
                            "data": {},
                            "msg": self.response_start_collection['msg']
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] start_collection - 采集开始失败: {self.response_start_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_start_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 8:
                    logging.warning("[API] start_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] start_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def finish_collection(self):
        try:
            logging.info("[API] finish_collection - 完成采集请求")
            data = request.get_json()
            logging.debug(f"[API] finish_collection - 请求数据: {data}")
            
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'finish_collection'})
            
            while True:
                if 0 < self.response_finish_collection["timestamp"] - now_time < 100:
                    if self.response_finish_collection['msg'] == "success":
                        logging.info("[API] finish_collection - 采集完成成功")
                        response_data = {
                            "code": 200,
                            "data": self.response_finish_collection['data'],
                            "msg": self.response_finish_collection['msg']
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] finish_collection - 采集完成失败: {self.response_finish_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_finish_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 100:
                    logging.warning("[API] finish_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] finish_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
        
    def discard_collection(self):
        try:
            logging.info("[API] discard_collection - 丢弃采集请求")
            data = request.get_json()
            logging.debug(f"[API] discard_collection - 请求数据: {data}")
            
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'discard_collection'})
            
            while True:
                if 0 < self.response_discard_collection["timestamp"] - now_time < 8:
                    if self.response_discard_collection['msg'] == "success":
                        logging.info("[API] discard_collection - 采集丢弃成功")
                        response_data = {
                            "code": 200,
                            "data": {},
                            "msg": "success"
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] discard_collection - 采集丢弃失败: {self.response_discard_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_discard_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 8:
                    logging.warning("[API] discard_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] discard_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
        
    def submit_collection(self):
        try:
            logging.info("[API] submit_collection - 提交采集请求")
            data = request.get_json()
            logging.debug(f"[API] submit_collection - 请求数据: {data}")
            
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'submit_collection'})
            
            while True:
                if 0 < self.response_submit_collection["timestamp"] - now_time < 5:
                    if self.response_submit_collection['msg'] == "success":
                        logging.info("[API] submit_collection - 采集提交成功")
                        response_data = {
                            "code": 200,
                            "data": {},
                            "msg": "success"
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] submit_collection - 采集提交失败: {self.response_submit_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_submit_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5:
                    logging.warning("[API] submit_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] submit_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def standby(self):
        if 'user_id' not in session:
            logging.warning("[API] standby - 未授权访问")
            return jsonify({"error": "Unauthorized"}), 401
        logging.info("[API] standby - 待命状态")
        pass

    def manual_upload_nas(self):
        try:
            logging.info("[API] manual_upload_nas - 手动上传NAS请求")
            with self.upload_lock:
                if self.upload_nas_flag:
                    logging.warning("[API] manual_upload_nas - 数据上传中")
                    response_data = {
                        "code": 601,
                        "data": {},
                        "msg": '数据上传中'
                    }
                    return jsonify(response_data), 200
                else:
                    self.upload_nas_flag = True
                    if not upload_to_nas.nas_auth.get_auth_sid():
                        logging.error("[API] manual_upload_nas - 连接NAS异常")
                        response_data = {
                            "code": 601,
                            "data": {},
                            "msg": '连接nas异常'
                        }
                        self.upload_nas_flag = False
                        return jsonify(response_data), 200
                    if not self.login():
                        logging.error("[API] manual_upload_nas -网络异常")
                        response_data = {
                            "code": 601,
                            "data": {},
                            "msg": '网络异常'
                        }
                        self.upload_nas_flag = False
                        return jsonify(response_data), 200
                    upload_manual_thread = threading.Thread(target=self.local_to_nas, daemon=True)
                    upload_manual_thread.start()
                    logging.info("[API] manual_upload_nas - 启动上传线程")
                    response_data = {
                        "code": 200,
                        "data": {},
                        "msg": 'success'
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] manual_upload_nas - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500

    # ---------------------------------------upload----------------------------------------------
    def upload_start(self):
        try:
            logging.info("[API] upload_start - 上传开始通知")
            data = request.get_json()
            logging.debug(f"[API] upload_start - 请求数据: {data}")
            
            data['transfer_type'] = 'local_to_nas'
            self.make_request_with_token('eai/dts/upload/start', data)
            logging.info("[API] upload_start - 上传开始通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_start - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def upload_finish(self):
        try:
            logging.info("[API] upload_finish - 上传完成通知")
            data = request.get_json()
            logging.debug(f"[API] upload_finish - 请求数据: {data}")
            
            response_data = {
                "task_id": data["task_id"],                 
                "task_data_id": data["task_data_id"],           
                "transfer_type": "local_to_nas",     
                "status": "SUCCESS" 
            }
            self.make_request_with_token('eai/dts/upload/complete', response_data)
            logging.info("[API] upload_finish - 上传完成通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_finish - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    def upload_fail(self):
        try:
            logging.info("[API] upload_fail - 上传失败通知")
            data = request.get_json()
            logging.debug(f"[API] upload_fail - 请求数据: {data}")
            
            if data.get("expand"):
                response_data = {
                    "task_id": data["task_id"],                 
                    "task_data_id": data["task_data_id"],           
                    "transfer_type": "local_to_nas",     
                    "status": "FAILED",
                    "expand": data['expand'] 
                }
            else:
                response_data = {
                    "task_id": data["task_id"],                 
                    "task_data_id": data["task_data_id"],           
                    "transfer_type": "local_to_nas",     
                    "status": "FAILED",
                    "expand": '{"nas_failed_msg":"网络通讯错误"}' 
                }
            self.make_request_with_token('eai/dts/upload/complete', response_data)
            logging.info("[API] upload_fail - 上传失败通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_fail - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def upload_process(self):
        try:
            logging.info("[API] upload_process - 上传进度通知")
            data = request.get_json()
            logging.debug(f"[API] upload_process - 请求数据: {data}")
            
            data['transfer_type'] = 'local_to_nas'
            self.make_request_with_token('eai/dts/upload/process', data)
            logging.info("[API] upload_process - 上传进度通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_process - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # ---------------------------------------robot------------------------------------------------
    def update_frame(self, stream_id):
        try:
            #logging.info(f"[API] update_frame - 更新帧请求: {stream_id}")
            
            try:
                stream_id = int(stream_id)
            except ValueError:
                #logging.warning(f"[API] update_frame - 无效的流ID: {stream_id}")
                return jsonify({"error": "无效的流ID,必须为数字"}), 400

            if stream_id not in self.video_streams:
                #logging.error(f"[API] update_frame - 无效的视频流ID: {stream_id}")
                return jsonify({"error": "无效的视频流ID"}), 401

            frame_data = request.get_data()
            if not frame_data:
                #logging.error("[API] update_frame - 未接收到帧数据")
                return jsonify({"error": "未接收到帧数据"}), 402
            
            try:
                img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Invalid JPEG data")
            except Exception as e:
                #logging.error(f"[API] update_frame - 无效的帧数据: {str(e)}")
                return jsonify({"error": "无效的帧数据"}), 404

            self.video_streams[stream_id].update_frame(frame_data)
            #logging.info(f"[API] update_frame - 成功更新帧: {stream_id}")
            return jsonify({"msg": "帧已更新"}), 200
        except Exception as e:
            logging.error(f"[API Error] update_frame - 异常: {str(e)}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def robot_get_video_list(self):
        try:
            logging.info("[API] robot_get_video_list - 机器人获取视频列表请求")
            new_list = request.get_json()
            logging.debug(f"[API] robot_get_video_list - 请求数据: {new_list}")
            
            self.video_list = json.loads(new_list)
            logging.debug(f"[API] robot_get_video_list - 更新视频列表: {self.video_list}")
            
            self.video_timestamp = time.time()
            if not self.init_streams_flag:
                self.init_streams()
                self.init_streams_flag = True
            logging.info("[API] robot_get_video_list - 处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] robot_get_video_list - 异常: {str(e)}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def robot_response(self):
        try:
            logging.info("[API] robot_response - 机器人响应")
            data = request.get_json()
            logging.debug(f"[API] robot_response - 响应数据: {data}")
            
            if data["cmd"] == "start_collection":
                self.response_start_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "finish_collection":
                self.response_finish_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"],
                    "data": data["data"]
                }
            elif data["cmd"] == "discard_collection":
                self.response_discard_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "submit_collection":
                self.response_submit_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            logging.info("[API] robot_response - 响应处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] robot_response - 异常: {str(e)}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def get_task_steps(self):
        logging.info("[API] get_task_steps - 获取任务步骤请求")
        logging.debug(f"[API] get_task_steps - 返回数据: {self.task_steps}")
        return jsonify(self.task_steps), 200
    
    def run(self):
        logging.info("[Server] run - 启动服务器")
        self.upload_thread.start()
        self.socketio.run(self.app, host='0.0.0.0', port=8080, debug=False)




if __name__ == '__main__':
    server = FlaskServer()
    server.run()
