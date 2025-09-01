import os
import datetime
import json
import yaml


def get_today_date():
    # 获取当前日期和时间
    today = datetime.datetime.now()
    
    # 格式化日期为字符串，格式为 "YYYY-MM-DD"
    date_string = today.strftime("%Y%m%d")
    return date_string

def file_size(path,n):
    has_directory = False
    has_file = False
    file_size = 0

    # 获取目录中的所有条目
    pre_entries = os.listdir(path)

    for entry in pre_entries:
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            has_directory = True
        elif os.path.isfile(entry_path):
            has_file = True
        break
    if has_file:
        for file_name in pre_entries:
            # 分割文件名和扩展名
            base, ext = file_name.split(".")
            # 分割前缀和数字部分
            prefix, old_num = base.rsplit("_", 1)
            # 计算数字部分的位数
            num_digits = len(old_num)
            # 格式化新数字，保持位数（用 zfill 补零）
            new_num = str(n).zfill(num_digits)
            # 重新组合文件名
            new_file_name = f"{prefix}_{new_num}.{ext}"
            break
        file_path = os.path.join(path,new_file_name)
        #print(file_path)
        file_size += os.path.getsize(file_path)  # 获取文件大小（字节）
        return file_size

    if has_directory:
        # 遍历子目录，查找第 n 个文件
        for subdir in pre_entries:
            pre_entry =  os.listdir(os.path.join(path,subdir))
            for file_name in pre_entry:
                # 分割文件名和扩展名
                base, ext = file_name.split(".")
                # 分割前缀和数字部分
                prefix, old_num = base.rsplit("_", 1)
                # 计算数字部分的位数
                num_digits = len(old_num)
                # 格式化新数字，保持位数（用 zfill 补零）
                new_num = str(n).zfill(num_digits)
                # 重新组合文件名
                new_file_name = f"{prefix}_{new_num}.{ext}"
                break
            
            file_path = os.path.join(path,subdir,new_file_name)
            #print(file_path)
            file_size += os.path.getsize(file_path)  # 获取文件大小（字节）
        return file_size
                                

def data_size(fold_path, data): # 文件大小单位(MB)
    try:
        size_bytes = 0
        directory_path = os.path.join(fold_path, get_today_date())
        print(directory_path)
        #directory_path = os.path.join(fold_path1, "2025701")
        if not os.path.exists(directory_path):
            return 500
        task_path = os.path.join(directory_path,f"{str(data['task_name'])}_{str(data['task_id'])}")
        opdata_path = os.path.join(task_path,"meta","op_dataid.jsonl")
        with open(opdata_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["dataid"] == str(data["task_data_id"]):
                        episode_index = json_object_data["episode_index"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        
        entries_1 = os.listdir(task_path) 
        for entry in entries_1:
            if entry == "meta":
                continue
            data_path = os.path.join(task_path,entry,"chunk-000")
            size_bytes += file_size(data_path,episode_index)
        size_mb = round(size_bytes / (1024 * 1024),2)
        return size_mb


    except Exception as e:
        print(str(e))
        return 500
    




def data_duration(fold_path,data):  # 文件时长单位(s)
    try:
        directory_path = os.path.join(fold_path, get_today_date())
        print(directory_path)
        #directory_path = os.path.join(fold_path1, "2025701")
        if not os.path.exists(directory_path):
            return 30
        task_path = os.path.join(directory_path,f"{str(data['task_name'])}_{str(data['task_id'])}")
        info_path = os.path.join(task_path,"meta","info.json")
        opdata_path = os.path.join(task_path,"meta","op_dataid.jsonl")
        episodes_path = os.path.join(task_path,"meta","episodes.jsonl")
        with open(info_path,"r",encoding="utf-8") as f:
            info_data = json.load(f)
            fps = info_data["fps"] # 
        with open(opdata_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["dataid"] == str(data["task_data_id"]):
                        episode_index = json_object_data["episode_index"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        with open(episodes_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["episode_index"] == episode_index:
                        length= json_object_data["length"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        duration = round(length/fps,2)
        return duration        
    except Exception as e:
        print(str(e))
        return 30

def setup_from_yaml():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, 'setup.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return config_dict
    
def get_machine_info():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    machine_txt = os.path.join(script_dir, 'machine_information.json')
    if os.path.exists(machine_txt):
        with open(machine_txt, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return config_dict
        

# import os
# import pandas as pd
# import numpy as np
# import cv2
# from pathlib import Path
# from collections import defaultdict

# class LerobotDataValidator:
#     def __init__(self, meta_dir = "/home/liuyou/Documents/data/叠衣服_0707_183/meta",data_dir="/home/liuyou/Documents/data/叠衣服_0707_183/data", images_dir="/home/liuyou/Documents/data/叠衣服_0707_183/images"):
#         self.data_dir = Path(data_dir)
#         self.images_dir = Path(images_dir)
#         self.meta_dir = Path(meta_dir)
#         # 配置参数
#         self.action_window = 5  # 动作数据校验窗口大小
#         self.image_sample_interval = 30  # 图像抽样间隔
#         self.max_static_frames = 10  # 允许的最大连续静止帧数
#         self.action_change_threshold = 0.1  # 动作变化阈值（可根据实际情况调整）
#         self.image_change_threshold = 0.95  # 图像相似度阈值（0-1）
#         self.joint_change_thresholds = np.array([0.1] * 26)  # 默认突变阈值
#         self.max_duplicate_ratio = 0.3  # 允许的最大重复比例
#         self.episodes_stats = "episodes_stats.jsonl"
#         self.fps = 30

#     def calculate_thresholds(self, episode_id):
#         # 读取JSON文件（逐行查找指定episode_id）
#         target_json = None
#         with open(self.meta_dir / self.episodes_stats, 'r') as file:
#             for line in file:
#                 try:
#                     json_object = json.loads(line.strip())
#                     if json_object["episode_index"] == episode_id:
#                         target_json = json_object
#                         break
#                 except json.JSONDecodeError as e:
#                     print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
#                     continue  # 跳过错误行，继续查找
    
#         if target_json is None:
#             raise ValueError(f"未找到 episode_id = {episode_id} 的数据！")
    
#         # 提取 max 和 min，并计算范围（保留原始精度）
#         max_vals = np.array(target_json['stats']['action']['max'], dtype=np.float32)  # 强制高精度浮点
#         min_vals = np.array(target_json['stats']['action']['min'], dtype=np.float32)
        
#         # 检查维度一致性
#         if len(max_vals) != len(min_vals):
#             raise ValueError(f"max 和 min 的维度不一致！max: {len(max_vals)}, min: {len(min_vals)}")
    
#         # 计算阈值： (max - min) 的 10%，保留绝对值
#         action_range = max_vals - min_vals
#         self.joint_change_thresholds = np.abs(action_range) * 0.2  # 阈值始终为正
#         self.joint_change_thresholds = np.where(
#             self.joint_change_thresholds == 0,  # 条件：等于0
#             1e-9,                               # 替换值
#             self.joint_change_thresholds       # 保留原值
#         )
    
#         # 打印阈值（默认完整精度）
#         print("Joint Change Thresholds (20% of action range):")
#         print(self.joint_change_thresholds)
    
#         return self.joint_change_thresholds  # 返回阈值供外部使用
            
        
    
#     def validate_session(self, session_id):
#         """验证单个会话的数据"""
#         print(f"Validating session: {session_id}")
#         print("计算得到的突变阈值:")
#         episode_id = int(session_id.split("_")[1])
#         self.calculate_thresholds(episode_id)
#         # 1. 加载数据
#         parquet_path = self.data_dir /"chunk-000" /f"{session_id}.parquet"
#         img_session_dir = self.images_dir
        
#         if not parquet_path.exists():
#             raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
#         if not img_session_dir.exists():
#             raise FileNotFoundError(f"Images directory not found: {img_session_dir}")
        
#         df = pd.read_parquet(parquet_path)
#         # 1. 基本校验
#         self._validate_timestamps(df)
#         # 2. 动作数据校验
#         self._validate_action_data(df)
#         camera_dirs = [d for d in img_session_dir.glob("*") if d.is_dir()]
#         if not camera_dirs:
#             raise FileNotFoundError(f"未找到相机目录: {img_session_dir}")
#         for camera_dir in camera_dirs:
#             camera_name = os.path.basename(camera_dir)
#             print(camera_name)
#             camera_dir = camera_dir / session_id # 例如"image_left"
#             img_files = sorted(camera_dir.glob("frame_*.png"), 
#                             key=lambda x: int(x.stem.split("_")[-1]))
        
#             # # 3. 基本校验
#             self._validate_frame_count(df, img_files)
#             # 4. 图像数据校验
#             self._validate_image_data(img_files, camera_name)
        
       
#         print(f"✅ Session {session_id} validation passed")
    
#     def _validate_frame_count(self, df, img_files):
#         """校验动作数据和图像帧数是否一致"""
#         if len(df) != len(img_files):
#             raise ValueError(
#                 f"Frame count mismatch: {len(df)} action frames vs "
#                 f"{len(img_files)} image frames"
#             )
#     def _validate_timestamps(self, df):
#         """校验时间戳是否符合固定帧率（如30FPS）"""
#         timestamps = df['timestamp'].values.astype(np.float64)  # 避免浮点精度问题
        
#         # 1. 检查单调性
#         if not np.all(np.diff(timestamps) >= 0):
#             error_msg = '检测到动作时间戳不单调递增'
#             print(error_msg)
#             return False,error_msg
           
        
#         # 2. 检查固定帧率（允许微小误差，如±1%）
#         expected_interval = 1.0 / self.fps
#         actual_intervals = np.diff(timestamps)
        
#         # 计算相对误差
#         tolerance = 0.01  # 1% 容差
#         relative_error = np.abs(actual_intervals - expected_interval) / expected_interval
        
#         if np.any(relative_error > tolerance):
#             bad_idx = np.where(relative_error > tolerance)[0][0]  # 第一个错误位置
#             error_msg = (
#                 f"帧 {bad_idx} 的时间戳间隔不符合预期帧率 {self.fps} FPS。\n"
#                 f"预期间隔: {expected_interval:.6f} 秒，\n"
#                 f"实际间隔: {actual_intervals[bad_idx]:.6f} 秒"
#             )
#             print(error_msg)
#             return False, error_msg  # 返回 False 和中文错误信息
    
#     def _validate_action_data(self, df):
#         """校验动作数据质量"""
#         if 'action' not in df.columns:
#             raise ValueError("Parquet file missing 'action' column")
        
#         action_data = np.stack(df['action'].values)  # shape: [n_samples, 26]
#         print(f"Action data shape: {action_data.shape}")

#         # ===== 检查是否存在全零帧 =====
#         zero_frames = np.where(np.all(action_data == 0, axis=1))[0]  # 获取全零帧的索引
#         if len(zero_frames) > 0:
#             error_msg = f"检测到 {len(zero_frames)} 帧全零数据（无效动作），位于索引: {list(zero_frames)}"
#             print(error_msg)
#             return False, error_msg


#         # ===== 2. 校验相邻帧突变 =====
#         joint_change_thresholds = self.joint_change_thresholds
#         diff = np.abs(action_data[1:]) - np.abs(action_data[:-1])  # 绝对差值
        
        
#         # 标记超出阈值的维度
#         violations = diff > joint_change_thresholds

#         # 检查是否存在突变
#         n_violations_per_frame = np.sum(violations, axis=1)
#         if np.any(n_violations_per_frame > 0):
#             first_violation_frame = np.where(n_violations_per_frame > 0)[0][0] + 1
#             problematic_frame_idx = first_violation_frame
#             prev_frame_data = action_data[problematic_frame_idx - 1]
#             curr_frame_data = action_data[problematic_frame_idx]
#             exceeding_dims = np.where(violations[problematic_frame_idx - 1])[0]

#             # 构造错误信息（高亮旋转维度的特殊处理）
#             error_msg = (
#                 f"Sudden joint change detected at frame {problematic_frame_idx}.\n"
#                 f"Dimensions exceeding threshold ({joint_change_thresholds[exceeding_dims]}): {exceeding_dims}\n"
#                 f"Previous frame ({problematic_frame_idx-1}) data: {prev_frame_data}\n"
#                 f"Current frame ({problematic_frame_idx}) data: {curr_frame_data}\n"
#                 f"Absolute differences (rotation dims treated as absolute): {diff[problematic_frame_idx - 1]}"
#             )
#             print(error_msg)
#             error_msg = f"检测到 {problematic_frame_idx} 帧发生突变（无效动作），位于索引: {exceeding_dims}"
#             print(error_msg)
#             return False, error_msg

#         # ===== 3. 统计重复数据比例 =====
#         unique_actions, counts = np.unique(action_data, axis=0, return_counts=True)
#         duplicate_ratio = 1 - (len(unique_actions) / len(action_data))
#         print(f"duplicate action ratio ({duplicate_ratio:.3%})")
#         most_common_action = unique_actions[np.argmax(counts)]
#         max_count = max(counts)
#         most_common_ratio = max_count / len(action_data)  
#         print(f"High duplicate action ratio ({most_common_ratio:.3%})")
#         if most_common_ratio > 0.9:
#             print(f"Warning: High duplicate action ratio ({most_common_ratio:.1%})")
#             most_common_action = unique_actions[np.argmax(counts)]
#             max_count = max(counts)
#             print(f"Most common action (appears {max_count} times):\n{most_common_action}")
    
#             # ===== 新增：记录重复帧的分布 =====
#             # 找到所有等于 most_common_action 的帧索引
#             repeated_frames = np.where((action_data == most_common_action).all(axis=1))[0]
#             print(f"Repeated frames (total {len(repeated_frames)}): {repeated_frames}")
    
#             # 可选：检查是否连续重复
#             diff_frames = np.diff(repeated_frames)
#             consecutive_blocks = np.split(repeated_frames, np.where(diff_frames > 1)[0] + 1)
#             print(f"Consecutive blocks: {[list(block) for block in consecutive_blocks]}")
#             error_msg = f"检测到{most_common_ratio:.3%}帧动作重复"
#             return False, error_msg
#         # ===== 新增：记录重复帧的分布 =====
#         # 找到所有等于 most_common_action 的帧索引
#         repeated_frames = np.where((action_data == most_common_action).all(axis=1))[0]
#         print(f"Repeated frames (total {len(repeated_frames)}): {repeated_frames}")

#         success_msg = f"检测到{most_common_ratio:.3%}帧动作重复"
#         return True, success_msg
    
#     def _validate_image_data(self, img_files, camera_dir):
#         """优化后的图像校验：通过抽样检测图像变化"""
#         if not img_files:
#             error_msg = f"检测到{camera_dir}没有图像数据"
#             print(error_msg)
#             return False,error_msg
    
#         # 1. 抽样检查图像变化（每30帧检查一次）
#         sample_indices = list(range(0, len(img_files), self.image_sample_interval))
#         if len(img_files) - 1 not in sample_indices:
#             sample_indices.append(len(img_files) - 1)  # 确保检查最后一帧
    
#         similar_pairs = []
#         for i in range(len(sample_indices) - 1):
#             idx1 = sample_indices[i]
#             idx2 = sample_indices[i + 1]
#             img1 = cv2.imread(str(img_files[idx1]))
#             img2 = cv2.imread(str(img_files[idx2]))
    
#             if img1 is None or img2 is None:
#                 error_msg = f"检测到无法读取{camera_dir}图像数据"
#                 print(error_msg)
#                 return False,error_msg
        
#             # 计算图像相似度（直方图比较）
#             similarity = self._compare_images(img1, img2)
#             if similarity > self.image_change_threshold:
#                 similar_pairs.append((idx1, idx2, similarity))
    
#         # 2. 分析抽样结果
#         if len(similar_pairs) > len(sample_indices) * 0.5:  # 如果超过50%的抽样对相似
#             # 进一步检查最近邻图像是否完全相同（覆盖相机故障场景）
#             first_img = cv2.imread(str(img_files[0]))
#             all_same = True
#             for img_file in img_files[1:]:
#                 img = cv2.imread(str(img_file))
#                 if not np.array_equal(first_img, img):
#                     all_same = False
#                     break
#             if all_same:
#                 error_msg = f"检测到{camera_dir}部分图像数据完全相同,"
#                 print(error_msg)
#                 return False,error_msg
#             else:
#                 # 如果不是完全相同，但抽样中存在大量相似对，可能是动作缓慢或场景未变化
#                 warning_msg = (
#                     f"Warning: High image similarity detected in {len(similar_pairs)}/"
#                     f"{len(sample_indices)} samples (threshold={self.image_change_threshold}). "
#                     "This may indicate slow motion or insufficient scene changes."
#                 )
#                 print(warning_msg)  # 仅警告，不中断流程
    
#         # 3. 检查首尾图像差异（确保数据采集正常启动和结束）
#         first_img = cv2.imread(str(img_files[0]))
#         last_img = cv2.imread(str(img_files[-1]))
#         end_similarity = self._compare_images(first_img, last_img)
#         # 如果首尾完全相同（相似度 >= 1.0，假设 _compare_images 返回 [0,1] 范围内的值）
#         if end_similarity >= 1:  # 或使用一个接近 1.0 的阈值（如 0.999）以避免浮点误差
#             error_msg = f"检测到{camera_dir}首尾图像完全相同"
#             print(error_msg)
#             return False, error_msg
        
#         if end_similarity > self.image_change_threshold:  # 容忍更低的变化阈值
#             print(
#                 f"Note: First and last images are very similar (similarity={end_similarity:.4f}). "
#                 "Verify if the session captured intended motion."
#             )
            
#     def _compare_images(self, img1, img2):
#         """简单比较两张图像的相似度"""
#         # 转换为灰度图
#         gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
#         # 计算直方图
#         hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
#         hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
#         # 比较直方图
#         similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
#         return similarity


# # 使用示例
# if __name__ == "__main__":
#     validator = LerobotDataValidator()
    
#     # 验证单个会话
#     try:
#         validator.validate_session("episode_000005")
#     except Exception as e:
#         print(f"❌ Validation failed: {str(e)}")
    
#     # 可以批量验证多个会话
#     # for session_dir in validator.images_dir.iterdir():
#     #     if session_dir.is_dir():
#     #         session_id = session_dir.name
#     #         try:
#     #             validator.validate_session(session_id)
#     #         except Exception as e:
#     #             print(f"❌ Session {session_id} failed: {str(e)}")







