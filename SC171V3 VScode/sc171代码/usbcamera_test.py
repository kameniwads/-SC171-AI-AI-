#!/usr/bin/env python3
# fibo_mqtt_video.py - Fibo开发板MQTT视频发布程序

import cv2
import paho.mqtt.client as mqtt
import base64
import json
import time
import threading
import os
from datetime import datetime

# ========== MQTT配置 ==========
MQTT_BROKER = "192.168.187.81"  # MQTT服务器地址
MQTT_PORT = 1883
MQTT_TOPIC_VIDEO = "/camera/fibo/video"  # 视频流主题
MQTT_TOPIC_STATUS = "/camera/fibo/status"  # 摄像头状态主题
MQTT_CLIENT_ID = "fibo_camera_publisher"

# ========== 摄像头配置 ==========
CAMERA_DEVICE = 2  # 使用/dev/video2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 21  # 目标帧率（MQTT传输建议较低帧率）
JPEG_QUALITY = 70  # JPEG压缩质量 (1-100)

CAMERA_BRIGHTNESS = 255    # 亮度 (0-255, 默认128)
CAMERA_CONTRAST = 255   # 对比度 (0-255, 默认128)  
CAMERA_SATURATION = 255   # 饱和度 (0-255, 默认128)

class FiboMQTTCamera:
    def __init__(self):
        self.running = False
        self.camera = None
        self.mqtt_client = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0
        
        # 确保Camera文件夹存在
        self.camera_folder = self.ensure_camera_folder()
        
    def ensure_camera_folder(self):
        """确保Camera文件夹存在"""
        folder_path = "Camera"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f" 已创建文件夹: {folder_path}")
        return folder_path
    
    def init_camera(self):
        """初始化摄像头"""
        print("📹 正在初始化摄像头...")
        
        # 尝试多种方式打开摄像头
        self.camera = cv2.VideoCapture(CAMERA_DEVICE)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(f'/dev/video{CAMERA_DEVICE}')
        
        if not self.camera.isOpened():
            raise Exception(f" 无法打开摄像头设备 {CAMERA_DEVICE}")
        
        # 设置摄像头基本参数
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        # 设置摄像头图像参数
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, CAMERA_BRIGHTNESS)
        self.camera.set(cv2.CAP_PROP_CONTRAST, CAMERA_CONTRAST)
        self.camera.set(cv2.CAP_PROP_SATURATION, CAMERA_SATURATION)
        
        # 测试读取
        ret, frame = self.camera.read()
        if not ret or frame is None:
            raise Exception(" 摄像头无法获取画面")
        
        # 获取实际设置的参数
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_brightness = int(self.camera.get(cv2.CAP_PROP_BRIGHTNESS))
        actual_contrast = int(self.camera.get(cv2.CAP_PROP_CONTRAST))
        actual_saturation = int(self.camera.get(cv2.CAP_PROP_SATURATION))
        
        print(f" 摄像头初始化成功")
        print(f"   分辨率: {actual_width}x{actual_height}")
        print(f"   目标FPS: {TARGET_FPS}")
        print(f"   亮度: {actual_brightness} (设置值: {CAMERA_BRIGHTNESS})")
        print(f"   对比度: {actual_contrast} (设置值: {CAMERA_CONTRAST})")
        print(f"   饱和度: {actual_saturation} (设置值: {CAMERA_SATURATION})")
        
        # 如果设置的值与实际值不同，给出提示
        if actual_brightness != CAMERA_BRIGHTNESS:
            print(f"  亮度设置可能不完全生效")
        if actual_contrast != CAMERA_CONTRAST:
            print(f"  对比度设置可能不完全生效")
        if actual_saturation != CAMERA_SATURATION:
            print(f"  饱和度设置可能不完全生效")
            
        # 检查摄像头支持的参数
        print(" 摄像头当前参数:")
        properties = [
            (cv2.CAP_PROP_BRIGHTNESS, "亮度"),
            (cv2.CAP_PROP_CONTRAST, "对比度"),
            (cv2.CAP_PROP_SATURATION, "饱和度"),
            (cv2.CAP_PROP_HUE, "色调"),
            (cv2.CAP_PROP_GAIN, "增益"),
            (cv2.CAP_PROP_EXPOSURE, "曝光"),
        ]
        
        for prop, name in properties:
            current_value = self.camera.get(prop)
            print(f"   {name}: {current_value}")
        
    def init_mqtt(self):
        """初始化MQTT客户端"""
        print(" 正在初始化MQTT客户端...")
        
        # 创建MQTT客户端
        try:
            if hasattr(mqtt, 'CallbackAPIVersion'):
                self.mqtt_client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION1, 
                    client_id=MQTT_CLIENT_ID
                )
            else:
                self.mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        except:
            self.mqtt_client = mqtt.Client()
        
        # 设置回调函数
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        self.mqtt_client.on_publish = self.on_mqtt_publish
        
        # 如果配置了认证
        if 'MQTT_USERNAME' in globals() and 'MQTT_PASSWORD' in globals():
            self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            print(f" 使用认证: {MQTT_USERNAME}")
        
        # 连接MQTT服务器
        try:
            print(f" 连接MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            return True
        except Exception as e:
            print(f" MQTT连接失败: {e}")
            return False
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            print(" MQTT连接成功")
            # 发送摄像头状态
            status_data = {
                "device": "fibo_camera",
                "status": "online",
                "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                "fps": TARGET_FPS,
                "timestamp": time.time()
            }
            client.publish(MQTT_TOPIC_STATUS, json.dumps(status_data))
        else:
            print(f" MQTT连接失败，错误代码: {rc}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT断开回调"""
        if rc != 0:
            print(f" MQTT意外断开: {rc}")
        else:
            print(" MQTT正常断开")
    
    def on_mqtt_publish(self, client, userdata, mid):
        """MQTT发布回调（可选，用于调试）"""
        pass
    
    def encode_frame(self, frame):
        """编码视频帧为base64"""
        try:
            # 添加时间戳和信息到画面
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Fibo Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, timestamp, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.actual_fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 编码为JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            
            if result:
                # 转换为base64
                frame_base64 = base64.b64encode(encoded_img).decode('utf-8')
                return frame_base64
            else:
                return None
                
        except Exception as e:
            print(f" 帧编码失败: {e}")
            return None
    
    def publish_frame(self, frame_base64):
        """发布视频帧到MQTT"""
        try:
            video_data = {
                "device": "fibo_camera",
                "frame": frame_base64,
                "timestamp": time.time(),
                "frame_count": self.frame_count,
                "fps": self.actual_fps,
                "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
            }
            
            # 发布到MQTT
            result = self.mqtt_client.publish(
                MQTT_TOPIC_VIDEO, 
                json.dumps(video_data),
                qos=0  # 使用QoS 0以获得最佳性能
            )
            
            return result.rc == 0
            
        except Exception as e:
            print(f" 发布帧失败: {e}")
            return False
    
    def calculate_fps(self):
        """计算实际FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.actual_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def save_frame(self, frame):
        """保存帧到本地（可选）"""
        try:
            filename = f"mqtt_frame_{int(time.time())}.jpg"
            filepath = os.path.join(self.camera_folder, filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f" 保存帧失败: {e}")
            return None
    
    def start_streaming(self):
        """开始视频流"""
        print(" 开始MQTT视频流...")
        print(f" 视频主题: {MQTT_TOPIC_VIDEO}")
        print(f" 状态主题: {MQTT_TOPIC_STATUS}")
        print(" 按 Ctrl+C 停止")
        print("-" * 50)
        
        self.running = True
        frame_interval = 1.0 / TARGET_FPS
        
        try:
            while self.running:
                start_time = time.time()
                
                # 读取摄像头帧
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print(" 无法读取摄像头画面")
                    time.sleep(0.1)
                    continue
                
                # 编码帧
                frame_base64 = self.encode_frame(frame)
                if frame_base64:
                    # 发布到MQTT
                    if self.publish_frame(frame_base64):
                        self.frame_count += 1
                    
                    # 计算FPS
                    self.calculate_fps()
                    
                    # 显示状态
                    if self.frame_count % (TARGET_FPS * 5) == 0:  # 每5秒显示一次
                        print(f" 已发送 {self.frame_count} 帧，实际FPS: {self.actual_fps:.1f}")
                
                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n 收到停止信号...")
        except Exception as e:
            print(f" 视频流错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止视频流"""
        print(" 正在停止视频流...")
        self.running = False
        
        # 发送离线状态
        if self.mqtt_client:
            status_data = {
                "device": "fibo_camera",
                "status": "offline",
                "timestamp": time.time()
            }
            self.mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps(status_data))
            time.sleep(0.5)  # 等待消息发送
            
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print(" MQTT客户端已断开")
        
        if self.camera:
            self.camera.release()
            print(" 摄像头已释放")
        
        print(" 视频流已停止")

def main():
    print(" Fibo开发板MQTT视频发布程序")
    print(f" MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}")
    print(f" 摄像头设备: /dev/video{CAMERA_DEVICE}")
    print(f" 目标FPS: {TARGET_FPS}")
    print(f" 分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f" 亮度: {CAMERA_BRIGHTNESS}")
    print(f" 对比度: {CAMERA_CONTRAST}")
    print(f" 饱和度: {CAMERA_SATURATION}")
    
    # 创建并启动摄像头
    fibo_camera = FiboMQTTCamera()
    
    try:
        # 初始化摄像头
        fibo_camera.init_camera()
        
        # 初始化MQTT
        if not fibo_camera.init_mqtt():
            print(" MQTT初始化失败，程序退出")
            return
        
        # 等待MQTT连接稳定
        time.sleep(2)
        
        # 开始视频流
        fibo_camera.start_streaming()
        
    except Exception as e:
        print(f" 程序启动失败: {e}")
    finally:
        fibo_camera.stop()

if __name__ == "__main__":
    main()