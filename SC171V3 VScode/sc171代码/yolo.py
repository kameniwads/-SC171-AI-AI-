#!/usr/bin/env python3
# fibo_mqtt_strawberry_test.py - Fibo开发板MQTT草莓图片测试发布程序
# 用于测试服务器端YOLOv5草莓检测功能

import paho.mqtt.client as mqtt
import base64
import json
import time
import os
import cv2
import numpy as np
from datetime import datetime
import urllib.request
import requests

# ========== MQTT配置 ==========
MQTT_BROKER = "192.168.187.81" 
MQTT_PORT = 1883
MQTT_TOPIC_VIDEO = "/camera/fibo/video"  # 视频流主题
MQTT_TOPIC_STATUS = "/camera/fibo/status"  # 摄像头状态主题
MQTT_CLIENT_ID = "fibo_strawberry_test_publisher"

# ========== 草莓测试图片配置 ==========
# 真实草莓图片URLs
STRAWBERRY_TEST_IMAGES = [
    {
        "name": "real_strawberry_field",
        "url": "https://ts1.tc.mm.bing.net/th/id/R-C.7dc1783556e9f2982bfd184e16b155dd?rik=R1XCR4KNh0uiew&riu=http%3a%2f%2fk.sinaimg.cn%2fn%2fsinacn10110%2f87%2fw1500h987%2f20190614%2fee35-hymscpq0292498.jpg%2fw700d1q75cms.jpg%3fby%3dcms_fixed_width&ehk=nCXye1jNgveawk5Onu%2f%2f%2b6CsbJ6vaOEEMcIaGqIcwLA%3d&risl=&pid=ImgRaw&r=0",
        "description": "真实草莓田 - 包含红色成熟草莓和绿色未成熟草莓，黑色地膜背景"
    }

]

# 只使用下载的真实草莓图片进行测试

class FiboStrawberryTestPublisher:
    def __init__(self):
        self.mqtt_client = None
        self.test_images = []
        self.current_image_index = 0
        self.images_folder = "strawberry_test_images"
        
        # 确保测试图片文件夹存在
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
            print(f" 已创建草莓测试图片文件夹: {self.images_folder}")
    
    def init_mqtt(self):
        """初始化MQTT客户端"""
        print(" 正在初始化MQTT客户端...")
        
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
            # 发送草莓测试状态
            status_data = {
                "device": "fibo_strawberry_test",
                "status": "online",
                "test_mode": True,
                "test_type": "strawberry_detection",
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
    
    def download_strawberry_images(self):
        """下载草莓测试图片"""
        print(" 正在下载草莓测试图片...")
        
        downloaded_images = []
        
        for img_info in STRAWBERRY_TEST_IMAGES:
            try:
                print(f" 下载: {img_info['name']} - {img_info['description']}")
                
                # 设置文件路径
                filename = f"{img_info['name']}.jpg"
                filepath = os.path.join(self.images_folder, filename)
                
                # 如果文件已存在，跳过下载
                if os.path.exists(filepath):
                    print(f"    文件已存在: {filename}")
                    downloaded_images.append({
                        'path': filepath,
                        'name': img_info['name'],
                        'description': img_info['description']
                    })
                    continue
                
                # 下载图片
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(img_info['url'], headers=headers, timeout=15)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    # 验证图片是否有效
                    test_img = cv2.imread(filepath)
                    if test_img is not None:
                        print(f"    下载成功: {filename}")
                        downloaded_images.append({
                            'path': filepath,
                            'name': img_info['name'],
                            'description': img_info['description']
                        })
                    else:
                        print(f"    图片文件损坏: {filename}")
                        if os.path.exists(filepath):
                            os.remove(filepath)
                else:
                    print(f"    下载失败: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"    下载异常: {e}")
        
        if not downloaded_images:
            print(" 未能下载任何草莓图片，请检查网络连接或图片URL")
            return False
        
        self.test_images = downloaded_images
        print(f" 草莓图片准备就绪，共 {len(self.test_images)} 张测试图片")
        
        return len(downloaded_images) > 0
    
    def encode_image_to_base64(self, image_path):
        """将图片编码为base64"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # 添加当前时间戳到图片上
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(img, f" Strawberry Test: {timestamp}", (10, img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 编码为JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encoded_img = cv2.imencode('.jpg', img, encode_param)
            
            if result:
                # 转换为base64
                img_base64 = base64.b64encode(encoded_img).decode('utf-8')
                return img_base64
            else:
                return None
                
        except Exception as e:
            print(f" 图片编码失败: {e}")
            return None
    
    def publish_strawberry_image(self, image_info):
        """发布草莓测试图片到MQTT"""
        try:
            print(f" 发布草莓图片: {image_info['name']}")
            print(f"   描述: {image_info['description']}")
            
            # 编码图片
            frame_base64 = self.encode_image_to_base64(image_info['path'])
            if not frame_base64:
                print("    图片编码失败")
                return False
            
            # 构造MQTT消息
            video_data = {
                "device": "fibo_strawberry_test",
                "frame": frame_base64,
                "timestamp": time.time(),
                "frame_count": self.current_image_index + 1,
                "fps": 1,  # 测试模式固定为1fps
                "resolution": "640x480",
                "test_mode": True,
                "test_type": "strawberry_detection",
                "test_image_name": image_info['name'],
                "test_description": image_info['description']
            }
            
            # 发布到MQTT
            result = self.mqtt_client.publish(
                MQTT_TOPIC_VIDEO, 
                json.dumps(video_data),
                qos=0
            )
            
            if result.rc == 0:
                print(f"    发布成功，数据大小: {len(frame_base64)} 字符")
                return True
            else:
                print(f"    发布失败，错误代码: {result.rc}")
                return False
                
        except Exception as e:
            print(f" 发布异常: {e}")
            return False
    
    def start_strawberry_test_cycle(self, interval=5):
        """开始草莓图片循环发送"""
        if not self.test_images:
            print(" 没有可用的草莓测试图片")
            return
        
        print(f" 开始草莓图片循环发送...")
        print(f" 草莓检测测试配置:")
        print(f"   图片数量: {len(self.test_images)}")
        print(f"   发送间隔: {interval} 秒")
        print(f"   MQTT主题: {MQTT_TOPIC_VIDEO}")
        print(f"   检测目标: 草莓（成熟度、数量、位置）")
        print(f" 操作提示:")
        print(f"   - 按 Ctrl+C 停止测试")
        print(f"   - 查看草莓检测结果: http://localhost:5000")
        
        try:
            cycle_count = 0
            while True:
                # 获取当前测试图片
                current_image = self.test_images[self.current_image_index]
                
                # 发布图片
                success = self.publish_strawberry_image(current_image)
                
                if success:
                    print(f" 第 {cycle_count + 1} 轮草莓测试 - 图片 {self.current_image_index + 1}/{len(self.test_images)}")
                
                # 切换到下一张图片
                self.current_image_index = (self.current_image_index + 1) % len(self.test_images)
                
                # 如果完成一轮，增加轮次计数
                if self.current_image_index == 0:
                    cycle_count += 1
                    print(f" 完成第 {cycle_count} 轮草莓检测测试，开始新一轮...")
                
                # 等待指定间隔
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n 草莓检测测试被用户中断")
            print(f" 测试统计:")
            print(f"   完成轮次: {cycle_count}")
            print(f"   发送草莓图片总数: {cycle_count * len(self.test_images) + self.current_image_index}")
    
    def manual_strawberry_test_mode(self):
        """手动草莓测试模式"""
        if not self.test_images:
            print(" 没有可用的草莓测试图片")
            return
        
        print(f" 手动草莓测试模式")
        print(f" 可用草莓测试图片:")
        for i, img_info in enumerate(self.test_images):
            print(f"   {i+1}. {img_info['name']} - {img_info['description']}")
        
        while True:
            print(f"\n 选择操作:")
            print(f"1-{len(self.test_images)}. 发送对应编号的草莓图片")
            print(f"a. 发送所有草莓图片")
            print(f"q. 退出")
            
            choice = input("请输入选择: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'a':
                print(f" 发送所有草莓测试图片...")
                for img_info in self.test_images:
                    self.publish_strawberry_image(img_info)
                    time.sleep(2)  # 间隔2秒
                print(f" 所有草莓图片发送完成")
            else:
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(self.test_images):
                        self.publish_strawberry_image(self.test_images[index])
                    else:
                        print(f" 无效选择，请输入 1-{len(self.test_images)}")
                except ValueError:
                    print(f" 无效输入，请输入数字")
    
    def stop(self):
        """停止测试程序"""
        print(" 正在停止草莓检测测试...")
        
        # 发送离线状态
        if self.mqtt_client:
            status_data = {
                "device": "fibo_strawberry_test",
                "status": "offline",
                "test_mode": True,
                "test_type": "strawberry_detection",
                "timestamp": time.time()
            }
            self.mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps(status_data))
            time.sleep(0.5)
            
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print(" MQTT客户端已断开")
        
        print(" 草莓检测测试程序已停止")

def main():
    print(" Fibo开发板MQTT草莓检测测试程序")
    print(" 用于测试服务器端YOLOv5草莓检测功能")
    print(" 检测目标：草莓成熟度、数量、位置识别")
    print(" 使用说明：")
    print("   1. 请确保网络连接正常以下载测试图片")
    print("   2. 如需使用自定义草莓图片，请修改STRAWBERRY_TEST_IMAGES中的URL")
    print("   3. 推荐图片分辨率：640x480或更高")
    print("   4. 图片应包含清晰的草莓（成熟/未成熟均可）")
    print(f" MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}")
    print(f" 视频主题: {MQTT_TOPIC_VIDEO}")
    print(f" 状态主题: {MQTT_TOPIC_STATUS}")
    
    # 创建草莓测试发布器
    publisher = FiboStrawberryTestPublisher()
    
    try:
        # 初始化MQTT
        if not publisher.init_mqtt():
            print(" MQTT初始化失败，程序退出")
            return
        
        # 下载草莓测试图片
        if not publisher.download_strawberry_images():
            print(" 草莓测试图片下载失败！")
            print(" 解决方案：")
            print("   1. 检查网络连接是否正常")
            print("   2. 确认图片URL是否有效")
            print("   3. 可以手动下载草莓图片到 strawberry_test_images/ 文件夹")
            print("   4. 修改代码中的STRAWBERRY_TEST_IMAGES，使用本地图片路径")
            return
        
        # 等待MQTT连接稳定
        time.sleep(2)
        
        # 选择测试模式
        print(f"\n 选择草莓检测测试模式:")
        print(f"1. 自动循环模式（推荐）- 持续发送草莓图片")
        print(f"2. 手动测试模式 - 手动选择发送特定草莓图片")
        
        mode = input("请输入选择 (1-2): ").strip()
        
        if mode == '1':
            interval = input("请输入发送间隔秒数 (默认5秒): ").strip()
            try:
                interval = int(interval) if interval else 5
            except:
                interval = 5
            
            publisher.start_strawberry_test_cycle(interval)
        elif mode == '2':
            publisher.manual_strawberry_test_mode()
        else:
            print(" 无效选择，使用默认自动模式")
            publisher.start_strawberry_test_cycle(5)
        
    except Exception as e:
        print(f" 程序运行错误: {e}")
    finally:
        publisher.stop()

if __name__ == "__main__":
    main()