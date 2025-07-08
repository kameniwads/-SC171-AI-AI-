# server_yolov5_strawberry_monitor.py - 增强版草莓识别监控系统 (本地YOLOv5修改版)
import warnings
import sys
import os
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 添加本地YOLOv5路径
YOLOV5_PATH = Path(__file__).parent.parent / "yolov5-master"
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH))

try:
    import paho.mqtt.client as mqtt
    import json
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    import time
    import base64
    from datetime import datetime, timedelta
    import random
    import re
    import traceback
    import cv2
    import numpy as np
    import requests
    import threading
    from concurrent.futures import ThreadPoolExecutor

    # YOLOv5相关导入
    YOLOV5_AVAILABLE = False
    try:
        import torch
        # 导入本地YOLOv5模块
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression, scale_boxes, xyxy2xywh, check_img_size
        from utils.torch_utils import select_device

        YOLOV5_AVAILABLE = True
        print(" 本地YOLOv5依赖可用")
    except ImportError as e:
        print(f" 本地YOLOv5依赖不可用: {e}")
        try:
            import torch

            YOLOV5_AVAILABLE = True
            print(" 基础YOLOv5依赖可用")
        except ImportError:
            print("️ YOLOv5依赖不可用，将使用模拟检测")

    # YAML解析（用于读取草莓类别配置）
    try:
        import yaml

        YAML_AVAILABLE = True
        print(" YAML解析可用")
    except ImportError:
        print(" PyYAML未安装，将使用默认草莓类别")
        YAML_AVAILABLE = False

except ImportError as e:
    print(f" 缺少依赖包: {e}")
    print("请运行: pip install paho-mqtt flask flask-socketio opencv-python requests PyYAML")
    if 'torch' in str(e):
        print("如需YOLOv5功能，请安装: pip install torch ultralytics")
    exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'strawberry-yolov5-deepseek-test-2024'
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# 添加CORS支持
try:
    from flask_cors import CORS

    CORS(app)
    print(" CORS支持已启用")
except ImportError:
    print(" flask-cors未安装，如遇跨域问题请安装: pip install flask-cors")
    pass

# 全局变量
sensor_data = {
    'temperature': {'value': None, 'unit': '°C', 'timestamp': None, 'status': 'waiting'},
    'humidity': {'value': None, 'unit': '%', 'timestamp': None, 'status': 'waiting'},
    'light': {'value': None, 'unit': 'lux', 'timestamp': None, 'status': 'waiting'}
}

# 视频流相关变量
video_data = {
    'current_frame': None,
    'device_status': 'offline',
    'last_frame_time': None,
    'frame_count': 0,
    'fps': 0,
    'resolution': None,
    'test_mode': False,
    'current_image_filename': None
}

# 通用YOLOv5检测结果相关变量
detection_data = {
    'last_detections': [],
    'detection_count': 0,
    'detection_history': [],
    'detection_stats': {
        'total_detections': 0,
        'object_counts': {},  # 累计物体类别统计
        'current_objects': {},  # 当前帧物体类别统计
        'last_update': None,
        'processing_time': 0,
        'category_distribution': {}  # 类别分布百分比
    }
}

#  草莓模型相关变量
strawberry_model = None
strawberry_model_path = "./best.pt"  # 草莓模型权重路径
strawberry_yaml_path = "./stra.yaml"  # 草莓模型配置路径

#  本地YOLOv5相关变量
local_yolo_device = None
local_yolo_imgsz = 640


#  修复：改进类别加载函数
def load_strawberry_classes():
    """从stra.yaml文件中加载草莓类别名称 - 修复版"""
    try:
        # 首先尝试从yolov5-master/data目录加载
        yaml_paths = [
            YOLOV5_PATH / "data/stra.yaml",
            Path(strawberry_yaml_path),
            Path("./stra.yaml")
        ]

        for yaml_path in yaml_paths:
            if os.path.exists(yaml_path) and YAML_AVAILABLE:
                import yaml
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if 'names' in config:
                        classes = config['names']
                        # 确保类别是列表格式
                        if isinstance(classes, dict):
                            # 如果是字典格式，按索引排序
                            max_idx = max(classes.keys()) if classes else 0
                            classes = [classes.get(i, f'Class_{i}') for i in range(max_idx + 1)]
                        print(f" 从{yaml_path}加载草莓类别: {classes}")
                        return classes
                    else:
                        print(f"⚠️ {yaml_path}中未找到'names'字段")
    except Exception as e:
        print(f" 加载草莓类别配置失败: {e}")

    # 默认类别（与您的stra.yaml保持一致）
    default_classes = ['Nearly Ripe', 'Ripe', 'Rotten', 'Unripe']
    print(f" 使用默认草莓类别: {default_classes}")
    return default_classes


#  全局草莓类别列表 - 移到这里，确保在init_strawberry_data()之前定义
STRAWBERRY_CLASSES = load_strawberry_classes()

#  修复：添加全局变量存储模型类别
STRAWBERRY_MODEL_CLASSES = []


#  草莓专用检测结果变量
def init_strawberry_data():
    """初始化草莓数据结构"""
    ripeness_counts = {}
    current_ripeness = {}

    #  根据动态加载的类别初始化
    for class_name in STRAWBERRY_CLASSES:
        ripeness_counts[class_name] = 0
        current_ripeness[class_name] = 0

    return {
        'last_detections': [],
        'detection_count': 0,
        'detection_history': [],
        'detection_stats': {
            'total_detections': 0,
            'ripeness_counts': ripeness_counts,  #  累计统计
            'current_ripeness': current_ripeness,  #  当前帧统计
            'last_update': None,
            'processing_time': 0,
            'harvest_ready': 0,  #  累计可收获数量
            'quality_score': 0,  #  累计质量评分
            'current_harvest_ready': 0,  #  当前帧可收获数量
            'current_quality_score': 0  #  当前帧质量评分
        },
        'enabled': False,  # 是否启用草莓检测
        'model_loaded': False
    }


strawberry_data = init_strawberry_data()

# DeepSeek R1 AI相关变量
deepseek_data = {
    'enabled': False,
    'api_url': 'http://localhost:11434',
    'model_name': 'deepseek-r1:1.5b',
    'status': 'disconnected',
    'chat_history': [],
    'last_response': None,
    'response_time': 0,
    'total_messages': 0,
    'error_count': 0
}

message_count = 0
mqtt_client = None
start_time = time.time()
ai_suggestions = []

# 服务器端YOLOv5模型
yolov5_model = None
detection_enabled = False

# 线程池执行器
executor = ThreadPoolExecutor(max_workers=4)  # 增加线程数支持并行检测

# ========== MQTT 配置区域 ==========
MQTT_BROKER = "192.168.187.81"  # 你的MQTT服务器地址
MQTT_PORT = 1883

# MQTT主题配置
MQTT_TOPICS = [
    "/sensor/temperature",  # 温度传感器
    "/sensor/humidity",  # 湿度传感器
    "/sensor/light",  # 光照传感器
    "/camera/fibo/video",  # Fibo图片流
    "/camera/fibo/status"  # Fibo状态
]

MQTT_KEEPALIVE = 60

# ========== YOLOv5配置 ==========
YOLO_MODEL_NAME = "yolov5s"  # 通用模型
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
DETECTION_SAVE_PATH = "detection_results"  # 检测结果保存路径

# 草莓检测配置
STRAWBERRY_CONF_THRESHOLD = 0.5  # 草莓检测阈值（较低以检测更多草莓）
STRAWBERRY_IOU_THRESHOLD = 0.45
STRAWBERRY_SAVE_PATH = "strawberry_results"  # 草莓检测结果保存路径

# ========== DeepSeek R1 配置 ==========
DEEPSEEK_API_URL = "http://localhost:11434"
DEEPSEEK_MODEL = "deepseek-r1:1.5b"
DEEPSEEK_TIMEOUT = 30  # API超时时间（秒）
DEEPSEEK_MAX_HISTORY = 20  # 最大对话历史记录数

# ========== 标注配置 ==========
DRAW_ANNOTATIONS = True  # 是否在图像上绘制检测框
ANNOTATION_THICKNESS = 2  # 边界框线条粗细
TEXT_SCALE = 0.6  # 文字大小
TEXT_THICKNESS = 2  # 文字粗细

# 草莓检测框颜色配置
STRAWBERRY_COLORS = {
    'Nearly Ripe': (0, 255, 255),  # 黄色
    'Ripe': (0, 255, 0),  # 绿色
    'Rotten': (0, 0, 255),  # 红色
    'Unripe': (255, 0, 0),  # 蓝色
}

# 草莓种植最适参数范围
OPTIMAL_RANGES = {
    'temperature': {'min': 18, 'max': 25, 'optimal': 22},
    'humidity': {'min': 60, 'max': 80, 'optimal': 70},
    'light': {'min': 15000, 'max': 35000, 'optimal': 25000}
}


def print_header():
    print("🍓" * 50)
    print("🤖 增强版草莓识别监控系统 (本地YOLOv5修改版)")
    print("🍓" * 50)
    print(f"📡 MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"📋 监控主题: {', '.join(MQTT_TOPICS)}")
    print(f"🌐 Web访问地址: http://localhost:5000")
    print("🤖 通用YOLOv5物体检测")
    print("🍓 本地YOLOv5草莓成熟度检测（4类别）")
    print("🧠 DeepSeek R1 AI智能对话助手")
    print("📷 支持图片数据集测试模式")
    print("📊 支持传感器数据监控")
    print("🔧 使用本地YOLOv5引擎")
    print("🍓" * 50)


# ========== 🍓 本地YOLOv5草莓模型相关函数 (修改版) ==========
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """图像letterbox预处理"""
    # 调整图像大小和填充以满足stride-multiple约束
    shape = im.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大(用于测试时的更好性能)
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度，高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度，高度比例

    dw /= 2  # 分配到两侧的填充
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return im, ratio, (dw, dh)


def init_strawberry_model():
    """初始化本地YOLOv5草莓检测模型 - 修改版"""
    global strawberry_model, strawberry_data, STRAWBERRY_MODEL_CLASSES, local_yolo_device, local_yolo_imgsz

    if not YOLOV5_AVAILABLE:
        print(" 本地YOLOv5不可用，草莓检测将使用模拟模式")
        strawberry_data['enabled'] = True
        strawberry_data['model_loaded'] = False
        return True

    try:
        print(" 正在初始化本地YOLOv5草莓检测模型...")

        # 检查模型文件是否存在
        model_path = Path(strawberry_model_path)
        if not model_path.exists():
            print(f" 草莓模型文件未找到: {model_path}")
            print(" 请将训练好的best.pt文件放在项目根目录")
            strawberry_data['enabled'] = True  # 启用模拟模式
            strawberry_data['model_loaded'] = False
            return True

        # 检查是否可以使用DetectMultiBackend（本地YOLOv5）
        use_local_yolo = False
        try:
            # 设置设备
            local_yolo_device = select_device("0")  # 使用GPU 0，如果不可用会自动使用CPU
            print(f"📱 使用设备: {local_yolo_device}")

            # 加载数据配置
            data_path = YOLOV5_PATH / "data/stra.yaml"
            if not data_path.exists():
                data_path = None
                print(" 数据配置文件未找到，将使用模型内置配置")

            print(f" 使用本地YOLOv5加载草莓模型: {model_path}")
            print(f" 数据配置: {data_path}")

            # 使用DetectMultiBackend加载模型
            strawberry_model = DetectMultiBackend(
                weights=str(model_path),
                device=local_yolo_device,
                dnn=False,
                data=str(data_path) if data_path else None,
                fp16=False
            )

            # 检查图像尺寸
            local_yolo_imgsz = check_img_size(local_yolo_imgsz, s=strawberry_model.stride)
            print(f" 图像尺寸: {local_yolo_imgsz}")

            # 模型预热
            print("模型预热中...")
            strawberry_model.warmup(imgsz=(1, 3, local_yolo_imgsz, local_yolo_imgsz))

            use_local_yolo = True
            print(" 成功使用本地YOLOv5 DetectMultiBackend")

        except Exception as e:
            print(f" 本地YOLOv5 DetectMultiBackend失败: {e}")
            print(" 回退到torch.hub.load方式")
            # 回退到原始的torch.hub.load方式
            try:
                strawberry_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                                  path=strawberry_model_path, force_reload=True)
                strawberry_model.conf = STRAWBERRY_CONF_THRESHOLD
                strawberry_model.iou = STRAWBERRY_IOU_THRESHOLD
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                strawberry_model.to(device)
                use_local_yolo = False
                print(" 成功使用torch.hub.load方式")
            except Exception as e2:
                print(f"torch.hub.load也失败: {e2}")
                print(" 将使用模拟检测模式")
                strawberry_data['enabled'] = True
                strawberry_data['model_loaded'] = False
                STRAWBERRY_MODEL_CLASSES = STRAWBERRY_CLASSES.copy()
                return True

        # 🔧 修复：获取模型的真实类别
        if use_local_yolo:
            # 本地YOLOv5方式获取类别
            if hasattr(strawberry_model, 'names'):
                if isinstance(strawberry_model.names, dict):
                    STRAWBERRY_MODEL_CLASSES = [strawberry_model.names[i] for i in
                                                sorted(strawberry_model.names.keys())]
                elif isinstance(strawberry_model.names, list):
                    STRAWBERRY_MODEL_CLASSES = strawberry_model.names
                else:
                    STRAWBERRY_MODEL_CLASSES = list(strawberry_model.names)
                print(f" 本地YOLOv5模型类别: {STRAWBERRY_MODEL_CLASSES}")
            else:
                STRAWBERRY_MODEL_CLASSES = STRAWBERRY_CLASSES.copy()
                print(f"未能获取本地模型类别，使用配置类别: {STRAWBERRY_MODEL_CLASSES}")
        else:
            # torch.hub方式获取类别
            if hasattr(strawberry_model, 'names'):
                STRAWBERRY_MODEL_CLASSES = list(strawberry_model.names.values()) if isinstance(strawberry_model.names,
                                                                                               dict) else list(
                    strawberry_model.names)
                print(f" torch.hub模型类别: {STRAWBERRY_MODEL_CLASSES}")
            else:
                STRAWBERRY_MODEL_CLASSES = STRAWBERRY_CLASSES.copy()
                print(f" 未能获取模型类别，使用配置类别: {STRAWBERRY_MODEL_CLASSES}")

        strawberry_data['enabled'] = True
        strawberry_data['model_loaded'] = True
        strawberry_data['use_local_yolo'] = use_local_yolo

        print(f" 草莓模型初始化成功")
        print(f"   引擎: {'本地YOLOv5' if use_local_yolo else 'torch.hub'}")
        print(f"   设备: {local_yolo_device if use_local_yolo else 'cuda/cpu'}")
        print(f"   置信度阈值: {STRAWBERRY_CONF_THRESHOLD}")
        print(f"   IoU阈值: {STRAWBERRY_IOU_THRESHOLD}")
        print(f"   检测类别: {STRAWBERRY_MODEL_CLASSES}")

        # 创建结果保存文件夹
        if not os.path.exists(STRAWBERRY_SAVE_PATH):
            os.makedirs(STRAWBERRY_SAVE_PATH)
            print(f" 创建草莓检测结果保存文件夹: {STRAWBERRY_SAVE_PATH}")

        return True

    except Exception as e:
        print(f" 草莓模型初始化失败: {e}")
        traceback.print_exc()
        print(" 将使用模拟检测模式")
        strawberry_data['enabled'] = True  # 启用模拟检测作为后备
        strawberry_data['model_loaded'] = False
        STRAWBERRY_MODEL_CLASSES = STRAWBERRY_CLASSES.copy()
        return True


def detect_with_local_yolo(image):
    """使用本地YOLOv5进行草莓检测"""
    global strawberry_model, local_yolo_device, local_yolo_imgsz

    detections = []

    try:
        # 图像预处理
        img = letterbox(image, local_yolo_imgsz, stride=strawberry_model.stride, auto=strawberry_model.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # 转换为tensor
        img = torch.from_numpy(img).to(local_yolo_device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # 推理
        pred = strawberry_model(img, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(
            pred,
            STRAWBERRY_CONF_THRESHOLD,
            STRAWBERRY_IOU_THRESHOLD,
            classes=None,
            agnostic=False,
            max_det=1000
        )

        # 处理结果
        for i, det in enumerate(pred):
            if len(det):
                # 将检测框从img_size缩放到原图尺寸
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()

                # 解析每个检测结果
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)
                    confidence = float(conf)

                    # 获取类别名称
                    class_name = "Unknown"
                    if class_id < len(STRAWBERRY_MODEL_CLASSES):
                        class_name = STRAWBERRY_MODEL_CLASSES[class_id]
                    elif class_id < len(STRAWBERRY_CLASSES):
                        class_name = STRAWBERRY_CLASSES[class_id]
                    else:
                        class_name = f'Unknown_Class_{class_id}'
                        print(f" 未知类别ID: {class_id}")

                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'source': 'local_yolov5'
                    }
                    detections.append(detection)

        return detections

    except Exception as e:
        print(f" 本地YOLOv5检测失败: {e}")
        traceback.print_exc()
        return []


def simulate_strawberry_detection(image):
    """模拟草莓检测（当真实模型不可用时）"""
    print(" 使用草莓模拟检测模式")

    mock_detections = []

    # 随机生成1-5个草莓检测结果
    num_strawberries = random.randint(1, 5)

    # 🔧 修复：使用正确的类别列表
    strawberry_classes = STRAWBERRY_MODEL_CLASSES if STRAWBERRY_MODEL_CLASSES else STRAWBERRY_CLASSES

    for i in range(num_strawberries):
        # 随机生成边界框
        x1 = random.randint(50, 300)
        y1 = random.randint(50, 200)
        x2 = x1 + random.randint(30, 80)  # 草莓通常较小
        y2 = y1 + random.randint(30, 80)

        # 确保边界框在图像范围内
        x2 = min(x2, image.shape[1] - 10)
        y2 = min(y2, image.shape[0] - 10)

        ripeness = random.choice(strawberry_classes)

        detection = {
            'bbox': [x1, y1, x2, y2],
            'confidence': random.uniform(0.4, 0.95),
            'class_id': strawberry_classes.index(ripeness),
            'class_name': ripeness,
            'source': 'simulation'
        }
        mock_detections.append(detection)

    return mock_detections


def detect_strawberries(frame_base64):
    """草莓专用检测函数 - 本地YOLOv5修改版"""
    if not strawberry_data['enabled']:
        return [], None

    start_time = time.time()

    try:
        # 解码base64图像
        img_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if original_image is None:
            print(" 草莓检测：图像解码失败")
            return [], None

        print(f" 正在进行草莓检测 ({original_image.shape[1]}x{original_image.shape[0]})")

        # 根据是否有真实模型选择检测方法
        if strawberry_data['model_loaded'] and strawberry_model is not None:
            # 检查是否使用本地YOLOv5
            if strawberry_data.get('use_local_yolo', False):
                # 本地YOLOv5检测
                detections = detect_with_local_yolo(original_image)
                print(f" 本地YOLOv5检测完成，发现 {len(detections)} 个草莓")
            else:
                # torch.hub方式检测
                results = strawberry_model(original_image)
                detections = []
                if len(results.xyxy[0]) > 0:
                    for *box, conf, cls in results.xyxy[0].cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        class_id = int(cls)
                        confidence = float(conf)

                        # 🔧 修复：使用正确的类别获取逻辑
                        class_name = "Unknown"

                        # 优先使用模型自带的类别名称
                        if hasattr(strawberry_model, 'names') and class_id in strawberry_model.names:
                            class_name = strawberry_model.names[class_id]
                        elif hasattr(results, 'names') and class_id in results.names:
                            class_name = results.names[class_id]
                        elif class_id < len(STRAWBERRY_MODEL_CLASSES):
                            class_name = STRAWBERRY_MODEL_CLASSES[class_id]
                        elif class_id < len(STRAWBERRY_CLASSES):
                            class_name = STRAWBERRY_CLASSES[class_id]
                        else:
                            class_name = f'Unknown_Class_{class_id}'
                            print(f" 未知类别ID: {class_id}, 模型类别数: {len(STRAWBERRY_MODEL_CLASSES)}")

                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'source': 'torch_hub'
                        }
                        detections.append(detection)

                print(f" torch.hub检测完成，发现 {len(detections)} 个草莓")

            # 🔧 打印调试信息
            if detections:
                print(" 检测详情:")
                for i, det in enumerate(detections):
                    print(f"   草莓{i + 1}: {det['class_name']} (ID:{det['class_id']}, 置信度:{det['confidence']:.3f})")

        else:
            # 模拟草莓检测
            detections = simulate_strawberry_detection(original_image)
            print(f" 草莓模拟检测完成，生成 {len(detections)} 个草莓")

        # 绘制草莓检测结果到图像上
        annotated_image = None
        if DRAW_ANNOTATIONS and detections:
            annotated_image = draw_strawberry_detections(original_image.copy(), detections)
            print(f" 已在图像上标注 {len(detections)} 个草莓检测框")
        else:
            annotated_image = original_image.copy()

        # 记录处理时间
        processing_time = time.time() - start_time
        strawberry_data['detection_stats']['processing_time'] = processing_time

        # 更新草莓统计数据
        update_strawberry_stats(detections)

        print(f"⏱ 草莓检测耗时: {processing_time:.2f}秒")

        # 保存草莓检测结果（可选）
        if detections and len(detections) > 0:
            save_strawberry_result(annotated_image, detections)

        return detections, annotated_image

    except Exception as e:
        print(f" 草莓检测过程发生错误: {e}")
        traceback.print_exc()
        return [], None


def draw_strawberry_detections(image, detections):
    """在图像上绘制草莓检测结果 - 修复版"""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']

        # 选择颜色
        color = STRAWBERRY_COLORS.get(class_name, (255, 255, 255))  # 默认白色

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, ANNOTATION_THICKNESS)

        # 🔧 修复：确保标签格式正确，移除可能导致显示问题的字符
        label = f"{class_name} {confidence:.2f}"

        # 计算文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS
        )

        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )

        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS
        )

    # 添加草莓检测统计信息
    stats_text = f"Strawberries: {len(detections)}"
    cv2.putText(image, stats_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # 橙色

    return image


def update_strawberry_stats(detections):
    """更新草莓检测统计数据 - 修复版"""
    global strawberry_data

    # 🔧 修复：使用正确的类别列表
    available_classes = STRAWBERRY_MODEL_CLASSES if STRAWBERRY_MODEL_CLASSES else STRAWBERRY_CLASSES

    #  重置当前帧计数（使用可用类别）
    current_counts = {}
    for class_name in available_classes:
        current_counts[class_name] = 0

    # 统计当前检测结果
    for detection in detections:
        class_name = detection['class_name']

        #  确保类别在统计中存在
        if class_name in available_classes:
            current_counts[class_name] += 1
            #  累计统计（历史数据用）
            if class_name not in strawberry_data['detection_stats']['ripeness_counts']:
                strawberry_data['detection_stats']['ripeness_counts'][class_name] = 0
            strawberry_data['detection_stats']['ripeness_counts'][class_name] += 1
        else:
            print(f" 发现未知草莓类别: {class_name}")
            # 处理未知类别
            if 'Unknown' not in current_counts:
                current_counts['Unknown'] = 0
            current_counts['Unknown'] += 1

    #  更新当前帧数据（用于前端实时显示）
    strawberry_data['detection_stats']['current_ripeness'] = current_counts

    # 计算当前帧的可收获数量和质量评分
    current_harvest_ready = current_counts.get('Ripe', 0) + current_counts.get('Nearly Ripe', 0)
    total_current_strawberries = sum(current_counts.values())

    if total_current_strawberries > 0:
        good_strawberries = current_counts.get('Ripe', 0) + current_counts.get('Nearly Ripe', 0)
        current_quality_score = (good_strawberries / total_current_strawberries) * 100
    else:
        current_quality_score = 0

    #  更新当前帧统计
    strawberry_data['detection_stats']['current_harvest_ready'] = current_harvest_ready
    strawberry_data['detection_stats']['current_quality_score'] = round(current_quality_score, 1)

    #  更新累计统计（保留原有逻辑用于历史分析）
    total_harvest_ready = strawberry_data['detection_stats']['ripeness_counts'].get('Ripe', 0) + \
                          strawberry_data['detection_stats']['ripeness_counts'].get('Nearly Ripe', 0)
    strawberry_data['detection_stats']['harvest_ready'] = total_harvest_ready

    # 计算累计质量评分
    total_strawberries = sum(strawberry_data['detection_stats']['ripeness_counts'].values())
    if total_strawberries > 0:
        total_good = strawberry_data['detection_stats']['ripeness_counts'].get('Ripe', 0) + \
                     strawberry_data['detection_stats']['ripeness_counts'].get('Nearly Ripe', 0)
        quality_score = (total_good / total_strawberries) * 100
        strawberry_data['detection_stats']['quality_score'] = round(quality_score, 1)
    else:
        strawberry_data['detection_stats']['quality_score'] = 0

    # 更新总检测数和时间戳
    strawberry_data['detection_stats']['total_detections'] += len(detections)
    strawberry_data['detection_stats']['last_update'] = time.time()

    #  打印当前帧统计信息
    if detections:
        print(f" 当前帧草莓分布: {current_counts}")
        print(f" 当前帧质量评分: {current_quality_score:.1f}%")
        print(f" 累计草莓统计: {strawberry_data['detection_stats']['ripeness_counts']}")

        #  详细输出检测到的每个草莓
        for i, detection in enumerate(detections):
            print(f"   草莓{i + 1}: {detection['class_name']} (置信度: {detection['confidence']:.3f})")
    else:
        print(" 当前帧未检测到草莓")


def save_strawberry_result(annotated_image, detections):
    """保存草莓检测结果"""
    try:
        timestamp = int(time.time())
        filename = f"strawberry_{timestamp}.jpg"
        filepath = os.path.join(STRAWBERRY_SAVE_PATH, filename)

        cv2.imwrite(filepath, annotated_image)

        # 保存检测数据
        json_filename = f"strawberry_{timestamp}.json"
        json_filepath = os.path.join(STRAWBERRY_SAVE_PATH, json_filename)

        # 🔧 修复：使用正确的类别进行统计
        available_classes = STRAWBERRY_MODEL_CLASSES if STRAWBERRY_MODEL_CLASSES else STRAWBERRY_CLASSES
        ripeness_distribution = {class_name: 0 for class_name in available_classes}

        for detection in detections:
            class_name = detection['class_name']
            if class_name in ripeness_distribution:
                ripeness_distribution[class_name] += 1

        detection_record = {
            'timestamp': timestamp,
            'detections': detections,
            'image_filename': filename,
            'strawberry_count': len(detections),
            'ripeness_distribution': ripeness_distribution,
            'harvest_ready': ripeness_distribution.get('Ripe', 0) + ripeness_distribution.get('Nearly Ripe', 0),
            'quality_score': strawberry_data['detection_stats']['quality_score'],
            'model_classes': STRAWBERRY_MODEL_CLASSES,
            'config_classes': STRAWBERRY_CLASSES,
            'detection_engine': strawberry_data.get('use_local_yolo', False) and 'local_yolov5' or 'torch_hub'
        }

        with open(json_filepath, 'w') as f:
            json.dump(detection_record, f, indent=2)

        print(f" 草莓检测结果已保存: {filepath}")

    except Exception as e:
        print(f" 保存草莓检测结果失败: {e}")


# ========== 通用YOLOv5检测函数 ==========

def init_yolov5():
    """初始化YOLOv5模型"""
    global yolov5_model, detection_enabled

    if not YOLOV5_AVAILABLE:
        print(" YOLOv5不可用，将使用模拟检测")
        detection_enabled = True  # 启用模拟检测
        return True

    try:
        print(" 正在初始化通用YOLOv5模型...")
        print(f" 模型: {YOLO_MODEL_NAME}")

        # 加载预训练模型
        yolov5_model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True)

        # 设置参数
        yolov5_model.conf = YOLO_CONF_THRESHOLD
        yolov5_model.iou = YOLO_IOU_THRESHOLD

        # 设备配置
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        yolov5_model.to(device)

        detection_enabled = True

        print(f" 通用YOLOv5模型初始化成功")
        print(f"   设备: {device}")
        print(f"   置信度阈值: {YOLO_CONF_THRESHOLD}")
        print(f"   IoU阈值: {YOLO_IOU_THRESHOLD}")

        # 创建结果保存文件夹
        if not os.path.exists(DETECTION_SAVE_PATH):
            os.makedirs(DETECTION_SAVE_PATH)
            print(f" 创建检测结果保存文件夹: {DETECTION_SAVE_PATH}")

        return True

    except Exception as e:
        print(f" YOLOv5模型初始化失败: {e}")
        print(" 将使用模拟检测模式")
        detection_enabled = True  # 启用模拟检测作为后备
        return True


def simulate_detection(image):
    """模拟YOLOv5检测（当真实YOLOv5不可用时）"""
    print(" 使用模拟检测模式")

    # 模拟一些检测结果
    mock_detections = []

    # 随机生成1-3个检测结果
    num_detections = random.randint(0, 4)

    # 模拟的类别
    mock_classes = [
        'person', 'car', 'bottle', 'chair', 'phone',
        'book', 'laptop', 'mouse', 'keyboard', 'cup',
        'dog', 'cat', 'tv', 'clock', 'potted plant'
    ]

    for i in range(num_detections):
        # 随机生成边界框
        x1 = random.randint(50, 300)
        y1 = random.randint(50, 200)
        x2 = x1 + random.randint(50, 200)
        y2 = y1 + random.randint(50, 150)

        # 确保边界框在图像范围内
        x2 = min(x2, image.shape[1] - 10)
        y2 = min(y2, image.shape[0] - 10)

        class_name = random.choice(mock_classes)

        detection = {
            'bbox': [x1, y1, x2, y2],
            'confidence': random.uniform(0.5, 0.95),
            'class_id': mock_classes.index(class_name),
            'class_name': class_name,
            'source': 'simulation'
        }
        mock_detections.append(detection)

    return mock_detections


def detect_objects(frame_base64):
    """YOLOv5物体检测并返回标注后的图像"""
    if not detection_enabled:
        return [], None

    start_time = time.time()

    try:
        # 解码base64图像
        img_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if original_image is None:
            print(" 图像解码失败")
            return [], None

        print(f"🔍 正在检测图像 ({original_image.shape[1]}x{original_image.shape[0]})")

        # 根据是否有真实模型选择检测方法
        if YOLOV5_AVAILABLE and yolov5_model is not None:
            # 真实YOLOv5检测
            results = yolov5_model(original_image)

            # 解析结果
            detections = []
            if len(results.xyxy[0]) > 0:
                for *box, conf, cls in results.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)
                    confidence = float(conf)

                    # 获取类别名称
                    class_name = results.names[class_id] if hasattr(results, 'names') else f'Class_{class_id}'

                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'source': 'yolov5'
                    }
                    detections.append(detection)

            print(f" YOLOv5检测完成，发现 {len(detections)} 个物体")
        else:
            # 模拟检测
            detections = simulate_detection(original_image)
            print(f" 模拟检测完成，生成 {len(detections)} 个物体")

        # 绘制检测结果到图像上
        annotated_image = None
        if DRAW_ANNOTATIONS and detections:
            annotated_image = draw_detections(original_image.copy(), detections)
            print(f" 已在图像上标注 {len(detections)} 个检测框")
        else:
            # 如果没有检测到物体，或者不绘制标注，返回原图
            annotated_image = original_image.copy()

        # 记录处理时间
        processing_time = time.time() - start_time
        detection_data['detection_stats']['processing_time'] = processing_time

        #  更新通用检测统计数据
        update_detection_stats(detections)

        print(f"⏱️ 检测耗时: {processing_time:.2f}秒")

        # 保存检测结果图像（可选）
        if detections and len(detections) > 0:
            save_detection_result(annotated_image, detections)

        return detections, annotated_image

    except Exception as e:
        print(f" 检测过程发生错误: {e}")
        traceback.print_exc()
        return [], None


def update_detection_stats(detections):
    """更新通用检测统计数据"""
    global detection_data

    # 重置当前帧物体统计
    current_objects = {}

    # 统计当前检测结果
    for detection in detections:
        class_name = detection.get('class_name', 'unknown')

        # 更新当前帧统计
        if class_name not in current_objects:
            current_objects[class_name] = 0
        current_objects[class_name] += 1

        # 更新累计统计
        if class_name not in detection_data['detection_stats']['object_counts']:
            detection_data['detection_stats']['object_counts'][class_name] = 0
        detection_data['detection_stats']['object_counts'][class_name] += 1

    # 更新检测数据
    detection_data['detection_stats']['current_objects'] = current_objects
    detection_data['detection_stats']['total_detections'] += len(detections)
    detection_data['detection_stats']['last_update'] = time.time()

    # 计算类别分布百分比
    total_objects = sum(detection_data['detection_stats']['object_counts'].values())
    if total_objects > 0:
        category_distribution = {}
        for class_name, count in detection_data['detection_stats']['object_counts'].items():
            category_distribution[class_name] = round((count / total_objects) * 100, 1)
        detection_data['detection_stats']['category_distribution'] = category_distribution

    # 保存检测历史
    if detections:
        detection_record = {
            'timestamp': time.time(),
            'detections': detections,
            'object_counts': current_objects.copy(),
            'source': 'general_detection'
        }
        detection_data['detection_history'].append(detection_record)
        if len(detection_data['detection_history']) > 50:
            detection_data['detection_history'].pop(0)

        # 打印当前检测统计
        print(f" 当前检测统计: {current_objects}")


def generate_detection_summary(detections):
    """生成检测结果摘要"""
    if not detections:
        return "未检测到任何物体"

    # 统计类别
    category_counts = {}
    for detection in detections:
        class_name = detection.get('class_name', 'unknown')
        category_counts[class_name] = category_counts.get(class_name, 0) + 1

    # 生成摘要文本
    summary_parts = []
    for class_name, count in category_counts.items():
        summary_parts.append(f"{class_name}: {count}个")

    return f"检测到 {len(detections)} 个物体 ({', '.join(summary_parts)})"


def draw_detections(image, detections):
    """在图像上绘制检测结果"""
    colors = [
        (0, 255, 0),  # 绿色
        (255, 0, 0),  # 蓝色
        (0, 0, 255),  # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 黄色
    ]

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']

        # 选择颜色
        color = colors[i % len(colors)]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, ANNOTATION_THICKNESS)

        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"

        # 计算文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS
        )

        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )

        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS
        )

    # 添加检测统计信息
    stats_text = f"Detected: {len(detections)} objects"
    cv2.putText(image, stats_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image


def save_detection_result(annotated_image, detections):
    """保存检测结果"""
    try:
        timestamp = int(time.time())
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(DETECTION_SAVE_PATH, filename)

        cv2.imwrite(filepath, annotated_image)

        # 保存检测数据
        json_filename = f"detection_{timestamp}.json"
        json_filepath = os.path.join(DETECTION_SAVE_PATH, json_filename)

        # 统计当前检测的类别分布
        category_counts = {}
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            category_counts[class_name] = category_counts.get(class_name, 0) + 1

        detection_record = {
            'timestamp': timestamp,
            'detections': detections,
            'image_filename': filename,
            'detection_count': len(detections),
            'category_counts': category_counts,
            'processing_time': detection_data['detection_stats']['processing_time']
        }

        with open(json_filepath, 'w') as f:
            json.dump(detection_record, f, indent=2)

        print(f" 检测结果已保存: {filepath}")

    except Exception as e:
        print(f" 保存检测结果失败: {e}")


# ========== DeepSeek R1 AI 相关函数 ==========
def test_deepseek_connection():
    """测试DeepSeek R1连接"""
    try:
        print(f" 正在测试连接到 {deepseek_data['api_url']}")
        response = requests.get(f"{deepseek_data['api_url']}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            available_models = [model['name'] for model in data.get('models', [])]
            print(f"可用模型: {available_models}")

            # 检查目标模型是否存在
            if deepseek_data['model_name'] in available_models:
                deepseek_data['status'] = 'connected'
                deepseek_data['enabled'] = True
                print(f" DeepSeek R1 模型 {deepseek_data['model_name']} 连接成功")
                return True
            else:
                deepseek_data['status'] = 'model_not_found'
                deepseek_data['enabled'] = False
                print(f" 模型 {deepseek_data['model_name']} 未找到")
                print(f" 请确保已下载模型: ollama pull {deepseek_data['model_name']}")
                return False
        else:
            deepseek_data['status'] = 'error'
            deepseek_data['enabled'] = False
            print(f" DeepSeek R1 连接失败: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        deepseek_data['status'] = 'connection_refused'
        deepseek_data['enabled'] = False
        print(f" 连接被拒绝，请确保 Ollama 服务正在运行")
        print(f" 启动命令: ollama serve")
        return False
    except requests.exceptions.Timeout:
        deepseek_data['status'] = 'timeout'
        deepseek_data['enabled'] = False
        print(f" 连接超时")
        return False
    except Exception as e:
        deepseek_data['status'] = 'error'
        deepseek_data['enabled'] = False
        print(f" DeepSeek R1 连接测试失败: {e}")
        return False


def call_deepseek_api(message, context=None):
    """调用DeepSeek R1 API"""
    if not deepseek_data['enabled']:
        return None, "DeepSeek R1 未连接"

    try:
        start_time = time.time()

        # 构建提示词，加入系统上下文
        system_prompt = """你是一个智能农业监控系统的AI助手。你可以：
1. 分析传感器数据（温度、湿度、光照）
2. 解读物体检测结果
3. 提供农业种植建议
4. 回答用户的问题

请用简洁、专业的语言回答。"""

        # 如果有上下文信息，加入到提示中
        if context:
            enhanced_message = f"{system_prompt}\n\n当前系统状态：\n{context}\n\n用户问题：{message}"
        else:
            enhanced_message = f"{system_prompt}\n\n用户问题：{message}"

        # 调用API
        response = requests.post(
            f"{deepseek_data['api_url']}/api/generate",
            json={
                "model": deepseek_data['model_name'],
                "prompt": enhanced_message,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=DEEPSEEK_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            ai_response = data.get('response', '抱歉，没有收到有效回复。')

            # 移除思考标签
            ai_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL).strip()
            if not ai_response:
                ai_response = '正在思考中...'

            # 记录统计信息
            deepseek_data['response_time'] = time.time() - start_time
            deepseek_data['total_messages'] += 1
            deepseek_data['last_response'] = ai_response

            # 更新对话历史
            deepseek_data['chat_history'].append({
                'role': 'user',
                'content': message,
                'timestamp': time.time()
            })
            deepseek_data['chat_history'].append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': time.time()
            })

            # 限制历史记录长度
            if len(deepseek_data['chat_history']) > DEEPSEEK_MAX_HISTORY * 2:
                deepseek_data['chat_history'] = deepseek_data['chat_history'][-DEEPSEEK_MAX_HISTORY:]

            return ai_response, None

        else:
            error_msg = f"API请求失败: HTTP {response.status_code}"
            deepseek_data['error_count'] += 1
            return None, error_msg

    except requests.exceptions.Timeout:
        error_msg = "请求超时，请检查网络连接"
        deepseek_data['error_count'] += 1
        return None, error_msg
    except Exception as e:
        error_msg = f"API调用失败: {str(e)}"
        deepseek_data['error_count'] += 1
        return None, error_msg


def generate_system_context():
    """生成系统上下文信息供AI参考 - 增强版"""
    context = []
    missing_data = []

    # 传感器状态
    context.append(" 传感器状态:")
    sensor_available = False
    for sensor_type, data in sensor_data.items():
        if data['value'] is not None:
            sensor_name = {'temperature': '温度', 'humidity': '湿度', 'light': '光照'}[sensor_type]
            context.append(f"- {sensor_name}: {data['value']}{data['unit']} ({data['status']})")
            sensor_available = True
        else:
            sensor_name = {'temperature': '温度', 'humidity': '湿度', 'light': '光照'}[sensor_type]
            context.append(f"- {sensor_name}: 无数据")

    if not sensor_available:
        missing_data.append("环境传感器数据")

    # 通用检测状态
    context.append("\n 通用检测状态:")
    if detection_enabled and detection_data['last_detections']:
        current_objects = detection_data['detection_stats']['current_objects']
        object_summary = ', '.join([f"{name}:{count}个" for name, count in current_objects.items()])
        context.append(f"- 检测到: {object_summary} (共{len(detection_data['last_detections'])}个)")
    elif not detection_enabled:
        context.append("- 通用检测系统未启用")
        missing_data.append("通用检测数据")
    else:
        context.append("- 未检测到物体")

    # 草莓检测状态
    context.append("\n 草莓检测状态:")
    if strawberry_data['enabled'] and strawberry_data['last_detections']:
        ripeness_counts = strawberry_data['detection_stats']['current_ripeness']
        strawberry_summary = ', '.join([f"{name}:{count}个" for name, count in ripeness_counts.items() if count > 0])
        context.append(f"- 草莓状况: {strawberry_summary}")
        context.append(f"- 可收获: {strawberry_data['detection_stats']['current_harvest_ready']}个")
        context.append(f"- 质量评分: {strawberry_data['detection_stats']['current_quality_score']}%")

        # 添加检测引擎信息
        engine_info = "本地YOLOv5" if strawberry_data.get('use_local_yolo', False) else "torch.hub"
        context.append(f"- 检测引擎: {engine_info}")
    elif not strawberry_data['enabled']:
        context.append("- 草莓检测系统未启用")
        missing_data.append("草莓检测数据")
    else:
        context.append("- 未检测到草莓")

    # 设备状态
    context.append(f"\n📷 设备状态: {video_data['device_status']}")
    if video_data.get('test_mode'):
        context.append(f"- 测试模式，当前图片: {video_data.get('current_image_filename', 'unknown')}")

    # MQTT连接状态
    mqtt_status = mqtt_client.is_connected() if mqtt_client else False
    context.append(f"\n📡 MQTT连接: {'已连接' if mqtt_status else '未连接'}")
    if not mqtt_status:
        missing_data.append("MQTT数据传输")

    # 数据缺失警告
    if missing_data:
        context.append(f"\n⚠️ 数据缺失: {', '.join(missing_data)}")
        context.append("请在分析中说明数据缺失的影响")

    return "\n".join(context)


def process_ai_request_async(message, user_context=None):
    """异步处理AI请求"""

    def ai_task():
        try:
            # 生成系统上下文
            system_context = generate_system_context()
            if user_context:
                system_context += f"\n\n补充信息: {user_context}"

            # 调用DeepSeek API
            response, error = call_deepseek_api(message, system_context)

            if response:
                print(f" AI回复: {response[:100]}...")

                # 推送AI回复到前端
                socketio.emit('ai_response', {
                    'message': message,
                    'response': response,
                    'timestamp': time.time(),
                    'response_time': deepseek_data['response_time'],
                    'error': None
                })
            else:
                print(f" AI请求失败: {error}")

                # 推送错误信息到前端
                socketio.emit('ai_response', {
                    'message': message,
                    'response': None,
                    'timestamp': time.time(),
                    'error': error
                })

        except Exception as e:
            print(f" AI处理异常: {e}")
            socketio.emit('ai_response', {
                'message': message,
                'response': None,
                'timestamp': time.time(),
                'error': f"处理异常: {str(e)}"
            })

    # 在线程池中执行AI请求
    executor.submit(ai_task)


# ========== 传感器数据处理函数 ==========
def analyze_sensor_value(sensor_type, value):
    """分析传感器数值并返回状态"""
    if sensor_type not in OPTIMAL_RANGES:
        return 'unknown'

    ranges = OPTIMAL_RANGES[sensor_type]

    if ranges['min'] <= value <= ranges['max']:
        if abs(value - ranges['optimal']) <= (ranges['max'] - ranges['min']) * 0.1:
            return 'optimal'
        else:
            return 'good'
    elif value < ranges['min']:
        return 'low'
    else:
        return 'high'


def parse_sensor_data(payload, topic):
    """智能解析传感器数据 - 增强版"""
    data = None
    parse_method = ""

    try:
        # 方法1: 尝试标准JSON解析
        try:
            data = json.loads(payload)
            parse_method = "标准JSON格式"
            return data, parse_method
        except json.JSONDecodeError:
            pass

        # 方法2: 解析自定义格式 {key:value}
        try:
            # 支持 {humidity:6.703333333333333} 格式
            pattern = r'\{(\w+):([\d\.-]+)\}'
            match = re.search(pattern, payload.strip())

            if match:
                key = match.group(1)
                value = float(match.group(2))
                data = {key: value}
                parse_method = f"自定义格式 {{{key}:value}}"
                return data, parse_method
        except Exception:
            pass

        # 方法3: 纯数值解析
        try:
            numeric_value = float(payload.strip())
            if 'temperature' in topic:
                data = {"temperature": numeric_value}
            elif 'humidity' in topic:
                data = {"humidity": numeric_value}
            elif 'light' in topic:
                data = {"light": numeric_value}
            else:
                data = {"value": numeric_value}
            parse_method = "纯数值格式"
            return data, parse_method
        except ValueError:
            pass

        # 方法4: 关键词匹配解析
        try:
            # 匹配类似 "temperature=25.6" 或 "humidity:60%" 的格式
            for pattern in [r'(\w+)[=:]([\d\.-]+)', r'([\d\.-]+)\s*([°%]?[CFclux]*)', r'(\w+)\s+([\d\.-]+)']:
                match = re.search(pattern, payload)
                if match:
                    if len(match.groups()) >= 2:
                        key_or_value = match.group(1)
                        value_or_unit = match.group(2)

                        try:
                            value = float(value_or_unit if value_or_unit.replace('.', '').replace('-',
                                                                                                  '').isdigit() else key_or_value)

                            if 'temperature' in topic or 'temp' in payload.lower():
                                data = {"temperature": value}
                            elif 'humidity' in topic or 'hum' in payload.lower():
                                data = {"humidity": value}
                            elif 'light' in topic or 'lux' in payload.lower():
                                data = {"light": value}
                            else:
                                data = {"value": value}
                            parse_method = f"关键词匹配: {pattern}"
                            return data, parse_method
                        except ValueError:
                            continue
        except Exception:
            pass

        # 方法5: 默认处理
        data = {"value": payload, "raw": True}
        parse_method = "字符串格式"
        return data, parse_method

    except Exception as e:
        data = {"value": payload, "error": str(e)}
        parse_method = "异常处理"
        return data, parse_method


def process_sensor_data(payload, topic):
    """处理传感器数据并推送到前端"""
    global sensor_data, message_count

    try:
        # 解析传感器数据
        data, parse_method = parse_sensor_data(payload, topic)
        print(f"     解析方法: {parse_method}")

        # 确定传感器类型
        sensor_type = None
        value = None

        if 'temperature' in topic or 'temperature' in data:
            sensor_type = 'temperature'
            value = data.get('temperature', data.get('value'))
            unit = '°C'
        elif 'humidity' in topic or 'humidity' in data:
            sensor_type = 'humidity'
            value = data.get('humidity', data.get('value'))
            unit = '%'
        elif 'light' in topic or 'light' in data:
            sensor_type = 'light'
            value = data.get('light', data.get('value'))
            unit = 'lux'

        if sensor_type and value is not None:
            try:
                # 转换为数值
                numeric_value = float(value)

                # 分析传感器状态
                status = analyze_sensor_value(sensor_type, numeric_value)

                # 更新全局传感器数据
                sensor_data[sensor_type].update({
                    'value': numeric_value,
                    'unit': unit,
                    'timestamp': time.time(),
                    'status': status
                })

                print(f"     {sensor_type}: {numeric_value}{unit} - 状态: {status}")

                # 推送到前端
                socketio.emit('sensor_update', {
                    'sensor_type': sensor_type,
                    'data': sensor_data[sensor_type],
                    'message_count': message_count,
                    'timestamp': time.time()
                })

                # 推送所有传感器数据
                socketio.emit('all_sensors', sensor_data)

                return True

            except ValueError:
                print(f"     数值转换失败: {value}")
                return False
        else:
            print(f"    ⚠️ 未识别的传感器类型或数值: {data}")
            return False

    except Exception as e:
        print(f"     传感器数据处理失败: {e}")
        traceback.print_exc()
        return False


def process_image_data(payload):
    """处理Fibo发送的图片数据 - 修复版：只传输标注后的图片"""
    global video_data, detection_data, strawberry_data

    try:
        data = json.loads(payload)

        if 'frame' in data:
            print(f"📷 接收到图片数据: {data.get('test_image_name', 'unknown')}")

            # 更新视频数据
            video_data.update({
                'current_frame': data['frame'],
                'last_frame_time': time.time(),
                'frame_count': data.get('frame_count', 0),
                'resolution': data.get('resolution', 'Unknown'),
                'device_status': 'online',
                'test_mode': data.get('test_mode', False),
                'current_image_filename': data.get('test_image_name', 'unknown')
            })

            # 并行执行通用检测和草莓检测
            def run_general_detection():
                try:
                    detections, annotated_image = detect_objects(data['frame'])

                    # 更新通用检测数据
                    if detections and len(detections) > 0:
                        detection_data['last_detections'] = detections
                        detection_data['detection_count'] = len(detections)
                        detection_data['detection_stats']['last_update'] = time.time()

                        print(f" 通用检测结果: {len(detections)} 个物体")

                        # 打印检测到的物体详情
                        current_objects = detection_data['detection_stats']['current_objects']
                        for class_name, count in current_objects.items():
                            print(f"   - {class_name}: {count}个")
                    else:
                        detection_data['last_detections'] = []
                        detection_data['detection_count'] = 0
                        detection_data['detection_stats']['current_objects'] = {}

                    # 推送通用检测结果（包含详细统计）
                    socketio.emit('detection_update', {
                        'detections': detections or [],
                        'detection_count': len(detections) if detections else 0,
                        'timestamp': time.time(),
                        'image_filename': data.get('test_image_name', 'unknown'),
                        'stats': detection_data['detection_stats'],
                        'current_objects': detection_data['detection_stats']['current_objects'],
                        'category_distribution': detection_data['detection_stats']['category_distribution'],
                        'summary': generate_detection_summary(detections or [])
                    })

                    #  只推送通用检测标注图像
                    if annotated_image is not None:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                        result, encoded_img = cv2.imencode('.jpg', annotated_image, encode_param)
                        if result:
                            general_frame = base64.b64encode(encoded_img).decode('utf-8')
                            socketio.emit('general_frame', {
                                'frame': general_frame,
                                'timestamp': time.time(),
                                'image_filename': data.get('test_image_name', 'unknown'),
                                'detection_count': len(detections) if detections else 0,
                                'frame_type': 'general_detection'  # 标识帧类型
                            })

                except Exception as e:
                    print(f" 通用检测异常: {e}")
                    traceback.print_exc()

            def run_strawberry_detection():
                try:
                    strawberry_detections, strawberry_annotated = detect_strawberries(data['frame'])

                    # 更新草莓检测数据
                    if strawberry_detections and len(strawberry_detections) > 0:
                        strawberry_data['last_detections'] = strawberry_detections
                        strawberry_data['detection_count'] = len(strawberry_detections)

                        print(f" 草莓检测结果: {len(strawberry_detections)} 个草莓")
                        for detection in strawberry_detections:
                            print(f"   - {detection['class_name']}: {detection['confidence']:.2f}")
                    else:
                        strawberry_data['last_detections'] = []
                        strawberry_data['detection_count'] = 0

                    # 推送草莓检测结果
                    socketio.emit('strawberry_update', {
                        'detections': strawberry_detections or [],
                        'detection_count': len(strawberry_detections) if strawberry_detections else 0,
                        'timestamp': time.time(),
                        'image_filename': data.get('test_image_name', 'unknown'),
                        'stats': strawberry_data['detection_stats']
                    })

                    #  推送草莓检测标注图像
                    if strawberry_annotated is not None:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                        result, encoded_img = cv2.imencode('.jpg', strawberry_annotated, encode_param)
                        if result:
                            strawberry_frame = base64.b64encode(encoded_img).decode('utf-8')
                            socketio.emit('strawberry_frame', {
                                'frame': strawberry_frame,
                                'timestamp': time.time(),
                                'image_filename': data.get('test_image_name', 'unknown'),
                                'strawberry_count': len(strawberry_detections) if strawberry_detections else 0,
                                'frame_type': 'strawberry_detection'  # 标识帧类型
                            })

                except Exception as e:
                    print(f" 草莓检测异常: {e}")

            # 并行执行两种检测
            detection_futures = []

            if detection_enabled:
                future = executor.submit(run_general_detection)
                detection_futures.append(('general', future))

            if strawberry_data['enabled']:
                future = executor.submit(run_strawberry_detection)
                detection_futures.append(('strawberry', future))

            # 等待所有检测完成后，发送一个状态更新（不包含原始图片）
            def send_status_update():
                try:
                    # 等待所有检测任务完成
                    for task_type, future in detection_futures:
                        try:
                            future.result(timeout=10)  # 10秒超时
                        except Exception as e:
                            print(f" {task_type}检测任务异常: {e}")

                    # 发送综合状态更新（不包含图片数据）
                    socketio.emit('frame_processed', {
                        'timestamp': time.time(),
                        'image_filename': data.get('test_image_name', 'unknown'),
                        'frame_count': data.get('frame_count', 0),
                        'resolution': data.get('resolution', 'Unknown'),
                        'test_mode': data.get('test_mode', False),
                        'general_detection_count': detection_data['detection_count'],
                        'strawberry_count': strawberry_data['detection_count'],
                        'device_status': 'online'
                    })

                except Exception as e:
                    print(f" 状态更新异常: {e}")

            # 异步发送状态更新
            executor.submit(send_status_update)

            return True

    except Exception as e:
        print(f" 图片数据处理失败: {e}")
        traceback.print_exc()
        return False


def process_device_status(payload):
    """处理设备状态"""
    global video_data

    try:
        data = json.loads(payload)

        status = data.get('status', 'unknown')
        device = data.get('device', 'unknown')

        video_data['device_status'] = status

        if status == 'online':
            video_data.update({
                'resolution': data.get('resolution', 'Unknown'),
                'test_mode': True
            })
            print(f"📷 {device} 图片发送器上线")
            if data.get('image_count'):
                print(f" 图片数量: {data['image_count']}")
        elif status == 'offline':
            print(f" {device} 图片发送器离线")

        # 推送设备状态到前端
        socketio.emit('camera_status', {
            'status': status,
            'device': device,
            'resolution': video_data.get('resolution'),
            'test_mode': video_data.get('test_mode', False),
            'image_count': data.get('image_count', 0),
            'timestamp': time.time()
        })

        return True

    except Exception as e:
        print(f" 设备状态处理失败: {e}")
        return False


# ========== MQTT相关函数 ==========
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f" MQTT连接成功!")

        # 订阅所有主题
        for topic in MQTT_TOPICS:
            client.subscribe(topic)
            print(f"📡 已订阅主题: {topic}")

        print("🎯 等待图片和传感器数据...")

        # 通知前端MQTT连接状态
        socketio.emit('mqtt_status', {
            'connected': True,
            'broker': MQTT_BROKER,
            'topics': MQTT_TOPICS
        })
    else:
        error_messages = {
            1: "协议版本不正确", 2: "客户端标识符被拒绝",
            3: "服务器不可用", 4: "用户名或密码错误", 5: "未授权访问"
        }
        print(f" MQTT连接失败，错误代码: {rc}")
        print(f"   错误原因: {error_messages.get(rc, '未知错误')}")


def on_message(client, userdata, msg):
    """处理MQTT消息 - 修复版"""
    global message_count
    message_count += 1

    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')

        print(f" [{message_count:04d}] 主题: {topic}")

        # 处理图片数据
        if '/camera/fibo/video' in topic:
            print(f"     接收图片数据，大小: {len(payload)} 字节")
            if process_image_data(payload):
                print(f"     图片处理和检测完成")
            return

        # 处理设备状态
        elif '/camera/fibo/status' in topic:
            print(f"     接收设备状态: {payload[:100]}...")
            if process_device_status(payload):
                print(f"     设备状态更新成功")
            return

        # 处理传感器数据
        elif any(sensor in topic for sensor in ['/sensor/temperature', '/sensor/humidity', '/sensor/light']):
            print(f"     传感器数据: {payload[:50]}...")
            if process_sensor_data(payload, topic):
                print(f"     传感器数据处理完成")
            return

        # 其他未处理的消息
        else:
            print(f"     未处理的主题: {topic}")
            print(f"     数据: {payload[:100]}...")

        print("-" * 60)

    except Exception as e:
        print(f" 处理消息时发生错误: {e}")
        traceback.print_exc()


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f" MQTT意外断开连接，代码: {rc}")
        socketio.emit('mqtt_status', {'connected': False, 'error': '连接意外断开'})
    else:
        print(" MQTT正常断开连接")


def create_mqtt_client():
    print(" 正在初始化MQTT客户端...")

    client_id = f"server_yolov5_strawberry_monitor_{int(time.time())}"
    print(f" 客户端ID: {client_id}")

    try:
        if hasattr(mqtt, 'CallbackAPIVersion'):
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
        else:
            client = mqtt.Client(client_id=client_id)
    except Exception as e:
        print(f" API版本检测失败，使用默认方式: {e}")
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        print(f" 正在连接到 {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
        return client
    except Exception as e:
        print(f" MQTT连接失败: {e}")
        return None


# ========== Web API路由 ==========
@app.route('/')
def index():
    return render_template('strawberry_enhanced.html')


@app.route('/api/ai/status')
def get_ai_status():
    return jsonify({
        'enabled': deepseek_data['enabled'],
        'status': deepseek_data['status'],
        'api_url': deepseek_data['api_url'],
        'model_name': deepseek_data['model_name'],
        'total_messages': deepseek_data['total_messages'],
        'error_count': deepseek_data['error_count'],
        'last_response_time': deepseek_data['response_time']
    })


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': '消息不能为空'}), 400

        if not deepseek_data['enabled']:
            return jsonify({'error': 'DeepSeek R1 未连接'}), 503

        print(f" 收到AI聊天请求: {message[:50]}...")

        # 异步处理AI请求
        process_ai_request_async(message)

        return jsonify({
            'status': 'processing',
            'message': '正在处理您的请求...',
            'timestamp': time.time()
        })

    except Exception as e:
        print(f" AI聊天API错误: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/api/ai/test', methods=['POST'])
def test_ai_connection():
    """测试AI连接"""
    try:
        # 如果请求体包含配置，先更新配置
        if request.is_json:
            data = request.get_json()
            if data and 'api_url' in data:
                deepseek_data['api_url'] = data['api_url'].strip()
            if data and 'model_name' in data:
                deepseek_data['model_name'] = data['model_name'].strip()

        success = test_deepseek_connection()
        return jsonify({
            'success': success,
            'status': deepseek_data['status'],
            'enabled': deepseek_data['enabled'],
            'api_url': deepseek_data['api_url'],
            'model_name': deepseek_data['model_name']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ai/history')
def get_ai_history():
    return jsonify({
        'chat_history': deepseek_data['chat_history'][-20:],  # 最近20条
        'total_messages': deepseek_data['total_messages'],
        'error_count': deepseek_data['error_count']
    })


@app.route('/api/detection/status')
def get_detection_status():
    return jsonify({
        'detection_enabled': detection_enabled,
        'yolov5_available': YOLOV5_AVAILABLE,
        'model_name': YOLO_MODEL_NAME,
        'conf_threshold': YOLO_CONF_THRESHOLD,
        'total_detections': detection_data['detection_stats']['total_detections'],
        'last_processing_time': detection_data['detection_stats']['processing_time'],
        'current_objects': detection_data['detection_stats']['current_objects'],
        'category_distribution': detection_data['detection_stats']['category_distribution']
    })


@app.route('/api/detections')
def get_detections():
    try:
        # 检查检测系统状态
        if not detection_enabled:
            return jsonify({
                'error': '通用检测系统未启用',
                'status': 'disabled',
                'last_detections': [],
                'detection_count': 0,
                'detection_stats': {},
                'current_objects': {},
                'category_distribution': {},
                'summary': '检测系统未启用'
            }), 503

        # 返回检测数据
        response_data = {
            'last_detections': detection_data['last_detections'],
            'detection_count': detection_data['detection_count'],
            'detection_stats': detection_data['detection_stats'],
            'detection_history': detection_data['detection_history'][-10:],  # 最近10条
            'current_objects': detection_data['detection_stats']['current_objects'],
            'category_distribution': detection_data['detection_stats']['category_distribution'],
            'summary': generate_detection_summary(detection_data['last_detections']),
            'status': 'active' if detection_data['last_detections'] else 'waiting',
            'last_update': detection_data['detection_stats'].get('last_update'),
            'system_enabled': detection_enabled
        }

        return jsonify(response_data)

    except Exception as e:
        print(f" 获取检测数据失败: {e}")
        return jsonify({
            'error': f'获取检测数据失败: {str(e)}',
            'status': 'error',
            'last_detections': [],
            'detection_count': 0
        }), 500


@app.route('/api/detections/categories')
def get_detection_categories():
    """获取检测类别统计"""
    return jsonify({
        'current_objects': detection_data['detection_stats']['current_objects'],
        'all_time_objects': detection_data['detection_stats']['object_counts'],
        'category_distribution': detection_data['detection_stats']['category_distribution'],
        'total_detections': detection_data['detection_stats']['total_detections'],
        'last_update': detection_data['detection_stats']['last_update']
    })


@app.route('/api/strawberry/status')
def get_strawberry_status():
    return jsonify({
        'enabled': strawberry_data['enabled'],
        'model_loaded': strawberry_data['model_loaded'],
        'model_path': strawberry_model_path,
        'conf_threshold': STRAWBERRY_CONF_THRESHOLD,
        'total_detections': strawberry_data['detection_stats']['total_detections'],
        'last_processing_time': strawberry_data['detection_stats']['processing_time'],
        'use_local_yolo': strawberry_data.get('use_local_yolo', False)
    })


@app.route('/api/strawberry/detections')
def get_strawberry_detections():
    try:
        # 检查草莓检测系统状态
        if not strawberry_data['enabled']:
            return jsonify({
                'error': '草莓检测系统未启用',
                'status': 'disabled',
                'last_detections': [],
                'detection_count': 0,
                'detection_stats': {},
                'ripeness_counts': {},
                'current_ripeness': {},
                'harvest_ready': 0,
                'quality_score': 0,
                'current_harvest_ready': 0,
                'current_quality_score': 0
            }), 503

        # 返回草莓检测数据
        response_data = {
            'last_detections': strawberry_data['last_detections'],
            'detection_count': strawberry_data['detection_count'],
            'detection_stats': strawberry_data['detection_stats'],
            'ripeness_counts': strawberry_data['detection_stats']['ripeness_counts'],
            'harvest_ready': strawberry_data['detection_stats']['harvest_ready'],
            'quality_score': strawberry_data['detection_stats']['quality_score'],
            'current_ripeness': strawberry_data['detection_stats']['current_ripeness'],
            'current_harvest_ready': strawberry_data['detection_stats']['current_harvest_ready'],
            'current_quality_score': strawberry_data['detection_stats']['current_quality_score'],
            'status': 'active' if strawberry_data['last_detections'] else 'waiting',
            'last_update': strawberry_data['detection_stats'].get('last_update'),
            'system_enabled': strawberry_data['enabled'],
            'model_loaded': strawberry_data['model_loaded'],
            'use_local_yolo': strawberry_data.get('use_local_yolo', False)
        }

        return jsonify(response_data)

    except Exception as e:
        print(f" 获取草莓检测数据失败: {e}")
        return jsonify({
            'error': f'获取草莓检测数据失败: {str(e)}',
            'status': 'error',
            'last_detections': [],
            'detection_count': 0
        }), 500


@app.route('/api/strawberry/toggle', methods=['POST'])
def toggle_strawberry_detection():
    try:
        strawberry_data['enabled'] = not strawberry_data['enabled']
        return jsonify({
            'success': True,
            'enabled': strawberry_data['enabled'],
            'message': f"草莓检测已{'启用' if strawberry_data['enabled'] else '禁用'}"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/strawberry/classes')
def get_strawberry_classes():
    """获取草莓类别配置信息"""
    return jsonify({
        'config_classes': STRAWBERRY_CLASSES,
        'model_classes': STRAWBERRY_MODEL_CLASSES,
        'class_count': len(STRAWBERRY_CLASSES),
        'model_class_count': len(STRAWBERRY_MODEL_CLASSES),
        'yaml_path': strawberry_yaml_path,
        'yaml_exists': os.path.exists(strawberry_yaml_path),
        'yaml_available': YAML_AVAILABLE
    })


@app.route('/api/strawberry/debug')
def get_strawberry_debug():
    """获取草莓检测调试信息 - 修复版"""
    model_classes = []
    model_names = {}

    if strawberry_model and hasattr(strawberry_model, 'names'):
        model_names = strawberry_model.names
        if isinstance(model_names, dict):
            model_classes = list(model_names.values())
        else:
            model_classes = list(model_names) if model_names else []

    return jsonify({
        'enabled': strawberry_data['enabled'],
        'model_loaded': strawberry_data['model_loaded'],
        'use_local_yolo': strawberry_data.get('use_local_yolo', False),
        'config_classes': STRAWBERRY_CLASSES,
        'model_classes': STRAWBERRY_MODEL_CLASSES,
        'model_names_dict': model_names,
        'class_count': len(STRAWBERRY_CLASSES),
        'model_class_count': len(STRAWBERRY_MODEL_CLASSES),
        'last_detections': strawberry_data['last_detections'][-5:],  # 最近5次检测
        'current_ripeness': strawberry_data['detection_stats']['current_ripeness'],
        'total_ripeness': strawberry_data['detection_stats']['ripeness_counts'],
        'model_path': strawberry_model_path,
        'yaml_path': strawberry_yaml_path,
        'yaml_exists': os.path.exists(strawberry_yaml_path),
        'processing_time': strawberry_data['detection_stats']['processing_time'],
        'total_detections': strawberry_data['detection_stats']['total_detections'],
        'conf_threshold': STRAWBERRY_CONF_THRESHOLD,
        'iou_threshold': STRAWBERRY_IOU_THRESHOLD
    })


@app.route('/api/sensors')
def get_sensors():
    try:
        # 检查传感器数据可用性
        sensor_status = {}
        has_data = False

        for sensor_type, data in sensor_data.items():
            if data['value'] is not None:
                has_data = True
                sensor_status[sensor_type] = 'active'
            else:
                sensor_status[sensor_type] = 'no_data'

        response_data = {
            'sensor_data': sensor_data,
            'message_count': message_count,
            'timestamp': time.time(),
            'status': 'active' if has_data else 'waiting',
            'sensor_status': sensor_status,
            'mqtt_connected': mqtt_client.is_connected() if mqtt_client else False
        }

        # 如果没有任何传感器数据，返回警告
        if not has_data:
            response_data['warning'] = '当前没有可用的传感器数据'

        return jsonify(response_data)

    except Exception as e:
        print(f" 获取传感器数据失败: {e}")
        return jsonify({
            'error': f'获取传感器数据失败: {str(e)}',
            'status': 'error',
            'sensor_data': sensor_data,
            'message_count': message_count,
            'timestamp': time.time()
        }), 500


@app.route('/api/status')
def get_system_status():
    return jsonify({
        'mqtt_connected': mqtt_client.is_connected() if mqtt_client else False,
        'message_count': message_count,
        'uptime': time.time() - start_time,
        'video_status': video_data['device_status'],
        'detection_enabled': detection_enabled,
        'yolov5_available': YOLOV5_AVAILABLE,
        'strawberry_enabled': strawberry_data['enabled'],
        'strawberry_model_loaded': strawberry_data['model_loaded'],
        'test_mode': video_data.get('test_mode', False),
        'detection_stats': detection_data['detection_stats'],
        'strawberry_stats': strawberry_data['detection_stats'],
        'sensor_status': {k: v['status'] for k, v in sensor_data.items()},
        'ai_status': {
            'enabled': deepseek_data['enabled'],
            'status': deepseek_data['status'],
            'total_messages': deepseek_data['total_messages'],
            'error_count': deepseek_data['error_count']
        },
        'strawberry_classes': {
            'config': STRAWBERRY_CLASSES,
            'model': STRAWBERRY_MODEL_CLASSES
        },
        'strawberry_engine': 'local_yolov5' if strawberry_data.get('use_local_yolo', False) else 'torch_hub',
        'version': 'Enhanced Strawberry Monitor v3.0 (本地YOLOv5版)'
    })


# ========== WebSocket事件处理 ==========
@socketio.on('connect')
def handle_connect():
    print(f" 新的增强系统客户端连接")

    # 发送系统状态
    emit('status_update', {
        'mqtt_connected': mqtt_client.is_connected() if mqtt_client else False,
        'message_count': message_count,
        'uptime': time.time() - start_time,
        'video_status': video_data['device_status'],
        'detection_enabled': detection_enabled,
        'strawberry_enabled': strawberry_data['enabled'],
        'test_mode': video_data.get('test_mode', False),
        'ai_enabled': deepseek_data['enabled'],
        'ai_status': deepseek_data['status']
    })

    # 发送当前传感器数据
    emit('all_sensors', sensor_data)

    # 发送当前通用检测数据
    if detection_data['last_detections']:
        emit('detection_update', {
            'detections': detection_data['last_detections'],
            'detection_count': detection_data['detection_count'],
            'timestamp': detection_data['detection_stats']['last_update'],
            'stats': detection_data['detection_stats'],
            'current_objects': detection_data['detection_stats']['current_objects'],
            'category_distribution': detection_data['detection_stats']['category_distribution'],
            'summary': generate_detection_summary(detection_data['last_detections'])
        })

    # 发送当前草莓检测数据
    if strawberry_data['last_detections']:
        emit('strawberry_update', {
            'detections': strawberry_data['last_detections'],
            'detection_count': strawberry_data['detection_count'],
            'timestamp': strawberry_data['detection_stats']['last_update'],
            'stats': strawberry_data['detection_stats']
        })

    # 发送AI状态
    emit('ai_status', {
        'enabled': deepseek_data['enabled'],
        'status': deepseek_data['status'],
        'total_messages': deepseek_data['total_messages'],
        'error_count': deepseek_data['error_count']
    })

    # 注意：这里不再推送video_frame事件
    # 前端应该监听general_frame和strawberry_frame事件来接收标注后的图像


@socketio.on('ai_message')
def handle_ai_message(data):
    """处理来自前端的AI消息请求"""
    try:
        message = data.get('message', '').strip()
        if not message:
            emit('ai_response', {
                'message': message,
                'response': None,
                'error': '消息不能为空',
                'timestamp': time.time()
            })
            return

        if not deepseek_data['enabled']:
            emit('ai_response', {
                'message': message,
                'response': None,
                'error': 'DeepSeek R1 未连接',
                'timestamp': time.time()
            })
            return

        print(f" WebSocket AI请求: {message[:50]}...")

        # 异步处理AI请求
        process_ai_request_async(message, data.get('context'))

        # 立即返回处理状态
        emit('ai_processing', {
            'message': message,
            'status': 'processing',
            'timestamp': time.time()
        })

    except Exception as e:
        print(f" WebSocket AI消息处理错误: {e}")
        emit('ai_response', {
            'message': data.get('message', ''),
            'response': None,
            'error': f'处理异常: {str(e)}',
            'timestamp': time.time()
        })


@socketio.on('ai_test_connection')
def handle_ai_test():
    """测试AI连接"""
    success = test_deepseek_connection()
    emit('ai_test_result', {
        'success': success,
        'status': deepseek_data['status'],
        'enabled': deepseek_data['enabled'],
        'timestamp': time.time()
    })


@socketio.on('request_detections')
def handle_request_detections():
    emit('detection_update', {
        'detections': detection_data['last_detections'],
        'detection_count': detection_data['detection_count'],
        'timestamp': detection_data['detection_stats']['last_update'],
        'stats': detection_data['detection_stats'],
        'current_objects': detection_data['detection_stats']['current_objects'],
        'category_distribution': detection_data['detection_stats']['category_distribution'],
        'summary': generate_detection_summary(detection_data['last_detections'])
    })


@socketio.on('request_strawberry_detections')
def handle_request_strawberry_detections():
    emit('strawberry_update', {
        'detections': strawberry_data['last_detections'],
        'detection_count': strawberry_data['detection_count'],
        'timestamp': strawberry_data['detection_stats']['last_update'],
        'stats': strawberry_data['detection_stats']
    })


@socketio.on('request_sensors')
def handle_request_sensors():
    emit('all_sensors', sensor_data)


@socketio.on('request_ai_history')
def handle_request_ai_history():
    emit('ai_history', {
        'chat_history': deepseek_data['chat_history'][-20:],
        'total_messages': deepseek_data['total_messages'],
        'timestamp': time.time()
    })


def create_templates_folder():
    """检查模板文件夹和文件"""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    template_file = os.path.join(templates_dir, 'strawberry_enhanced.html')
    if not os.path.exists(template_file):
        print(f" 未找到前端模板: {template_file}")
        print(" 请将前端HTML代码保存为: templates/strawberry_enhanced.html")
        return False

    return True


# ========== 主程序入口 ==========
if __name__ == '__main__':
    print_header()

    # 初始化通用YOLOv5
    if init_yolov5():
        print(" 通用YOLOv5检测引擎准备就绪")
    else:
        print(" 通用YOLOv5初始化失败，程序退出")
        exit(1)

    #  初始化本地YOLOv5草莓检测模型
    if init_strawberry_model():
        engine_type = "本地YOLOv5" if strawberry_data.get('use_local_yolo', False) else "torch.hub"
        print(f" 草莓检测引擎准备就绪 ({engine_type})")
        print(f" 配置类别: {STRAWBERRY_CLASSES}")
        print(f" 模型类别: {STRAWBERRY_MODEL_CLASSES}")
    else:
        print(" 草莓检测初始化失败，程序退出")
        exit(1)

    # 初始化DeepSeek R1
    print(" 正在测试DeepSeek R1连接...")
    deepseek_data['api_url'] = DEEPSEEK_API_URL
    deepseek_data['model_name'] = DEEPSEEK_MODEL

    if test_deepseek_connection():
        print(" DeepSeek R1 AI助手准备就绪")
    else:
        print(" DeepSeek R1 连接失败，AI功能将不可用")

    # 检查前端文件
    if not create_templates_folder():
        print("\n 缺少前端文件，请保存前端HTML文件")
        print(" 将前端代码保存为: templates/strawberry_enhanced.html")
        input("按回车键退出...")
        exit(1)

    # 启动MQTT客户端
    mqtt_client = create_mqtt_client()
    if mqtt_client:
        try:
            mqtt_client.loop_start()
            print("📡 MQTT客户端已启动")

            print("🌐 正在启动本地YOLOv5版Web服务器...")
            print("🍓 请在浏览器中访问: http://localhost:5000")
            print("📷 请在Fibo上启动图片发送程序")
            print("📊 请确保传感器数据正常发送")
            print("🧠 AI对话功能已集成")
            print("🍓 本地YOLOv5草莓检测功能已启用")
            print("🔧 智能回退机制：本地YOLOv5 → torch.hub → 模拟检测")
            print("🔧 按 Ctrl+C 停止服务器")
            print("🍓" * 50)

            socketio.run(
                app,
                debug=False,
                host='0.0.0.0',
                port=5000,
                allow_unsafe_werkzeug=True,
                log_output=False
            )

        except KeyboardInterrupt:
            print("\n\n 收到停止信号，正在关闭系统...")
        except Exception as e:
            print(f"\n 服务器运行错误: {e}")
        finally:
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            # 关闭线程池
            executor.shutdown(wait=True)
            print(" 本地YOLOv5监控系统已退出")
    else:
        print("\n 无法启动MQTT客户端")