#!/usr/bin/env python3
import time
import os

from light import read_light_lux
from moisture_reader import read_moisture_percent
from temperature import connect_serial, read_temperature, close_serial  # 新增引入
import water  # 导入浇水控制模块

# MQTT 参数
MQTT_HOST = "192.168.187.81"
MQTT_PORT = 1883

# 浇水控制参数
HUMIDITY_LOW_THRESHOLD = 50.0   # 湿度低于此值开始浇水
HUMIDITY_HIGH_THRESHOLD = 70.0  # 湿度高于此值停止浇水

def publish_mqtt(topic: str, payload: str,
                 host: str = MQTT_HOST, port: int = MQTT_PORT):
    """
    通过 mosquitto_pub 发布一条 MQTT 消息。
    """
    cmd = f'mosquitto_pub -h {host} -p {port} -t "{topic}" -m \'{payload}\''
    os.system(cmd)

def start_watering():
    """开始浇水 - 使用现有water模块的set_level函数"""
    try:
        water.set_level(1)
        print(f"[WATER] 开始浇水 - GPIO拉高")
        return True
    except Exception as e:
        print(f"[ERROR] 无法启动浇水: {e}")
        return False

def stop_watering():
    """停止浇水 - 使用现有water模块的set_level函数"""
    try:
        water.set_level(0)
        print(f"[WATER] 停止浇水 - GPIO拉低")
        return True
    except Exception as e:
        print(f"[ERROR] 无法停止浇水: {e}")
        return False

def check_gpio_available():
    """检查GPIO控制文件是否可用"""
    gpio_path = "/sys/class/sc171_gpio_class/sc171_gpio_dev/sc171_gpio_ctrl"
    if not os.path.exists(gpio_path):
        print(f"[ERROR] 找不到 GPIO 控制文件: {gpio_path}")
        return False
    return True

def control_watering(humidity: float, is_watering: bool) -> bool:
    """
    根据湿度控制浇水
    返回当前浇水状态
    """
    if humidity < HUMIDITY_LOW_THRESHOLD and not is_watering:
        # 湿度过低且未在浇水 -> 开始浇水
        if start_watering():
            print(f"[AUTO] 湿度过低({humidity:.1f}% < {HUMIDITY_LOW_THRESHOLD}%) - 开始自动浇水")
            return True
        else:
            print(f"[ERROR] 无法启动浇水系统")
            return is_watering
    
    elif humidity > HUMIDITY_HIGH_THRESHOLD and is_watering:
        # 湿度充足且正在浇水 -> 停止浇水
        if stop_watering():
            print(f"[AUTO] 湿度充足({humidity:.1f}% > {HUMIDITY_HIGH_THRESHOLD}%) - 停止自动浇水")
            return False
        else:
            print(f"[ERROR] 无法停止浇水系统")
            return is_watering
    
    return is_watering

def cleanup():
    """清理函数 - 确保GPIO拉低"""
    try:
        water.set_level(0)
        print("[WATER] 清理完成 - GPIO已拉低")
    except Exception as e:
        print(f"[ERROR] 清理时出错: {e}")

def main(poll_interval: float = 1.0):
    # 检查GPIO是否可用
    if not check_gpio_available():
        print("[ERROR] GPIO控制不可用，程序退出")
        return
    
    # 初始化浇水状态
    is_watering = False
    
    # 在循环前建立温度串口连接
    connect_serial()

    print("开始实时测量光照、湿度和温度，并自动控制浇水 (按 Ctrl+C 停止)…")
    print(f"浇水控制: 湿度 < {HUMIDITY_LOW_THRESHOLD}% 开始浇水, 湿度 > {HUMIDITY_HIGH_THRESHOLD}% 停止浇水")
    
    try:
        while True:
            # 读取光照
            uv, lux = read_light_lux()
            # 读取湿度
            humidity = read_moisture_percent()
            # 读取温度
            temp = 25

            # 自动浇水控制
            is_watering = control_watering(humidity, is_watering)

            # 打印到终端
            temp_str = f"{temp:5.1f}°C" if temp is not None else " --.-°C"
            water_status = "浇水中" if is_watering else "待机"
            print(
                f"光照: {lux:6.1f} lux | "
                f"湿度: {humidity:6.1f}% | "
                f"温度: {temp_str} | "
                f"浇水: {water_status}"
            )

            # 发布到 MQTT
            publish_mqtt("/sensor/light",       f'{{"light":{lux:.1f}}}')
            publish_mqtt("/sensor/humidity",    f'{{"humidity":{humidity:.1f}}}')
            publish_mqtt("/sensor/watering",    f'{{"watering":{str(is_watering).lower()}}}')
            if temp is not None:
                publish_mqtt("/sensor/temperature", f'{{"temperature":{temp:.1f}}}')

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n已停止测量，正在清理...")
    finally:
        # 结束时关闭串口和GPIO
        close_serial()
        cleanup()
        print("程序退出完成。")

if __name__ == "__main__":
    main()