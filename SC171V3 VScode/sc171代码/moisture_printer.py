# 文件：moisture_printer.py

import time
import signal
import sys
from moisture_reader import read_voltage_uv, read_moisture_percent
import os

def display_moisture(poll_interval: float = 1.0):
    """
    每隔 poll_interval 秒读取并打印一次电压和湿度百分比。
    按 Ctrl+C 退出。
    """
    def _cleanup(signum=None, frame=None):
        print("\n[INFO] 停止测量，退出。")
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    print("开始实时测量湿度 (按 Ctrl+C 停止)…")
    while True:
        uv = read_voltage_uv()
        humidity = read_moisture_percent()
        print(f"电压: {uv:>7d} µV | 湿度: {humidity:5.1f}%")
        cmd = "mosquitto_pub -h 192.168.187.81 -p 1883 -t \"/sensor/humidity\" -m \"{{\"humidity\":{humidity}}}\"".format(humidity=humidity)
        os.system(cmd)
        time.sleep(poll_interval)
