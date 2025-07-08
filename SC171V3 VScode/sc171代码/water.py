#!/usr/bin/env python3
import os
import sys

# —— 配置区域 —— 
GPIO_CTRL_PATH = "/sys/class/sc171_gpio_class/sc171_gpio_dev/sc171_gpio_ctrl"
GPIO_PIN       = 3  # 浇水控制的 GPIO 编号
# ————————————

def set_level(level: int):
    """
    设置 GPIO_PIN 的电平：
      level = 1 -> 高电平 (开始浇水)
      level = 0 -> 低电平 (停止浇水)
    """
    try:
        with open(GPIO_CTRL_PATH, "w") as f:
            f.write(f"{GPIO_PIN},{level}\n")
        return True
    except Exception as e:
        print(f"[ERROR] 无法设置 GPIO{GPIO_PIN} = {level}: {e}")
        return False

def start_watering():
    """开始浇水 - GPIO拉高"""
    success = set_level(1)
    if success:
        print(f"[WATER] 开始浇水 - GPIO{GPIO_PIN} 拉高")
    return success

def stop_watering():
    """停止浇水 - GPIO拉低"""
    success = set_level(0)
    if success:
        print(f"[WATER] 停止浇水 - GPIO{GPIO_PIN} 拉低")
    return success

def cleanup():
    """
    程序退出前将 GPIO 拉低，确保清理干净
    """
    set_level(0)
    print(f"[WATER] 清理完成 - GPIO{GPIO_PIN} 已拉低")

def check_gpio_available():
    """检查GPIO控制文件是否可用"""
    if not os.path.exists(GPIO_CTRL_PATH):
        print(f"[ERROR] 找不到 GPIO 控制文件: {GPIO_CTRL_PATH}")
        return False
    return True

# 如果直接运行此文件，则执行原来的持续拉高逻辑
if __name__ == "__main__":
    import time
    import signal
    
    def signal_cleanup(signum=None, frame=None):
        cleanup()
        sys.exit(0)
    
    if not check_gpio_available():
        sys.exit(1)
    
    # 捕获信号
    signal.signal(signal.SIGINT, signal_cleanup)
    signal.signal(signal.SIGTERM, signal_cleanup)
    
    # 持续拉高GPIO3
    start_watering()
    print(f"[INFO] 已将 GPIO{GPIO_PIN} 持续拉高，按 Ctrl+C 停止")
    
    try:
        while True:
            time.sleep(1)
    finally:
        cleanup()