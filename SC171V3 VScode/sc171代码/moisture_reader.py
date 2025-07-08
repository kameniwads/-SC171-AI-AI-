# 文件：moisture_reader.py

"""
moisture_reader 模块
提供从外部 ADC 读取电压并转换为湿度百分比的功能。
"""

# 根据你的硬件修改这条路径
ADC_SYSFS_PATH = "/sys/bus/iio/devices/iio:device0/in_voltage_pm7325_rain_input"

def read_voltage_uv() -> int:
    """
    读取 ADC 通道的原始电压值，返回单位：微伏 (µV)。
    """
    with open(ADC_SYSFS_PATH, "r") as f:
        return int(f.read().strip())

def read_moisture_percent(ref_uv: int = 1_890_000, dry_uv: int = 1_090_000) -> float:

    uv = read_voltage_uv()  # 读取当前电压值
    
    # 计算湿度：最大电压（最干）对应 100%，最小电压（最湿）对应 0%
    if uv >= ref_uv:  # 如果电压大于或等于最湿电压，湿度为 0%
        return 0.0
    if uv <= dry_uv:  # 如果电压小于或等于最干电压，湿度为 100%
        return 100.0
    
    # 计算湿度百分比
    humidity = ((ref_uv - uv) / (ref_uv - dry_uv)) * 100
    return humidity

