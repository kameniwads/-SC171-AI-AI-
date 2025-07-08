# light.py
from typing import Tuple

# ADC 设备文件路径
ADC_SYSFS_PATH = "/sys/bus/iio/devices/iio:device0/in_voltage_pm7325_light_input"
# ADC 理论最大电压（µV）和对应的最大光照（lux）
MAX_UV = 1_900_000
MAX_LUX = 20_000.0

def read_light_voltage_uv(path: str = ADC_SYSFS_PATH) -> int:
    """
    读取 ADC sysfs 文件，返回光照通道原始电压值（µV）。
    """
    with open(path, "r") as f:
        return int(f.read().strip())

def voltage_to_lux(uv: int, max_uv: int = MAX_UV, max_lux: float = MAX_LUX) -> float:
    """
    将电压值映射到光照强度 (lux)：
      0 µV   -> max_lux
      max_uv -> 0 lux
    """
    uv = max(0, min(uv, max_uv))
    return (1 - uv / max_uv) * max_lux

def read_light_lux() -> Tuple[int, float]:
    """
    一步完成读取并转换，返回 (uv, lux)。
    """
    uv = read_light_voltage_uv()
    lux = voltage_to_lux(uv)
    return uv, lux

if __name__ == "__main__":
    uv, lux = read_light_lux()
    print(f"电压: {uv:>7d} µV | 光照: {lux:6.1f} lux")
