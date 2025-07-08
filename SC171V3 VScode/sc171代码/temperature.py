# temperature.py
#!/usr/bin/env python3
"""
温度传感器模块（UART + STM32）
子函数接口：connect_serial, read_temperature, close_serial
"""

import serial
import re

_serial_conn = None

def connect_serial(port: str = '/dev/ttyHS1', baudrate: int = 115200) -> bool:
    """
    打开并配置串口，清空缓冲区，返回连接是否成功。
    """
    global _serial_conn
    try:
        if _serial_conn and _serial_conn.is_open:
            _serial_conn.close()
        _serial_conn = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            write_timeout=0.1,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
            inter_byte_timeout=None
        )
        _serial_conn.reset_input_buffer()
        _serial_conn.reset_output_buffer()
        print(f"✅ 串口已打开: {port}@{baudrate}")
        return True
    except Exception as e:
        print(f"串口连接失败: {e}")
        _serial_conn = None
        return False

def extract_temperature(text: str):
    """
    从串口文本中提取温度，匹配 'DATA:xx.x' 或 'TEMP:xx.x'。
    """
    m = re.search(r'(?:DATA|TEMP):([\d.]+)', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None

def read_temperature() -> float:
    """
    读取并返回温度（°C），读取失败时返回 None。
    """
    global _serial_conn
    if _serial_conn is None or not _serial_conn.is_open:
        if not connect_serial():
            return None
    raw = _serial_conn.readline()
    text = raw.decode('utf-8', errors='ignore').strip()
    return extract_temperature(text)

def close_serial() -> None:
    """
    关闭串口连接。
    """
    global _serial_conn
    if _serial_conn:
        _serial_conn.close()
        _serial_conn = None
        print("串口已关闭")
