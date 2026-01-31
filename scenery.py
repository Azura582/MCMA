# 使用场景定义函数
import numpy as np

def scenario_video_streaming(t):
    """视频流场景 - 使用平滑的周期性波动模拟真实使用"""
    # 基础参数
    base_brightness = 0.7
    base_cpu = 0.3
    base_data_rate = 2.0
    
    # 使用正弦函数模拟平滑的周期性变化
    # 周期约1小时,模拟用户注意力波动和内容复杂度变化
    period = 3600  # 1小时周期
    phase = 2 * np.pi * t / period
    
    # 亮度变化: 0.5-0.9之间平滑波动
    brightness = base_brightness + 0.2 * np.sin(phase)
    brightness = np.clip(brightness, 0.5, 0.9)
    
    # CPU使用率: 0.2-0.4之间波动(视频解码负载变化)
    cpu_usage = base_cpu + 0.1 * np.sin(phase + np.pi/4)
    cpu_usage = np.clip(cpu_usage, 0.2, 0.4)
    
    # 数据率: 1.5-2.5 Mbps之间波动(码率自适应)
    data_rate = base_data_rate + 0.5 * np.sin(phase + np.pi/2)
    data_rate = np.clip(data_rate, 1.5, 2.5)
    
    scenario = {
        'screen_on': True,
        'brightness': brightness,
        'cpu_usage': cpu_usage,
        'data_rate': data_rate,
        'gps_on': False,
        'T_amb': 298.15  # 25°C
    }
    return scenario

def scenario_gaming(t):
    """游戏场景"""
    scenario = {
        'screen_on': True,
        'brightness': 1.0,
        'cpu_usage': 0.8,
        'data_rate': 0.1,
        'gps_on': False,
        'T_amb': 298.15
    }
    return scenario

def scenario_navigation(t):
    """导航场景"""
    scenario = {
        'screen_on': True,
        'brightness': 0.8,
        'cpu_usage': 0.4,
        'data_rate': 0.5,
        'gps_on': True,
        'T_amb': 298.15
    }
    return scenario
def scenario_free(t):
    """空闲场景"""
    scenario = {
        'screen_on': False,
        'brightness': 0.0,
        'cpu_usage': 0.0,  # 空闲时CPU使用率接近0
        'data_rate': 0.0,   # 待机时无数据传输
        'gps_on': False,
        'T_amb': 298.15
    }
    return scenario
def scenario_cold_weather(t):
    """低温场景"""
    scenario = scenario_video_streaming(t)
    scenario['T_amb'] = 273.15  # -10°C
    return scenario
