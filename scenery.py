# 使用场景定义函数
import numpy as np

def scenario_video_streaming(t):
    """视频流场景 - 使用阶跃式突变模拟真实使用变化"""
    # 基础参数
    base_brightness = 0.7
    base_cpu = 0.3
    base_data_rate = 2.0
    
    # 使用多个阶跃变化模拟真实使用场景
    # 周期300秒(5分钟)，模拟用户操作、广告、切换视频等突变
    cycle = t % 300
    
    if cycle < 60:
        # 正常观看
        brightness = 0.7
        cpu_usage = 0.3
        data_rate = 2.0
    elif cycle < 90:
        # 广告时段（亮度高、CPU低）
        brightness = 0.9
        cpu_usage = 0.2
        data_rate = 1.5
    elif cycle < 150:
        # 高清片段（CPU和数据率升高）
        brightness = 0.7
        cpu_usage = 0.5
        data_rate = 3.0
    elif cycle < 180:
        # 暂停/菜单操作（低功耗）
        brightness = 0.5
        cpu_usage = 0.15
        data_rate = 0.5
    elif cycle < 240:
        # 恢复观看
        brightness = 0.7
        cpu_usage = 0.35
        data_rate = 2.5
    else:
        # 缓冲/加载（高数据率）
        brightness = 0.6
        cpu_usage = 0.4
        data_rate = 4.0
    
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
