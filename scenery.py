# 使用场景定义函数
def scenario_video_streaming(t):
    """视频流场景"""
    scenario = {
        'screen_on': True,
        'brightness': 0.7,
        'cpu_usage': 0.3,
        'data_rate': 2.0,  # Mbps
        'gps_on': False,
        'T_amb': 298.15  # 25°C
    }
    # 模拟周期性变化
    if t % 3600 < 1800:  # 每半小时休息5分钟
        scenario['screen_on'] = False
        scenario['cpu_usage'] = 0.1
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
