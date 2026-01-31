import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  model import  SmartphoneBatteryModel
from scenery import *
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
# 扩展功能：蒙特卡洛不确定性分析
def monte_carlo_analysis(battery, scenario_func, n_simulations=1000):
    """蒙特卡洛不确定性分析"""
    np.random.seed(42)
    
    empty_times = []
    
    for i in range(n_simulations):
        # 随机扰动参数（±10%）
        battery_temp = SmartphoneBatteryModel()
        
        # 随机扰动
        battery_temp.Q0 *= np.random.uniform(0.9, 1.1)
        battery_temp.R0 *= np.random.uniform(0.9, 1.1)
        battery_temp.P_screen_max *= np.random.uniform(0.9, 1.1)
        battery_temp.P_cpu_max *= np.random.uniform(0.9, 1.1)
        
        # 模拟（四个初始状态：SOC, T_batt, U1, U2）
        sol = battery_temp.simulate((0, 24*3600), [1.0, 298.15, 0.0, 0.0], scenario_func)
        empty_time = battery_temp.find_empty_time(sol)
        empty_times.append(empty_time/3600)
    
    empty_times = np.array(empty_times)
    
    print("\n=== 蒙特卡洛分析结果 ===")
    print(f"模拟次数: {n_simulations}")
    print(f"平均放空时间: {np.mean(empty_times):.2f} 小时")
    print(f"标准差: {np.std(empty_times):.2f} 小时")
    print(f"95%置信区间: [{np.percentile(empty_times, 2.5):.2f}, "
          f"{np.percentile(empty_times, 97.5):.2f}] 小时")
    
    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(empty_times, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(empty_times), color='red', linestyle='--', 
                label=f'均值 = {np.mean(empty_times):.2f}h')
    plt.xlabel('放空时间 (小时)')
    plt.ylabel('频数')
    plt.title('蒙特卡洛模拟：放空时间分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('monte_carlo_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return empty_times

