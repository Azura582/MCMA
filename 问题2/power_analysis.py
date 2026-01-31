import numpy as np
import matplotlib.pyplot as plt
from model import SmartphoneBatteryModel
from scenery import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

class PowerComponentAnalyzer:
    """功耗组件分析器"""
    
    def __init__(self):
        self.battery = SmartphoneBatteryModel()
        
    def analyze_power_components(self, scenario_func, scenario_name):
        """
        分析单个场景下各组件的功耗
        
        返回:
            dict: {'屏幕': power, 'CPU': power, '网络': power, 'GPS': power, '基础': power}
        """
        # 获取场景参数（t=0时刻）
        scenario = scenario_func(0)
        
        components = {
            '屏幕': 0.0,
            'CPU': 0.0,
            '网络': 0.0,
            'GPS': 0.0,
            '基础': self.battery.P_base
        }
        
        # 屏幕功耗
        if scenario.get('screen_on', False):
            brightness = float(np.clip(scenario.get('brightness', 0.5), 0.0, 1.0))
            components['屏幕'] = self.battery.P_a * brightness * self.battery.P_refresh * self.battery.P_screen_square
        
        # CPU功耗
        if 'cpu_usage' in scenario:
            cpu_usage = float(np.clip(scenario.get('cpu_usage', 0.0), 0.0, 1.0))
            cpu_power = self.battery.P_cpu_idle + cpu_usage * self.battery.P_cpu_B * (self.battery.P_cpu_f ** 3)
            components['CPU'] = cpu_power
        
        # 网络功耗
        if 'data_rate' in scenario:
            data_rate = max(0.0, float(scenario.get('data_rate', 0.0)))
            net_power = self.battery.P_net_idle + self.battery.beta * data_rate
            components['网络'] = net_power
        
        # GPS功耗
        if scenario.get('gps_on', False):
            components['GPS'] = self.battery.P_gps
        
        # 计算总功耗
        total_power = sum(components.values())
        
        return components, total_power
    
    def analyze_all_scenarios(self):
        """分析所有场景的功耗组件"""
        scenarios = {
            '视频流': scenario_video_streaming,
            '游戏': scenario_gaming,
            '导航': scenario_navigation,
            '低温视频流': scenario_cold_weather,
            '空闲': scenario_free
        }
        
        results = {}
        
        print("="*70)
        print("各场景功耗组件分析 (满电状态)")
        print("="*70)
        
        for name, func in scenarios.items():
            components, total = self.analyze_power_components(func, name)
            results[name] = components
            
            print(f"\n【{name}】")
            print(f"  总功耗: {total:.3f} W")
            for comp, power in components.items():
                percentage = (power / total * 100) if total > 0 else 0
                print(f"  - {comp}: {power:.3f} W ({percentage:.1f}%)")
        
        print("\n" + "="*70)
        
        return results


def plot_radar_chart(results):
    """
    绘制科技感雷达图
    
    参数:
        results: dict, {场景名: {组件: 功耗}}
    """
    print("\n生成科技感雷达图...")
    
    # 组件列表（雷达图的各个轴）
    components = ['屏幕', 'CPU', '网络', 'GPS', '基础']
    num_components = len(components)
    
    # 场景列表
    scenarios = list(results.keys())
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_components, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 创建图形（使用深色背景增加科技感）
    fig = plt.figure(figsize=(14, 10), facecolor='#0a0e27')
    ax = fig.add_subplot(111, projection='polar', facecolor='#0a0e27')
    
    # 定义渐变色系（科技感配色）
    colors = ['#00D9FF', '#FF00E6', '#00FF94', '#FFD700', '#FF6B6B']
    
    # 为每个场景绘制雷达图
    for idx, scenario in enumerate(scenarios):
        values = [results[scenario][comp] for comp in components]
        values += values[:1]  # 闭合图形
        
        # 绘制填充区域
        ax.plot(angles, values, 'o-', linewidth=2.5, 
                label=scenario, color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # 设置雷达图的标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(components, fontsize=13, fontweight='bold', color='white')
    
    # 设置网格样式（科技感）
    ax.grid(True, color='#1a2332', linewidth=1.5, linestyle='-', alpha=0.7)
    ax.spines['polar'].set_color('#00D9FF')
    ax.spines['polar'].set_linewidth(2)
    
    # 设置刻度颜色
    ax.tick_params(colors='white', labelsize=11)
    
    # 设置径向刻度
    max_power = max([max([results[s][c] for c in components]) for s in scenarios])
    ax.set_ylim(0, max_power * 1.2)
    ax.set_yticks(np.linspace(0, max_power * 1.2, 5))
    ax.set_yticklabels([f'{v:.2f}W' for v in np.linspace(0, max_power * 1.2, 5)], 
                       fontsize=10, color='#00D9FF', fontweight='bold')
    
    # 添加标题（科技感字体）
    ax.set_title('各场景功耗组件雷达分析图\nPower Component Radar Analysis', 
                 fontsize=18, fontweight='bold', color='white', pad=30,
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a2332', 
                          edgecolor='#00D9FF', linewidth=2))
    
    # 添加图例（优化位置和样式）
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                      fontsize=12, frameon=True, shadow=False, 
                      facecolor='#1a2332', edgecolor='#00D9FF', 
                      framealpha=0.9, labelcolor='white')
    legend.get_frame().set_linewidth(2)
    
    # 添加水印/标注
    fig.text(0.5, 0.02, 'Smartphone Battery Model - Power Analysis System', 
            ha='center', fontsize=10, color='#00D9FF', alpha=0.7, 
            style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('power_radar_chart.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0e27', edgecolor='none')
    print("科技感雷达图已保存为 power_radar_chart.png")
    plt.show()

