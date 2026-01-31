import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]  # 父目录的上级是 MCMA
sys.path.insert(0, str(project_root))
from model import SmartphoneBatteryModel
from scenery import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class EntropyWeightAnalyzer:
    """熵权法分析器 - 评估各组件对续航的影响程度"""
    
    def __init__(self):
        self.battery = SmartphoneBatteryModel()
        
    def collect_data(self):
        """
        收集4个场景（视频、游戏、导航、空闲）的数据
        返回：功耗数据和续航时间
        """
        scenarios = {
            '视频流': scenario_video_streaming,
            '游戏': scenario_gaming,
            '导航': scenario_navigation,
            '空闲': scenario_free
        }
        
        # 存储数据矩阵
        data_matrix = []
        scenario_names = []
        discharge_times = []
        
        print("="*70)
        print("收集场景数据用于熵权法分析")
        print("="*70)
        
        for name, func in scenarios.items():
            scenario = func(0)
            
            # 提取各组件参数（原始值，不是功耗）
            screen_brightness = scenario.get('brightness', 0.0) if scenario.get('screen_on', False) else 0.0
            cpu_usage = scenario.get('cpu_usage', 0.0)
            data_rate = scenario.get('data_rate', 0.0)
            gps_on = 1.0 if scenario.get('gps_on', False) else 0.0
            
            # 计算功耗
            components = self._calculate_power_components(scenario)
            total_power = sum(components.values())
            
            # 估算满电续航时间（简化计算）
            battery_energy = self.battery.Q0 * self.battery.V_nom  # mAh * V = mWh
            discharge_time = battery_energy / (total_power * 1000)  # 小时
            
            data_matrix.append([
                components['屏幕'],
                components['CPU'],
                components['网络'],
                components['GPS'],
                components['基础']
            ])
            
            scenario_names.append(name)
            discharge_times.append(discharge_time)
            
            print(f"\n【{name}】")
            print(f"  屏幕功耗: {components['屏幕']:.3f} W")
            print(f"  CPU功耗: {components['CPU']:.3f} W")
            print(f"  网络功耗: {components['网络']:.3f} W")
            print(f"  GPS功耗: {components['GPS']:.3f} W")
            print(f"  基础功耗: {components['基础']:.3f} W")
            print(f"  总功耗: {total_power:.3f} W")
            print(f"  预计续航: {discharge_time:.2f} 小时")
        
        print("\n" + "="*70)
        
        return np.array(data_matrix), scenario_names, discharge_times
    
    def _calculate_power_components(self, scenario):
        """计算各组件功耗"""
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
            components['CPU'] = self.battery.P_cpu_idle + cpu_usage * self.battery.P_cpu_B * (self.battery.P_cpu_f ** 3)
        
        # 网络功耗
        if 'data_rate' in scenario:
            data_rate = max(0.0, float(scenario.get('data_rate', 0.0)))
            components['网络'] = self.battery.P_net_idle + self.battery.beta * data_rate
        
        # GPS功耗
        if scenario.get('gps_on', False):
            components['GPS'] = self.battery.P_gps
        
        return components
    
    def calculate_entropy_weights(self, data_matrix):
        """
        熵权法计算各指标权重
        
        步骤：
        1. 数据标准化
        2. 计算信息熵
        3. 计算信息效用值
        4. 计算权重
        
        参数：
            data_matrix: n×m 矩阵，n个方案，m个指标
        
        返回：
            weights: m维权重向量
        """
        n, m = data_matrix.shape
        
        print("\n" + "="*70)
        print("熵权法计算过程")
        print("="*70)
        
        # 步骤1: 数据标准化（归一化）
        # 将每个指标转换到[0,1]区间
        normalized = np.zeros_like(data_matrix)
        for j in range(m):
            col = data_matrix[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            if max_val - min_val > 1e-10:
                normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, j] = 1.0  # 如果所有值相同
        
        # 避免log(0)，将0替换为极小值
        normalized = np.where(normalized == 0, 1e-10, normalized)
        
        print("\n1. 标准化数据矩阵:")
        print(normalized)
        
        # 步骤2: 计算各指标的信息熵
        entropy = np.zeros(m)
        k = 1.0 / np.log(n)  # 熵的系数
        
        for j in range(m):
            # 计算比重
            p = normalized[:, j] / np.sum(normalized[:, j])
            # 计算信息熵
            entropy[j] = -k * np.sum(p * np.log(p))
        
        print("\n2. 各指标的信息熵 e_j:")
        component_names = ['屏幕', 'CPU', '网络', 'GPS', '基础']
        for i, name in enumerate(component_names):
            print(f"   {name}: {entropy[i]:.6f}")
        
        # 步骤3: 计算信息效用值（差异系数）
        # d_j = 1 - e_j，熵越大，差异越小，权重越小
        divergence = 1 - entropy
        
        print("\n3. 信息效用值 d_j = 1 - e_j:")
        for i, name in enumerate(component_names):
            print(f"   {name}: {divergence[i]:.6f}")
        
        # 步骤4: 计算权重
        # w_j = d_j / sum(d_j)
        weights = divergence / np.sum(divergence)
        
        print("\n4. 熵权法计算的权重 w_j:")
        for i, name in enumerate(component_names):
            print(f"   {name}: {weights[i]:.6f} ({weights[i]*100:.2f}%)")
        
        print("\n" + "="*70)
        
        return weights
    
    def analyze_impact(self, data_matrix, weights, scenario_names):
        """
        分析各组件对续航的综合影响
        """
        print("\n" + "="*70)
        print("各场景综合影响评分（加权求和）")
        print("="*70)
        
        component_names = ['屏幕', 'CPU', '网络', 'GPS', '基础']
        
        # 计算每个场景的综合得分
        scores = np.dot(data_matrix, weights)
        
        for i, name in enumerate(scenario_names):
            print(f"\n【{name}】综合影响评分: {scores[i]:.4f}")
            print("  各组件贡献:")
            for j, comp in enumerate(component_names):
                contribution = data_matrix[i, j] * weights[j]
                print(f"    {comp}: {data_matrix[i, j]:.3f}W × {weights[j]:.4f} = {contribution:.4f}")
        
        print("\n" + "="*70)
        
        return scores


def plot_entropy_weights(weights, component_names):
    """绘制权重柱状图"""
    print("\n生成熵权法权重图...")
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0e27')
    ax.set_facecolor('#0a0e27')
    
    x = np.arange(len(component_names))
    colors = ['#00D9FF', '#FFD700', '#4ECDC4', '#FF6B6B', '#95E1D3']
    
    bars = ax.bar(x, weights, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{weight:.4f}\n({weight*100:.2f}%)',
               ha='center', va='bottom', fontsize=11, 
               fontweight='bold', color='white')
    
    ax.set_xlabel('功耗组件', fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel('熵权法权重', fontsize=13, fontweight='bold', color='white')
    ax.set_title('各组件对续航影响程度（熵权法）\nEntropy Weight Method - Component Impact Analysis', 
                fontsize=16, fontweight='bold', color='white', pad=20,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a2332', 
                         edgecolor='#00D9FF', linewidth=2))
    ax.set_xticks(x)
    ax.set_xticklabels(component_names, fontsize=12, color='white', fontweight='bold')
    ax.tick_params(colors='white', labelsize=11)
    ax.set_ylim(0, max(weights) * 1.3)
    
    # 网格
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', color='#00D9FF')
    
    # 设置坐标轴颜色
    ax.spines['bottom'].set_color('#00D9FF')
    ax.spines['left'].set_color('#00D9FF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # 添加说明文字
    textstr = '权重越大，对续航影响越显著\n熵权法基于数据差异性自动赋权'
    props = dict(boxstyle='round', facecolor='#1a2332', edgecolor='#00FF94', 
                linewidth=2, alpha=0.9)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=props, color='#00FF94', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('entropy_weights.png', dpi=300, bbox_inches='tight',
                facecolor='#0a0e27', edgecolor='none')
    print("熵权法权重图已保存为 entropy_weights.png")
    plt.show()


# 主程序
if __name__ == "__main__":
    print("="*70)
    print("熵权法分析系统 - 各组件对续航影响程度评估")
    print("Entropy Weight Method - Battery Life Impact Analysis")
    print("="*70)
    
    # 创建分析器
    analyzer = EntropyWeightAnalyzer()
    
    # 收集数据（4个场景：视频、游戏、导航、空闲）
    data_matrix, scenario_names, discharge_times = analyzer.collect_data()
    
    # 使用熵权法计算权重
    component_names = ['屏幕', 'CPU', '网络', 'GPS', '基础']
    weights = analyzer.calculate_entropy_weights(data_matrix)
    
    # 分析综合影响
    scores = analyzer.analyze_impact(data_matrix, weights, scenario_names)
    
    # 生成可视化图表
    plot_entropy_weights(weights, component_names)
   
    # 输出结论
    print("\n" + "="*70)
    print("熵权法分析结论")
    print("="*70)
    
    # 找出影响最大和最小的组件
    max_idx = np.argmax(weights)
    min_idx = np.argmin(weights)
    
    print(f"\n📊 影响程度排序（从大到小）：")
    sorted_indices = np.argsort(weights)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. {component_names[idx]}: {weights[idx]:.4f} ({weights[idx]*100:.2f}%)")
    
    print(f"\n🔥 对续航影响最大的组件: {component_names[max_idx]} (权重: {weights[max_idx]:.4f})")
    print(f"💡 对续航影响最小的组件: {component_names[min_idx]} (权重: {weights[min_idx]:.4f})")
    
    # 场景续航排序
    print(f"\n⏱️ 续航时间排序（从长到短）：")
    sorted_time_indices = np.argsort(discharge_times)[::-1]
    for rank, idx in enumerate(sorted_time_indices, 1):
        print(f"  {rank}. {scenario_names[idx]}: {discharge_times[idx]:.2f}小时")
    
    print("\n" + "="*70)
    print("分析完成！生成的文件:")
    print("  1. entropy_weights.png - 熵权法权重柱状图")
    print("  2. component_impact_heatmap.png - 组件影响热力图")
    print("  3. power_vs_discharge_comparison.png - 功耗与续航对比图")
    print("="*70)
