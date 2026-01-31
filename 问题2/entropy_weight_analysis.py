"""
Entropy Weight Method Analysis - Component Impact on Total Energy Consumption
5 Scenarios, 5 Components, Donut Chart Visualization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from model import SmartphoneBatteryModel
from scenery import (scenario_video_streaming, scenario_gaming, 
                     scenario_navigation, scenario_free, scenario_cold_weather)

# English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class EntropyWeightAnalyzer:
    """Entropy Weight Analyzer - Component Impact Assessment"""
    
    def __init__(self):
        self.battery = SmartphoneBatteryModel()
        
    def collect_data(self):
        """ 
        Collect data from 5 scenarios
        Returns: power data matrix (5 scenarios × 5 components)
        """
        scenarios = {
            'Video Streaming': scenario_video_streaming,
            'Gaming': scenario_gaming,
            'Navigation': scenario_navigation,
            'Idle': scenario_free,
            'Cold Weather': scenario_cold_weather
        }
        
        # Data matrix
        data_matrix = []
        scenario_names = []
        
        print("="*70)
        print("Collecting Data from 5 Scenarios for Entropy Weight Analysis")
        print("="*70)
        
        for name, func in scenarios.items():
            scenario = func(0)
            
            # Calculate component power
            components = self._calculate_power_components(scenario)
            total_power = sum(components.values())
            
            data_matrix.append([
                components['Screen'],
                components['CPU'],
                components['Network'],
                components['GPS'],
                components['Base']
            ])
            
            scenario_names.append(name)
            
            print(f"\n【{name}】")
            print(f"  Screen:  {components['Screen']:.3f} W")
            print(f"  CPU:     {components['CPU']:.3f} W")
            print(f"  Network: {components['Network']:.3f} W")
            print(f"  GPS:     {components['GPS']:.3f} W")
            print(f"  Base:    {components['Base']:.3f} W")
            print(f"  Total:   {total_power:.3f} W")
        
        print("\n" + "="*70)
        
        return np.array(data_matrix), scenario_names
    
    def _calculate_power_components(self, scenario):
        """Calculate component power"""
        components = {
            'Screen': 0.0,
            'CPU': 0.0,
            'Network': 0.0,
            'GPS': 0.0,
            'Base': self.battery.P_base
        }
        
        # Screen power
        if scenario.get('screen_on', False):
            brightness = float(np.clip(scenario.get('brightness', 0.5), 0.0, 1.0))
            components['Screen'] = self.battery.P_a * brightness * self.battery.P_refresh * self.battery.P_screen_square
        
        # CPU power
        if 'cpu_usage' in scenario:
            cpu_usage = float(np.clip(scenario.get('cpu_usage', 0.0), 0.0, 1.0))
            components['CPU'] = self.battery.P_cpu_idle + cpu_usage * self.battery.P_cpu_B * (self.battery.P_cpu_f ** 2)
        
        # Network power
        if 'data_rate' in scenario:
            data_rate = max(0.0, float(scenario.get('data_rate', 0.0)))
            components['Network'] = self.battery.P_net_idle + self.battery.beta * data_rate
        
        # GPS power
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
        
        print("\n2. Information Entropy e_j:")
        component_names = ['Screen', 'CPU', 'Network', 'GPS', 'Base']
        for i, name in enumerate(component_names):
            print(f"   {name}: {entropy[i]:.6f}")
        
        # 步骤3: 计算信息效用值(差异系数)
        # d_j = 1 - e_j,熵越大,差异越小,权重越小
        divergence = 1 - entropy
        
        # Handle negative weights (for features with no variation)
        divergence = np.maximum(divergence, 1e-10)  # Ensure all divergences are positive
        
        print("\n3. Divergence d_j = 1 - e_j:")
        for i, name in enumerate(component_names):
            print(f"   {name}: {divergence[i]:.6f}")
        
        # 步骤4: 计算权重
        # w_j = d_j / sum(d_j)
        weights = divergence / np.sum(divergence)
        
        print("\n4. Entropy Weights w_j:")
        for i, name in enumerate(component_names):
            print(f"   {name}: {weights[i]:.6f} ({weights[i]*100:.2f}%)")
        
        print("\n" + "="*70)
        
        return weights


def plot_donut_chart(weights, component_names):
    """Plot donut chart showing entropy weights"""
    print("\nGenerating Entropy Weight Donut Chart...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for components
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        weights,
        labels=component_names,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='black', linewidth=1.5)
    )
    
    # Customize text
    for text in texts:
        text.set_fontsize(13)
        text.set_fontweight('bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # Center text
    ax.text(0, 0, 'Component\nImpact\nWeights', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='#ECF0F1', 
                     edgecolor='#2C3E50', linewidth=2))
    
    # Title
    #ax.set_title('Entropy Weight Method: Component Impact on Total Energy Consumption\n(Based on 5 Scenarios)',
      #           fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('entropy_weights.png', dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Donut chart saved: entropy_weights.png")
    plt.close()


# Main program
if __name__ == "__main__":
    print("="*70)
    print("ENTROPY WEIGHT METHOD ANALYSIS")
    print("Component Impact on Total Energy Consumption")
    print("5 Scenarios × 5 Components")
    print("="*70)
    
    # Create analyzer
    analyzer = EntropyWeightAnalyzer()
    
    # Collect data (5 scenarios)
    data_matrix, scenario_names = analyzer.collect_data()
    
    # Calculate entropy weights
    component_names = ['Screen', 'CPU', 'Network', 'GPS', 'Base']
    weights = analyzer.calculate_entropy_weights(data_matrix)
    
    # Plot donut chart
    plot_donut_chart(weights, component_names)
   
    # Output conclusion
    print("\n" + "="*70)
    print("ENTROPY WEIGHT ANALYSIS RESULTS")
    print("="*70)
    
    # Find max and min impact components
    max_idx = np.argmax(weights)
    min_idx = np.argmin(weights)
    
    print(f"\n📊 Impact Ranking (Descending):")
    sorted_indices = np.argsort(weights)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. {component_names[idx]}: {weights[idx]:.4f} ({weights[idx]*100:.2f}%)")
    
    print(f"\n🔥 Maximum Impact: {component_names[max_idx]} (Weight: {weights[max_idx]:.4f})")
    print(f"💡 Minimum Impact: {component_names[min_idx]} (Weight: {weights[min_idx]:.4f})")
    
    # Total energy across scenarios
    print(f"\n⚡ Total Energy by Scenario:")
    for i, name in enumerate(scenario_names):
        total = np.sum(data_matrix[i])
        print(f"  {name}: {total:.3f} W")
    
    print("\n" + "="*70)
    print("Analysis Complete! Generated file:")
    print("  entropy_weights.png - Entropy Weight Donut Chart")
    print("="*70)
