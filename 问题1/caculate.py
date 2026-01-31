import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]  # 父目录的上级是 MCMA
sys.path.insert(0, str(project_root))

from  model import  SmartphoneBatteryModel
from scenery import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def simulate_discharge_times():
    """模拟不同SOC初始量下的放电时间"""
    # 初始化模型
    battery = SmartphoneBatteryModel()
    
    # 模拟参数
    t_span = (0, 24*3600)  # 24小时
    T_init = 298.15  # 初始温度25°C
    
    # 不同场景
    scenarios = {
        'VideoStream': scenario_video_streaming,
        'game': scenario_gaming,
        'navigation': scenario_navigation,
        'VideoStream Low-Temp': scenario_cold_weather,
        'Idle State':scenario_free
    }
    
    # 不同的初始SOC (20%, 40%, 60%, 80%, 100%)
    soc_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # 存储结果
    results_matrix = {}  # {场景名: [不同SOC下的放电时间]}
    
    print("="*70)
    print("多场景、多初始SOC放电时间预测")
    print("="*70)
    
    # 二重循环：外层SOC，内层场景
    for soc in soc_values:
        print(f"\n{'='*70}")
        print(f"初始SOC: {soc*100:.0f}%")
        print(f"{'='*70}")
        
        for name, scenario_func in scenarios.items():
            print(f"  模拟场景: {name}...", end=' ')
            
            # 初始状态: [SOC, T_batt, U1, U2]
            y0 = [soc, T_init, 0.0, 0.0]
            
            # 运行仿真
            sol = battery.simulate(t_span, y0, scenario_func)
            
            # 计算放电时间
            empty_time_hours = battery.find_empty_time(sol) / 3600
            
            # 存储结果
            if name not in results_matrix:
                results_matrix[name] = []
            results_matrix[name].append(empty_time_hours)
            
            print(f"放空时间: {empty_time_hours:.2f} 小时")
    
    return soc_values, results_matrix

def create_results_table(soc_values, results_matrix):
    """创建结果表格"""
    print("\n生成结果表格...")
    
    # 创建DataFrame
    data = {}
    data['初始SOC (%)'] = [f"{soc*100:.0f}%" for soc in soc_values]
    
    for scenario, times in results_matrix.items():
        data[scenario] = [f"{t:.2f}h" for t in times]
    
    df = pd.DataFrame(data)
    
    # 打印表格
    print("\n" + "="*70)
    print("多场景放电时间预测结果汇总")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # 保存到CSV
    df.to_csv('discharge_time_prediction.csv', index=False, encoding='utf-8-sig')
    print("\n结果已保存到 discharge_time_prediction.csv")
    
    return df


def plot_bar_chart(soc_values, results_matrix):
    """绘制分组柱状图"""
    print("\n生成分组柱状图...")
    
    scenarios = list(results_matrix.keys())
    colors = ["#E6E6EA", "#EBDBD7", "#E8ABB0", "#C14944", "#F1AD48"]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(soc_values))
    width = 0.15
    
    for idx, scenario in enumerate(scenarios):
        times = results_matrix[scenario]
        offset = (idx - len(scenarios)/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=scenario, 
                     color=colors[idx], alpha=0.8, edgecolor='white', linewidth=1.5)
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('初始SOC (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('放电时间 (小时)', fontsize=13, fontweight='bold')
    ax.set_title('不同场景放电时间对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s*100:.0f}%' for s in soc_values], fontsize=11)
    ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=True, 
             facecolor='white', edgecolor='#34495E')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig('discharge_time_bar_chart.png', dpi=300, bbox_inches='tight')
    print("分组柱状图已保存为 discharge_time_bar_chart.png")
    plt.show()


def plot_heatmap(soc_values, results_matrix):
    """绘制热力图"""
    print("\n生成热力图...")
    
    scenarios = list(results_matrix.keys())
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_matrix = np.array([results_matrix[s] for s in scenarios])
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(soc_values)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels([f'{s*100:.0f}%' for s in soc_values], fontsize=12)
    ax.set_yticklabels(scenarios, fontsize=12)
    ax.set_xlabel('Initial SOC(%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Use cases', fontsize=13, fontweight='bold')
    ax.set_title('Discharge Time Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # 添加数值标注
    for i in range(len(scenarios)):
        for j in range(len(soc_values)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", 
                          fontsize=11, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Discharge Time (h)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig('discharge_time_heatmap.png', dpi=300, bbox_inches='tight')
    print("热力图已保存为 discharge_time_heatmap.png")
    plt.show()


# 主程序
if __name__ == "__main__":
    # 运行仿真
    soc_values, results_matrix = simulate_discharge_times()
    
    # 创建结果表格
    df = create_results_table(soc_values, results_matrix)
    
    # 绘制分组柱状图
    plot_bar_chart(soc_values, results_matrix)
    
    # 绘制热力图
    plot_heatmap(soc_values, results_matrix)
    
    print("\n" + "="*70)
    print("所有分析完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  1. discharge_time_bar_chart.png - 分组柱状图")
    print("  2. discharge_time_heatmap.png - 热力图")
    print("  3. discharge_time_prediction.csv - 详细数据表")
    print("="*70)
    
    