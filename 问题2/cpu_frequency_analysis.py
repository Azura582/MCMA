"""
CPU频率功耗倍数分析
基准场景: 视频流满电, CPU频率2.0 GHz
分析场景: CPU频率1.0-3.5 GHz (以0.5为单位)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SmartphoneBatteryModel
from scenery import scenario_video_streaming

# 设置科研美赛风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
sns.set_palette("husl")

# 设置字体为Times New Roman（英文）+ SimHei（中文）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

def calculate_cpu_power(cpu_freq, cpu_usage=0.3, P_cpu_idle=0.1, P_cpu_B=0.3):
    """
    计算CPU功耗
    P_cpu = P_cpu_idle + cpu_usage × P_cpu_B × (cpu_freq)²
    
    参数:
        cpu_freq: CPU频率 (GHz)
        cpu_usage: CPU使用率 (0-1)
        P_cpu_idle: CPU空闲功耗 (W)
        P_cpu_B: CPU系数
    
    返回:
        CPU功耗 (W)
    """
    return P_cpu_idle + cpu_usage * P_cpu_B * (cpu_freq ** 2)

def simulate_discharge_time(model, cpu_freq):
    """
    模拟特定CPU频率下的放电时间
    
    参数:
        model: 电池模型实例
        cpu_freq: CPU频率 (GHz)
    
    返回:
        放电时间 (小时)
    """
    # 修改模型的CPU频率
    model.P_cpu_f = cpu_freq
    
    # 初始条件 (满电)
    y0 = [1.0, 298.15, 0.0, 0.0]  # SOC=1.0 (满电), T=298.15K, U1=0, U2=0
    
    # 时间跨度 (足够长以确保放电完成)
    t_span = (0, 100000)  # 0到100000秒
    
    # 使用视频流场景
    scenario_func = scenario_video_streaming
    
    # 模拟放电
    sol = model.simulate(t_span, y0, scenario_func, max_step=10.0)
    
    # 获取放电时间
    discharge_time_s = model.find_empty_time(sol)
    discharge_time_h = discharge_time_s / 3600
    
    return discharge_time_h

def main():
    # 基准参数 (CPU频率2.0 GHz, 视频流场景)
    baseline_cpu_freq = 2.0
    baseline_cpu_usage = 0.3
    
    # 计算基准CPU功耗
    baseline_cpu_power = calculate_cpu_power(baseline_cpu_freq, baseline_cpu_usage)
    
    print("=" * 70)
    print("CPU频率功耗倍数分析")
    print("=" * 70)
    print(f"\n【基准场景】")
    print(f"  场景: 视频流满电")
    print(f"  CPU频率: {baseline_cpu_freq} GHz")
    print(f"  CPU使用率: {baseline_cpu_usage * 100:.0f}%")
    print(f"  基准CPU功耗: {baseline_cpu_power:.4f} W")
    print("=" * 70)
    
    # 分析参数 - CPU频率范围
    cpu_frequencies = np.arange(1.0, 4.0, 0.5)  # 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 GHz
    
    # 存储结果
    results = []
    
    print(f"\n【功耗倍数计算】")
    print(f"CPU频率范围: {cpu_frequencies[0]:.1f} - {cpu_frequencies[-1]:.1f} GHz")
    print(f"总计算组数: {len(cpu_frequencies)} 组\n")
    
    # 创建电池模型实例
    model = SmartphoneBatteryModel()
    
    print("开始仿真放电时间...\n")
    
    # 计算所有频率
    for i, cpu_freq in enumerate(cpu_frequencies, 1):
        # 计算当前CPU功耗
        current_cpu_power = calculate_cpu_power(cpu_freq, baseline_cpu_usage)
        
        # 计算功耗倍数
        power_ratio = current_cpu_power / baseline_cpu_power
        
        # 模拟放电时间
        discharge_time = simulate_discharge_time(model, cpu_freq)
        
        results.append({
            'CPU_Frequency': cpu_freq,
            'CPU_Power': current_cpu_power,
            'Power_Ratio': power_ratio,
            'Discharge_Time': discharge_time
        })
        
        print(f"  [{i}/{len(cpu_frequencies)}] CPU频率: {cpu_freq:.1f} GHz | "
              f"功耗: {current_cpu_power:.4f} W | 倍数: {power_ratio:.2f}x | "
              f"放电时间: {discharge_time:.2f}h")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存到CSV
    csv_filename = 'cpu_frequency_results.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ 数据已保存到: {csv_filename}")
    
    # 打印统计信息
    print(f"\n【统计摘要】")
    print(f"CPU功耗范围: {df['CPU_Power'].min():.4f} W - {df['CPU_Power'].max():.4f} W")
    print(f"功耗倍数范围: {df['Power_Ratio'].min():.2f}x - {df['Power_Ratio'].max():.2f}x")
    print(f"放电时间范围: {df['Discharge_Time'].min():.2f}h - {df['Discharge_Time'].max():.2f}h")
    
    # 找出极值
    print(f"\n【极值场景】")
    min_idx = df['Power_Ratio'].idxmin()
    max_idx = df['Power_Ratio'].idxmax()
    
    print(f"最低功耗倍数: {df.loc[min_idx, 'Power_Ratio']:.2f}x")
    print(f"  → CPU频率: {df.loc[min_idx, 'CPU_Frequency']:.1f} GHz")
    print(f"  → CPU功耗: {df.loc[min_idx, 'CPU_Power']:.4f} W")
    print(f"  → 放电时间: {df.loc[min_idx, 'Discharge_Time']:.2f}h")
    
    print(f"\n最高功耗倍数: {df.loc[max_idx, 'Power_Ratio']:.2f}x")
    print(f"  → CPU频率: {df.loc[max_idx, 'CPU_Frequency']:.1f} GHz")
    print(f"  → CPU功耗: {df.loc[max_idx, 'CPU_Power']:.4f} W")
    print(f"  → 放电时间: {df.loc[max_idx, 'Discharge_Time']:.2f}h")
    
    # CPU频率与放电时间的关系
    print(f"\n【CPU频率影响分析】")
    for _, row in df.iterrows():
        print(f"  {row['CPU_Frequency']:.1f} GHz: 放电时间 {row['Discharge_Time']:.2f}h, "
              f"功耗倍数 {row['Power_Ratio']:.2f}x")
    
    # 创建可视化
    print(f"\n【生成可视化图表】")
    create_visualizations(df, baseline_cpu_power, baseline_cpu_freq)
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    
    return df

def create_visualizations(df, baseline_power, baseline_freq):
    """创建科研美赛风格的可视化图表"""
    
    # 获取基准点数据
    baseline_row = df[df['CPU_Frequency'] == baseline_freq].iloc[0]
    baseline_power_ratio = baseline_row['Power_Ratio']
    baseline_discharge_time = baseline_row['Discharge_Time']
    
    # ====================
    # 双Y轴图 - CPU频率 vs 功耗倍数 & 放电时间
    # ====================
    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=300)
    
    # 设置Seaborn风格
    sns.set_style("whitegrid")
    
    # 使用深色调专业配色方案（科研美赛标准）
    color1 = '#d62728'  # 深红色 - 功耗（深色暖色调，代表警示/能耗）
    color2 = '#1f77b4'  # 深蓝色 - 续航（深色冷色调，代表稳定/持久）
    baseline_color = '#7f7f7f'  # 深灰色 - 基准点（中性专业色）
    baseline_line_color = '#404040'  # 更深的灰色 - 基准线
    
    # 第一个Y轴 - 功耗倍数 (深红色系)
    ax1.set_xlabel('CPU Frequency (GHz)', fontsize=14, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Power Consumption Ratio', fontsize=14, fontweight='bold', 
                   labelpad=12, color=color1)
    
    # 绘制功耗倍数曲线
    line1 = ax1.plot(df['CPU_Frequency'], df['Power_Ratio'], 
                     marker='o', linewidth=4, markersize=15,
                     color=color1, alpha=0.9, label='Power Ratio',
                     markeredgecolor='black', markeredgewidth=2)
    
    # 功耗倍数基准线 (水平线)
    ax1.axhline(y=baseline_power_ratio, color=baseline_line_color, linestyle='--', 
                linewidth=2.8, alpha=0.7, label=f'Baseline Power (1.0x)')
    
    # 功耗倍数基准点 (垂直线)
    ax1.axvline(x=baseline_freq, color=baseline_line_color, linestyle=':', 
                linewidth=2.8, alpha=0.7, label=f'Baseline Frequency ({baseline_freq} GHz)')
    
    # 添加功耗倍数数值标签
    for _, row in df.iterrows():
        y_offset = 0.12 if row['CPU_Frequency'] != baseline_freq else 0.18
        ax1.text(row['CPU_Frequency'], row['Power_Ratio'] + y_offset, 
                f"{row['Power_Ratio']:.2f}x",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='white', 
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor=color1, alpha=0.9, 
                         edgecolor='black', linewidth=1.8))
    
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # 第二个Y轴 - 放电时间 (绿色系)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Discharge Time (hours)', fontsize=14, fontweight='bold', 
                   labelpad=12, color=color2)
    
    # 绘制放电时间曲线
    line2 = ax2.plot(df['CPU_Frequency'], df['Discharge_Time'], 
                     marker='s', linewidth=4, markersize=15,
                     color=color2, alpha=0.9, label='Discharge Time',
                     markeredgecolor='black', markeredgewidth=2)
    
    # 放电时间基准线 (水平线)
    ax2.axhline(y=baseline_discharge_time, color=baseline_line_color, linestyle='--', 
                linewidth=2.8, alpha=0.7, label=f'Baseline Time ({baseline_discharge_time:.2f}h)')
    
    # 添加放电时间数值标签
    for _, row in df.iterrows():
        y_offset = 0.18 if row['CPU_Frequency'] != baseline_freq else 0.28
        ax2.text(row['CPU_Frequency'], row['Discharge_Time'] + y_offset, 
                f"{row['Discharge_Time']:.2f}h",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='white', 
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor=color2, alpha=0.9, 
                         edgecolor='black', linewidth=1.8))
    
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
    
    # 标注基准交叉点 - 使用Y1轴的坐标，确保在正确位置
    ax1.scatter([baseline_freq], [baseline_power_ratio], 
               s=600, color='gold', marker='*', 
               edgecolor='black', linewidth=3, zorder=10,
               label='Baseline Point')
    
    # 组合图例 - 移到右上角避免遮挡
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # 合并图例，按逻辑顺序排列
    all_lines = [lines1[2], lines1[0], lines2[0], lines1[1], lines2[1], lines1[3]]
    all_labels = ['Baseline Point', 'Power Ratio', 'Discharge Time', 
                  'Baseline Power (1.0x)', f'Baseline Time ({baseline_discharge_time:.2f}h)', 
                  f'Baseline Frequency ({baseline_freq} GHz)']
    
    ax1.legend(all_lines, all_labels, 
              loc='upper right',  # 改到右上角
              frameon=True, 
              shadow=True, 
              fontsize=10.5,
              framealpha=0.98,
              edgecolor='black',
              fancybox=True,
              ncol=2,
              columnspacing=1.2,
              labelspacing=0.6)
    
    # 设置标题
    #ax1.set_title('CPU Frequency Impact on Power Consumption and Battery Life\n' + 
    #             f'Baseline: {baseline_freq} GHz, {baseline_power_ratio:.2f}x Power, {baseline_discharge_time:.2f}h Battery Life',
     #            fontsize=16, fontweight='bold', pad=25)
    
    # 设置X轴刻度
    ax1.set_xticks(df['CPU_Frequency'])
    ax1.set_xticklabels([f'{freq:.1f}' for freq in df['CPU_Frequency']])
    
    # 添加背景色块标识基准区域 - 使用更柔和的颜色
    ax1.axvspan(baseline_freq - 0.15, baseline_freq + 0.15, 
                alpha=0.12, color='gray', zorder=0)
    
    plt.tight_layout()
    filename = 'cpu_frequency_dual_axis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已生成: {filename}")


if __name__ == "__main__":
    df_results = main()
