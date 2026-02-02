"""
屏幕功耗倍数分析
基准场景: 视频流满电, 60Hz, 亮度0.3
分析场景: 刷新率60-120Hz (7组), 亮度0.5-0.9 (5组)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置科研美赛风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.6)  # 增大基础字体比例
sns.set_palette("husl")

# 设置字体为Times New Roman（英文）+ SimHei（中文）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

def calculate_screen_power(brightness, refresh_rate, screen_area=1.2):
    """
    计算屏幕功耗
    P_screen = 0.02 × brightness × refresh_rate × screen_area
    
    参数:
        brightness: 亮度 (0-1)
        refresh_rate: 刷新率 (Hz)
        screen_area: 屏幕面积 (dm²), 默认1.2
    
    返回:
        屏幕功耗 (W)
    """
    return 0.02 * brightness * refresh_rate * screen_area

def main():
    # 基准参数 (60Hz, 亮度0.3)
    baseline_brightness = 0.3
    baseline_refresh = 60
    baseline_area = 1.2  # 视频流场景下的典型屏幕面积
    
    # 计算基准功耗
    baseline_power = calculate_screen_power(baseline_brightness, baseline_refresh, baseline_area)
    
    print("=" * 70)
    print("屏幕功耗倍数分析")
    print("=" * 70)
    print(f"\n【基准场景】")
    print(f"  刷新率: {baseline_refresh} Hz")
    print(f"  亮度: {baseline_brightness * 100:.0f}%")
    print(f"  屏幕面积: {baseline_area} dm²")
    print(f"  基准功耗: {baseline_power:.4f} W")
    print("=" * 70)
    
    # 分析参数
    refresh_rates = [60, 70, 80, 90, 100, 110, 120]
    brightness_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 存储结果
    results = []
    
    print(f"\n【功耗倍数计算】")
    print(f"总计算组数: {len(refresh_rates)} × {len(brightness_levels)} = {len(refresh_rates) * len(brightness_levels)} 组\n")
    
    # 计算所有组合
    for refresh in refresh_rates:
        for brightness in brightness_levels:
            # 计算当前功耗
            current_power = calculate_screen_power(brightness, refresh, baseline_area)
            
            # 计算功耗倍数
            power_ratio = current_power / baseline_power
            
            results.append({
                'Refresh_Rate': refresh,
                'Brightness': brightness,
                'Brightness_Percent': int(brightness * 100),
                'Screen_Power': current_power,
                'Power_Ratio': power_ratio
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存到CSV
    csv_filename = 'power_ratio_results.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"✓ 数据已保存到: {csv_filename}")
    
    # 打印统计信息
    print(f"\n【统计摘要】")
    print(f"功耗倍数范围: {df['Power_Ratio'].min():.2f}x - {df['Power_Ratio'].max():.2f}x")
    print(f"平均功耗倍数: {df['Power_Ratio'].mean():.2f}x")
    print(f"中位数功耗倍数: {df['Power_Ratio'].median():.2f}x")
    
    # 按刷新率分组统计
    print(f"\n【按刷新率分组】(平均功耗倍数)")
    for refresh in refresh_rates:
        avg_ratio = df[df['Refresh_Rate'] == refresh]['Power_Ratio'].mean()
        min_ratio = df[df['Refresh_Rate'] == refresh]['Power_Ratio'].min()
        max_ratio = df[df['Refresh_Rate'] == refresh]['Power_Ratio'].max()
        print(f"  {refresh:3d} Hz: {avg_ratio:.2f}x (范围: {min_ratio:.2f}x - {max_ratio:.2f}x)")
    
    # 按亮度分组统计
    print(f"\n【按亮度分组】(平均功耗倍数)")
    for brightness in brightness_levels:
        avg_ratio = df[df['Brightness'] == brightness]['Power_Ratio'].mean()
        min_ratio = df[df['Brightness'] == brightness]['Power_Ratio'].min()
        max_ratio = df[df['Brightness'] == brightness]['Power_Ratio'].max()
        print(f"  {int(brightness*100):2d}%: {avg_ratio:.2f}x (范围: {min_ratio:.2f}x - {max_ratio:.2f}x)")
    
    # 找出极值组合
    print(f"\n【极值组合】")
    min_idx = df['Power_Ratio'].idxmin()
    max_idx = df['Power_Ratio'].idxmax()
    
    print(f"最低功耗倍数: {df.loc[min_idx, 'Power_Ratio']:.2f}x")
    print(f"  → 刷新率: {df.loc[min_idx, 'Refresh_Rate']:.0f} Hz, 亮度: {df.loc[min_idx, 'Brightness_Percent']}%")
    print(f"  → 功耗: {df.loc[min_idx, 'Screen_Power']:.4f} W")
    
    print(f"\n最高功耗倍数: {df.loc[max_idx, 'Power_Ratio']:.2f}x")
    print(f"  → 刷新率: {df.loc[max_idx, 'Refresh_Rate']:.0f} Hz, 亮度: {df.loc[max_idx, 'Brightness_Percent']}%")
    print(f"  → 功耗: {df.loc[max_idx, 'Screen_Power']:.4f} W")
    
    # 创建可视化
    print(f"\n【生成可视化图表】")
    create_visualizations(df, baseline_power)
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    
    return df

def create_visualizations(df, baseline_power):
    """创建科研美赛风格的可视化图表"""
    
    # ====================
    # 图1: 热力图 - 功耗倍数 (Seaborn专业风格)
    # ====================
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 创建透视表
    pivot_table = df.pivot(index='Brightness_Percent', 
                           columns='Refresh_Rate', 
                           values='Power_Ratio')
    
    # 使用seaborn绘制热力图
    sns.heatmap(pivot_table, 
                annot=True,  # 显示数值
                fmt='.2f',   # 保留2位小数
                cmap='RdYlGn_r',  # 红黄绿渐变（反向）
                cbar_kws={
                    'label': 'Power Ratio (relative to baseline)',
                    'shrink': 0.8
                },
                linewidths=1.5,
                linecolor='white',
                square=False,
                vmin=1.5,
                vmax=6.5,
                annot_kws={'fontsize': 13, 'fontweight': 'bold'},  # 增大热力图数字到13
                ax=ax)
    
    # 设置标题和标签
    #ax.set_title('Screen Power Consumption Ratio Heatmap\n(Baseline: 60Hz, 30% Brightness)', 
     #            fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Refresh Rate (Hz)', fontsize=15, fontweight='bold', labelpad=10)  # 增大到15
    ax.set_ylabel('Brightness Level (%)', fontsize=15, fontweight='bold', labelpad=10)  # 增大到15
    
    # 调整colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)  # 增大colorbar刻度到13
    cbar.set_label('Power Ratio (relative to baseline)', fontsize=14, fontweight='bold')  # 增大colorbar标签到14
    
    # 调整刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=13)  # 增大轴刻度到13
    
    plt.tight_layout()
    filename1 = 'power_ratio_heatmap.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [1/4] 已生成: {filename1}")
    
   
    


if __name__ == "__main__":
    df_results = main()
