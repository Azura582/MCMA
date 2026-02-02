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
sns.set_context("paper", font_scale=1.4)
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
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                ax=ax)
    
    # 设置标题和标签
    #ax.set_title('Screen Power Consumption Ratio Heatmap\n(Baseline: 60Hz, 30% Brightness)', 
     #            fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Refresh Rate (Hz)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Brightness Level (%)', fontsize=13, fontweight='bold', labelpad=10)
    
    # 调整colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    filename1 = 'power_ratio_heatmap.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [1/4] 已生成: {filename1}")
    
    # ====================
    # 图2: 线图 - 不同亮度下刷新率的影响 (Seaborn风格)
    # ====================
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 使用seaborn的色板
    palette = sns.color_palette("rocket_r", n_colors=len(df['Brightness'].unique()))
    
    for i, brightness in enumerate(sorted(df['Brightness'].unique())):
        subset = df[df['Brightness'] == brightness].sort_values('Refresh_Rate')
        ax.plot(subset['Refresh_Rate'], 
                subset['Power_Ratio'],
                marker='o',
                linewidth=2.5,
                markersize=10,
                label=f'{int(brightness*100)}% Brightness',
                color=palette[i],
                alpha=0.85)
    
    # 添加基准线
    ax.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=2.5, 
               label='Baseline (1.0x)', alpha=0.8, zorder=0)
    
    # 设置标题和标签
    ax.set_xlabel('Refresh Rate (Hz)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Power Consumption Ratio', fontsize=13, fontweight='bold', labelpad=10)
    #ax.set_title('Impact of Refresh Rate on Power Consumption\nat Different Brightness Levels\n(Baseline: 60Hz, 30% Brightness)', 
      #           fontsize=16, fontweight='bold', pad=20)
    
    # 设置图例
    ax.legend(loc='upper left', frameon=True, shadow=True, 
              fontsize=11, ncol=2, framealpha=0.95)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置刻度
    ax.set_xticks([60, 70, 80, 90, 100, 110, 120])
    
    plt.tight_layout()
    filename2 = 'power_ratio_by_refresh.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [2/4] 已生成: {filename2}")
    
    # ====================
    # 图3: 分组柱状图 - 不同刷新率下亮度的影响 (Seaborn风格)
    # ====================
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    # 准备数据
    plot_df = df.copy()
    plot_df['Brightness_Label'] = plot_df['Brightness_Percent'].astype(str) + '%'
    
    # 使用seaborn的barplot
    sns.barplot(data=plot_df, 
                x='Refresh_Rate', 
                y='Power_Ratio',
                hue='Brightness_Label',
                palette='mako_r',
                ax=ax,
                edgecolor='black',
                linewidth=1.2,
                alpha=0.85)
    
    # 添加基准线
    ax.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=2.5, 
               label='Baseline (1.0x)', alpha=0.8, zorder=0)
    
    # 设置标题和标签
    ax.set_xlabel('Refresh Rate (Hz)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Power Consumption Ratio', fontsize=13, fontweight='bold', labelpad=10)
    #ax.set_title('Power Consumption Ratio by Refresh Rate and Brightness\n(Baseline: 60Hz, 30% Brightness)', 
     #            fontsize=16, fontweight='bold', pad=20)
    
    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, 
              title='Brightness', 
              loc='upper left', 
              frameon=True, 
              shadow=True,
              fontsize=11,
              title_fontsize=12,
              ncol=3,
              framealpha=0.95)
    
    # 设置网格
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    filename3 = 'power_ratio_bar_chart.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [3/4] 已生成: {filename3}")
    
    # ====================
    # 图4: 雷达图 - 各刷新率平均功耗倍数对比 (新增)
    # ====================
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw=dict(projection='polar'))
    
    # 准备数据
    refresh_groups = df.groupby('Refresh_Rate')['Power_Ratio'].mean().reset_index()
    categories = [f'{int(r)} Hz' for r in refresh_groups['Refresh_Rate']]
    values = refresh_groups['Power_Ratio'].tolist()
    
    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=3, color='#3498db', label='Average Power Ratio')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    
    # 添加基准线
    baseline_values = [1.0] * len(angles)
    ax.plot(angles, baseline_values, '--', linewidth=2.5, color='#e74c3c', 
            label='Baseline (1.0x)', alpha=0.8)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # 设置标题
    ##ax.set_title('Average Power Consumption Ratio\nby Refresh Rate\n(Baseline: 60Hz, 30% Brightness)', 
       #          fontsize=16, fontweight='bold', pad=30, y=1.08)
    
    # 设置图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
              frameon=True, shadow=True, fontsize=11, framealpha=0.95)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    
    # 设置y轴范围
    ax.set_ylim(0, max(values) * 1.1)
    
    plt.tight_layout()
    filename4 = 'power_radar_chart.png'
    plt.savefig(filename4, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [4/4] 已生成: {filename4}")


if __name__ == "__main__":
    df_results = main()
