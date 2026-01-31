import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model import SmartphoneBatteryModel
from scenery import scenario_video_streaming

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


def simulate_full_video_discharge():
    """满电常温刷视频完整放电仿真"""
    model = SmartphoneBatteryModel()
    
    # 初始条件：满电、常温、零极化
    soc0 = 1.0
    T_init = 298.15  # 25°C
    y0 = [soc0, T_init, 0.0, 0.0]
    
    # 足够长的模拟时间
    t_span = (0, 48 * 3600)
    
    print("="*70)
    print("满电常温刷视频场景 - 完整放电分析")
    print("="*70)
    print(f"初始条件：SOC={soc0*100:.0f}%, T={T_init-273.15:.1f}°C")
    print("场景：视频流媒体（亮度70%, CPU使用30%, 网络2Mbps）")
    print(f"电池容量：{model.Q0:.2f} mAh")
    print(f"标称电压：{model.V_nom} V")
    print("="*70)
    
    # 运行仿真
    sol = model.simulate(t_span, y0, scenario_video_streaming, max_step=60)
    
    # 找到放空时间
    t_empty_s = model.find_empty_time(sol)
    t_empty_h = t_empty_s / 3600
    
    print(f"\n✓ 仿真完成")
    print(f"  放电时长：{t_empty_h:.2f} 小时")
    print(f"  仿真步数：{len(sol.t)} 步")
    
    # 截取到放空时刻
    mask = sol.t <= t_empty_s
    
    # 提取数据
    data = {
        't_s': sol.t[mask],
        't_h': sol.t[mask] / 3600,
        'SOC': sol.y[0, mask],
        'T_batt': sol.y[1, mask],
        'U1': sol.y[2, mask],
        'U2': sol.y[3, mask],
        't_empty_h': t_empty_h
    }
    
    # 计算衍生量
    data['V_oc'] = np.array([model.V_oc(soc) for soc in data['SOC']])
    data['R0'] = np.array([model.get_RC_params(data['SOC'][i], data['T_batt'][i])[0] 
                           for i in range(len(data['SOC']))])
    data['R1'] = np.array([model.get_RC_params(data['SOC'][i], data['T_batt'][i])[1] 
                           for i in range(len(data['SOC']))])
    data['R2'] = np.array([model.get_RC_params(data['SOC'][i], data['T_batt'][i])[2] 
                           for i in range(len(data['SOC']))])
    
    # 计算电流和功率
    I_A = []
    P_total = []
    V_terminal = []
    
    for i in range(len(data['t_s'])):
        scenario = scenario_video_streaming(data['t_s'][i])
        P = model.component_power(data['t_s'][i], scenario)
        I = model.solve_current(P, data['V_oc'][i], data['U1'][i], data['U2'][i], data['R0'][i])
        V_term = model.get_terminal_voltage(data['SOC'][i], data['T_batt'][i], 
                                            data['U1'][i], data['U2'][i], I)
        
        P_total.append(P)
        I_A.append(I)
        V_terminal.append(V_term)
    
    data['I_A'] = np.array(I_A)
    data['P_total'] = np.array(P_total)
    data['V_terminal'] = np.array(V_terminal)
    data['T_celsius'] = data['T_batt'] - 273.15
    
    return model, data


def plot_comprehensive_analysis(model, data):
    """绘制8合1综合分析图"""
    
    # 配色方案（现代科研风）
    colors = {
        'primary': '#2E86C1',    # 蓝色
        'secondary': '#E74C3C',  # 红色
        'tertiary': '#27AE60',   # 绿色
        'quaternary': '#8E44AD', # 紫色
        'U1': '#F5B7B1',         # 粉红
        'U2': '#AED6F1',         # 浅蓝
        'grid': '#BDC3C7',       # 浅灰
        'text': '#2C3E50'        # 深灰
    }
    
    # 创建大图（3行3列布局，最后一个跨两列）
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    t_h = data['t_h']
    t_empty = data['t_empty_h']
    
    # ========== 1. SOC下降曲线 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_h, data['SOC']*100, color=colors['primary'], linewidth=0.5)
    ax1.fill_between(t_h, 0, data['SOC']*100, color=colors['primary'], alpha=0.2)
    ax1.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.set_xlabel('时间 (h)', fontweight='bold')
    ax1.set_ylabel('SOC (%)', fontweight='bold')
    ax1.set_title('① 荷电状态 (SOC)', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, t_empty*1.05])
    ax1.set_ylim([0, 105])
    
    # ========== 2. 开路电压OCV ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_h, data['V_oc'], color=colors['tertiary'], linewidth=2.5)
    ax2.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax2.set_xlabel('时间 (h)', fontweight='bold')
    ax2.set_ylabel('电压 (V)', fontweight='bold')
    ax2.set_title('② 开路电压 ($V_{oc}$)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, t_empty*1.05])
    
    # ========== 3. 极化电压U1/U2堆叠 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    total = data['U1'] + data['U2']
    ax3.fill_between(t_h, 0, data['U1'], color=colors['U1'], alpha=0.85, label='$U_1$ (电化学)')
    ax3.fill_between(t_h, data['U1'], total, color=colors['U2'], alpha=0.85, label='$U_2$ (浓度)')
    ax3.plot(t_h, data['U1'], color='#C0392B', linewidth=1.3)
    ax3.plot(t_h, total, color='#2E86C1', linewidth=1.3)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.5)
    ax3.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax3.set_xlabel('时间 (h)', fontweight='bold')
    ax3.set_ylabel('极化电压 (V)', fontweight='bold')
    ax3.set_title('③ 极化电压堆叠', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, t_empty*1.05])
    
    # ========== 4. 内阻 + 温度（双Y轴）==========
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4_temp = ax4.twinx()
    ax4_temp.plot(t_h, data['T_celsius'], color='#E67E22', linewidth=1.5, label='电池温度')
    ax4_temp.axhline(25, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label='环境温度')
    ax4_temp.set_ylabel('温度 (°C)', fontweight='bold', color='#E67E22')
    ax4_temp.tick_params(axis='y', labelcolor='#E67E22')
    # 左轴：内阻
    ax4.plot(t_h, data['R0']*1000, color='#E74C3C', linewidth=1.2, label='$R_0$ (欧姆)')
    ax4.plot(t_h, data['R1']*1000, color='#F39C12', linewidth=1.8, label='$R_1$ (电化学)', linestyle='--')
    ax4.plot(t_h, data['R2']*1000, color='#9B59B6', linewidth=1.8, label='$R_2$ (浓度)', linestyle='--')
    ax4.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax4.set_xlabel('时间 (h)', fontweight='bold')
    ax4.set_ylabel('内阻 (mΩ)', fontweight='bold', color="#110F0F")
    ax4.tick_params(axis='y', labelcolor="#131111")
    ax4.set_title('内阻与温度变化', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim([0, t_empty*1.05])
    
    # 右轴：温度
    
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_temp.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, framealpha=0.9)
    
    # ========== 5. 放电电流 ==========
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_h, data['I_A'], color=colors['secondary'], linewidth=0.5)
    ax5.fill_between(t_h, 0, data['I_A'], color=colors['secondary'], alpha=0.2)
    ax5.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=0.5)
    ax5.set_xlabel('时间 (h)', fontweight='bold')
    ax5.set_ylabel('电流 (A)', fontweight='bold')
    ax5.set_title('⑤ 放电电流', fontsize=13, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_xlim([0, t_empty*1.05])
    
    # ========== 6. 终端电压 ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t_h, data['V_terminal'], color='#16A085', linewidth=2.5, label='终端电压')
    ax6.plot(t_h, data['V_oc'], color='#27AE60', linewidth=1.8, linestyle='--', alpha=0.7, label='开路电压')
    ax6.axhline(2.5, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label='截止电压')
    ax6.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax6.set_xlabel('时间 (h)', fontweight='bold')
    ax6.set_ylabel('电压 (V)', fontweight='bold')
    ax6.set_title('⑥ 终端电压 vs 开路电压', fontsize=13, fontweight='bold', pad=10)
    ax6.legend(loc='best', fontsize=9, framealpha=0.9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_xlim([0, t_empty*1.05])
    
    # ========== 7. 功率消耗 ==========
    ax7 = fig.add_subplot(gs[2, :])  # 跨三列
    ax7.plot(t_h, data['P_total'], color='#8E44AD', linewidth=2.5, label='总功率')
    ax7.fill_between(t_h, 0, data['P_total'], color='#8E44AD', alpha=0.2)
    
    # 计算平均功率
    P_avg = np.mean(data['P_total'])
    ax7.axhline(P_avg, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'平均功率 = {P_avg:.3f} W')
    ax7.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=1.5, 
                label=f'放空时刻 = {t_empty:.2f}h')
    
    ax7.set_xlabel('时间 (h)', fontweight='bold')
    ax7.set_ylabel('功率 (W)', fontweight='bold')
    ax7.set_title('⑦ 功率消耗曲线', fontsize=13, fontweight='bold', pad=10)
    ax7.legend(loc='best', fontsize=10, framealpha=0.9)
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.set_xlim([0, t_empty*1.05])
    
    # 总标题
    fig.suptitle('满电常温刷视频场景 - 完整放电分析 (7合1)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('video_full_analysis_8in1.png', dpi=400, bbox_inches='tight')
    print("\n✓ 7合1综合分析图已保存：video_full_analysis_8in1.png")
    plt.show()


def plot_energy_statistics(model, data):
    """绘制能量统计图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_h = data['t_h']
    t_empty = data['t_empty_h']
    
    # 计算能量相关量
    E_discharged = np.cumsum(data['P_total'] * np.gradient(data['t_s'])) / 3600  # Wh
    E_total = E_discharged[-1]
    
    # 1. 累积放电能量
    ax = axes[0, 0]
    ax.plot(t_h, E_discharged, color='#2E86C1', linewidth=2.5)
    ax.fill_between(t_h, 0, E_discharged, color='#2E86C1', alpha=0.2)
    ax.set_xlabel('时间 (h)', fontweight='bold', fontsize=11)
    ax.set_ylabel('累积能量 (Wh)', fontweight='bold', fontsize=11)
    ax.set_title(f'累积放电能量 (总计: {E_total:.2f} Wh)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 2. SOC vs OCV曲线（特性曲线）
    ax = axes[0, 1]
    ax.plot(data['SOC']*100, data['V_oc'], color='#27AE60', linewidth=2.5, marker='o', 
            markersize=2, markevery=50)
    ax.set_xlabel('SOC (%)', fontweight='bold', fontsize=11)
    ax.set_ylabel('开路电压 (V)', fontweight='bold', fontsize=11)
    ax.set_title('OCV-SOC特性曲线', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 100])
    
    # 3. 内阻 vs SOC
    ax = axes[1, 0]
    ax.plot(data['SOC']*100, data['R0']*1000, color='#E74C3C', linewidth=2.5, label='$R_0$')
    ax.plot(data['SOC']*100, data['R1']*1000, color='#F39C12', linewidth=2, label='$R_1$', linestyle='--')
    ax.plot(data['SOC']*100, data['R2']*1000, color='#9B59B6', linewidth=2, label='$R_2$', linestyle='-.')
    ax.set_xlabel('SOC (%)', fontweight='bold', fontsize=11)
    ax.set_ylabel('内阻 (mΩ)', fontweight='bold', fontsize=11)
    ax.set_title('内阻-SOC特性曲线', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 100])
    
    # 4. 关键参数汇总（文字表格）
    ax = axes[1, 1]
    ax.axis('off')
    
    # 计算统计量
    stats_text = f"""
    【关键参数统计】
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    放电时长：     {t_empty:.2f} 小时
    总放电能量：   {E_total:.2f} Wh
    平均功率：     {np.mean(data['P_total']):.3f} W
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    初始SOC：      {data['SOC'][0]*100:.1f}%
    终止SOC：      {data['SOC'][-1]*100:.1f}%
    SOC变化：      {(data['SOC'][0]-data['SOC'][-1])*100:.1f}%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    初始OCV：      {data['V_oc'][0]:.3f} V
    终止OCV：      {data['V_oc'][-1]:.3f} V
    OCV下降：      {data['V_oc'][0]-data['V_oc'][-1]:.3f} V
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    电流范围：     {data['I_A'].min():.3f} ~ {data['I_A'].max():.3f} A
    平均电流：     {np.mean(data['I_A']):.3f} A
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    温度范围：     {data['T_celsius'].min():.2f} ~ {data['T_celsius'].max():.2f} °C
    温升：         {data['T_celsius'].max()-data['T_celsius'].min():.2f} °C
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    电池容量：     {model.Q0:.2f} mAh
    标称电压：     {model.V_nom} V
    理论能量：     {model.Q0*model.V_nom/1000:.2f} Wh
    能量效率：     {E_total/(model.Q0*model.V_nom/1000)*100:.1f}%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8, edgecolor='#34495E', linewidth=2))
    
    fig.suptitle('能量分析与特性曲线', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('video_energy_analysis.png', dpi=400, bbox_inches='tight')
    print("✓ 能量分析图已保存：video_energy_analysis.png")
    plt.show()


if __name__ == '__main__':
    # 运行仿真
    model, data = simulate_full_video_discharge()
    
    # 绘制8合1综合分析图
    plot_comprehensive_analysis(model, data)
    
    # 绘制能量统计分析图
    plot_energy_statistics(model, data)
    
    print("\n" + "="*70)
    print("所有分析图表生成完成！")
    print("="*70)
    print("生成的文件：")
    print("  1. video_full_analysis_8in1.png   - 8合1综合分析图")
    print("  2. video_energy_analysis.png      - 能量与特性曲线分析")
    print("="*70)
