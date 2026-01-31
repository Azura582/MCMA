import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]  # 父目录的上级是 MCMA
sys.path.insert(0, str(project_root))
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
    
    # 运行仿真（启用密集输出以便插值）
    sol = model.simulate(t_span, y0, scenario_video_streaming, max_step=60)
    
    # 找到放空时间
    t_empty_s = model.find_empty_time(sol)
    t_empty_h = t_empty_s / 3600
    
    print(f"\n✓ 仿真完成")
    print(f"  放电时长：{t_empty_h:.2f} 小时")
    print(f"  原始仿真步数：{len(sol.t)} 步")
    
    # ========== 插值增加绘图点数 ==========
    from scipy.interpolate import interp1d
    
    # 原始数据（截取到放空时刻）
    mask = sol.t <= t_empty_s
    t_original = sol.t[mask]
    
    # 创建密集时间网格（增加到5000个点，可根据需要调整）
    n_dense_points = 5000
    t_dense = np.linspace(0, t_empty_s, n_dense_points)
    
    # 对各个状态变量进行插值
    f_soc = interp1d(t_original, sol.y[0, mask], kind='cubic', fill_value='extrapolate')
    f_T = interp1d(t_original, sol.y[1, mask], kind='cubic', fill_value='extrapolate')
    f_U1 = interp1d(t_original, sol.y[2, mask], kind='cubic', fill_value='extrapolate')
    f_U2 = interp1d(t_original, sol.y[3, mask], kind='cubic', fill_value='extrapolate')
    
    soc_dense = f_soc(t_dense)
    T_dense = f_T(t_dense)
    U1_dense = f_U1(t_dense)
    U2_dense = f_U2(t_dense)
    
    print(f"  插值后绘图点数：{n_dense_points} 步（增加 {n_dense_points/len(t_original):.1f}x）")
    
    # 提取数据（使用插值后的密集数据）
    data = {
        't_s': t_dense,
        't_h': t_dense / 3600,
        'SOC': soc_dense,
        'T_batt': T_dense,
        'U1': U1_dense,
        'U2': U2_dense,
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

def plot_individual_charts(model, data):
    """生成5张独立图表"""
    
    t_h = data['t_h']
    t_empty = data['t_empty_h']
    
    # ========== 图1: SOC荷电状态 ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t_h, data['SOC']*100, color='#2E86C1', linewidth=2.5)
    ax1.fill_between(t_h, 0, data['SOC']*100, color='#2E86C1', alpha=0.2)
    ax1.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax1.set_xlabel('时间 (h)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('SOC (%)', fontweight='bold', fontsize=13)
    ax1.set_title('① 荷电状态 (SOC)', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, t_empty*1.05])
    ax1.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig('1_SOC曲线.png', dpi=400, bbox_inches='tight')
    print("✓ 图1已保存: 1_SOC曲线.png")
    plt.close()
    
    # ========== 图2: 极化电压堆叠 ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    total = data['U1'] + data['U2']
    ax2.fill_between(t_h, 0, data['U1'], color='#F5B7B1', alpha=0.85, label='$U_1$ (电化学)')
    ax2.fill_between(t_h, data['U1'], total, color='#AED6F1', alpha=0.85, label='$U_2$ (浓度)')
    ax2.plot(t_h, data['U1'], color="#100E0E", linewidth=0.5)
    ax2.plot(t_h, total, color="#131415", linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax2.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax2.set_xlabel('时间 (h)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('极化电压 (V)', fontweight='bold', fontsize=13)
    ax2.set_title('③ 极化电压堆叠', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, t_empty*1.05])
    plt.tight_layout()
    plt.savefig('2_极化电压.png', dpi=400, bbox_inches='tight')
    print("✓ 图2已保存: 2_极化电压.png")
    plt.close()
    
    # ========== 图3: 内阻+温度双轴 ==========
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # 左轴：内阻
    ax3.plot(t_h, data['R0']*1000, color='#E74C3C', linewidth=2.5, label='$R_0$ (欧姆)')
    ax3.plot(t_h, data['R1']*1000, color='#F39C12', linewidth=2.5, label='$R_1$ (电化学)', linestyle='--')
    ax3.plot(t_h, data['R2']*1000, color='#9B59B6', linewidth=2.5, label='$R_2$ (浓度)', linestyle='--')
    ax3.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax3.set_xlabel('时间 (h)', fontweight='bold', fontsize=13)
    ax3.set_ylabel('内阻 (mΩ)', fontweight='bold', fontsize=13, color="#110F0F")
    ax3.tick_params(axis='y', labelcolor="#131111")
    ax3.set_title('④ 内阻与温度变化', fontsize=15, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, t_empty*1.05])
    
    # 右轴：温度
    ax3_temp = ax3.twinx()
    ax3_temp.plot(t_h, data['T_celsius'], color="#3B857C", linewidth=2.5, label='电池温度')
    ax3_temp.axhline(25, color='green', linestyle=':', linewidth=2, alpha=0.6, label='环境温度')
    ax3_temp.set_ylabel('温度 (°C)', fontweight='bold', fontsize=13, color="#833E85")
    ax3_temp.tick_params(axis='y', labelcolor="#852F9D")
    
    # 合并图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_temp.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('3_内阻与温度.png', dpi=400, bbox_inches='tight')
    print("✓ 图3已保存: 3_内阻与温度.png")
    plt.close()
    
    # ========== 图4: 放电电流 ==========
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(t_h, data['I_A'], color="#1B1F22", linewidth=1.5)
    ax4.fill_between(t_h, 0, data['I_A'], color='#E74C3C', alpha=0.2)
    ax4.axvline(t_empty, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax4.set_xlabel('时间 (h)', fontweight='bold', fontsize=13)
    ax4.set_ylabel('电流 (A)', fontweight='bold', fontsize=13)
    ax4.set_title('⑤ 放电电流', fontsize=15, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim([0, t_empty*1.05])
    plt.tight_layout()
    plt.savefig('4_放电电流.png', dpi=400, bbox_inches='tight')
    print("✓ 图4已保存: 4_放电电流.png")
    plt.close()
    
    # ========== 图5: 参数汇总表格 ==========
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    ax5.axis('off')
    
    # 计算能量
    E_discharged = np.cumsum(data['P_total'] * np.gradient(data['t_s'])) / 3600
    E_total = E_discharged[-1]
    
    # 参数汇总文本
    stats_text = f"""
    【关键参数统计汇总】
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⏱  放电时长：         {t_empty:.2f} 小时
    ⚡ 总放电能量：       {E_total:.2f} Wh
    📊 平均功率：         {np.mean(data['P_total']):.3f} W
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🔋 初始SOC：          {data['SOC'][0]*100:.1f}%
    🔋 终止SOC：          {data['SOC'][-1]*100:.1f}%
    📉 SOC变化：          {(data['SOC'][0]-data['SOC'][-1])*100:.1f}%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🔌 初始OCV：          {data['V_oc'][0]:.3f} V
    🔌 终止OCV：          {data['V_oc'][-1]:.3f} V
    📉 OCV下降：          {data['V_oc'][0]-data['V_oc'][-1]:.3f} V
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚡ 电流范围：         {data['I_A'].min():.3f} ~ {data['I_A'].max():.3f} A
    ⚡ 平均电流：         {np.mean(data['I_A']):.3f} A
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🌡  温度范围：         {data['T_celsius'].min():.2f} ~ {data['T_celsius'].max():.2f} °C
    🌡  温升：             {data['T_celsius'].max()-data['T_celsius'].min():.2f} °C
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🔋 电池容量：         {model.Q0:.2f} mAh
    🔌 标称电压：         {model.V_nom} V
    📦 理论能量：         {model.Q0*model.V_nom/1000:.2f} Wh
    ✅ 能量效率：         {E_total/(model.Q0*model.V_nom/1000)*100:.1f}%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    ax5.text(0.5, 0.5, stats_text, transform=ax5.transAxes, 
            fontsize=13, verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=1.5', facecolor='#ECF0F1', 
                     alpha=0.95, edgecolor='#34495E', linewidth=3))
    
    fig5.suptitle('⑤ 关键参数统计汇总', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('5_参数汇总.png', dpi=400, bbox_inches='tight')
    print("✓ 图5已保存: 5_参数汇总.png")
    plt.close()


if __name__ == '__main__':
    # 运行仿真
    model, data = simulate_full_video_discharge()
    
    # 生成5张独立图表
    plot_individual_charts(model, data)
    
    print("\n" + "="*70)
    print("所有独立图表生成完成！")
    print("="*70)
    print("生成的文件：")
    print("  1. 1_SOC曲线.png        - SOC荷电状态")
    print("  2. 2_极化电压.png       - 极化电压堆叠")
    print("  3. 3_内阻与温度.png     - 内阻与温度双轴")
    print("  4. 4_放电电流.png       - 放电电流曲线")
    print("  5. 5_参数汇总.png       - 关键参数统计")
    print("="*70)
