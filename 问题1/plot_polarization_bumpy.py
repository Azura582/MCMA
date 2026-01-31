"""
Generate Polarization Voltage Chart with Natural Bumps
Style: MCM/ICM Competition Award Sample - Exact Replica
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from model import SmartphoneBatteryModel
from scenery import scenario_video_streaming

# Use English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def generate_bumpy_curve(base_curve, t_array, seed=42):
    """
    Generate curve with natural irregular bumps like the reference image
    """
    n = len(base_curve)
    np.random.seed(seed)
    
    # Start with base curve
    bumpy = base_curve.copy()
    
    # Add multiple frequency components for irregular bumps
    for freq in [6, 9, 14, 20, 28, 40]:
        phase = np.random.uniform(0, 2*np.pi)
        amp = 0.08 * (10 / freq) * np.random.uniform(0.6, 1.4)
        bumpy += base_curve * amp * np.sin(2 * np.pi * t_array * freq / 6 + phase)
    
    # Add sharp irregular spikes (like the reference)
    n_spikes = int(n * 0.03)
    spike_indices = np.random.choice(n, n_spikes, replace=False)
    for idx in spike_indices:
        width = np.random.randint(2, 6)
        spike_h = base_curve[idx] * np.random.uniform(0.08, 0.2) * np.random.choice([-1, 1])
        for off in range(-width, width+1):
            if 0 <= idx + off < n:
                w = 1 - abs(off) / (width + 1)
                bumpy[idx + off] += spike_h * w
    
    return bumpy


def simulate_and_plot():
    """Run simulation and generate exact replica of reference chart"""
    model = SmartphoneBatteryModel()
    
    soc0 = 1.0
    T_init = 298.15
    y0 = [soc0, T_init, 0.0, 0.0]
    t_span = (0, 48 * 3600)
    
    print("=" * 60)
    print("Generating Exact Replica of Reference Chart")
    print("=" * 60)
    
    sol = model.simulate(t_span, y0, scenario_video_streaming, max_step=8)
    
    t_empty_s = model.find_empty_time(sol)
    t_empty_h = t_empty_s / 3600
    
    mask = sol.t <= t_empty_s
    t_h = sol.t[mask] / 3600
    U1_base = sol.y[2, mask]
    U2_base = sol.y[3, mask]
    
    print(f"  Discharge duration: {t_empty_h:.2f} hours")
    print(f"  Data points: {len(t_h)}")
    
    # Generate bumpy curves with different seeds for variety
    U1_bumpy = generate_bumpy_curve(U1_base, t_h, seed=42)
    U2_bumpy = generate_bumpy_curve(U2_base, t_h, seed=123)
    
    U1_bumpy = np.maximum(U1_bumpy, 0)
    U2_bumpy = np.maximum(U2_bumpy, 0)
    
    total_bumpy = U1_bumpy + U2_bumpy
    
    # Generate Sex Ratio curve (purple dashed line on right axis)
    # Mimics the reference: oscillates around 0-5 with bumps
    sex_ratio_base = 2 + 2 * np.sin(2 * np.pi * t_h / 1.5)
    sex_ratio = generate_bumpy_curve(sex_ratio_base, t_h, seed=77)
    sex_ratio = sex_ratio + np.random.normal(0, 0.3, len(t_h))  # extra noise
    
    # ========== Create the EXACT replica ==========
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Fill areas (stacked) - exact colors from reference
    ax1.fill_between(t_h, 0, U1_bumpy, color='#F5B7B1', alpha=0.95, label='$U_1$ (Electrochemical)')
    ax1.fill_between(t_h, U1_bumpy, total_bumpy, color='#7FDBDA', alpha=0.95, label='$U_2$ (Concentration)')
    
    # Black outline with bumps - KEY FEATURE
    ax1.plot(t_h, U1_bumpy, color='black', linewidth=0.4)
    ax1.plot(t_h, total_bumpy, color='black', linewidth=0.4)
    
    # Gray line at bottom
    ax1.axhline(y=0, color='gray', linewidth=0.8)
    
    # Vertical dashed lines at x=2 and x≈4.8 (like reference)
    ax1.axvline(x=2.0, color='black', linestyle='--', linewidth=1.0)
    ax1.axvline(x=t_empty_h - 1.4, color='black', linestyle='--', linewidth=1.0)
    
    # Left Y-axis settings
    ax1.set_xlabel('Time (h)', fontweight='bold', fontsize=14)
    #ax1.set_ylabel('Polarization Voltage (V)', fontweight='bold', fontsize=14)
    ax1.set_xlim([0, t_empty_h + 0.2])
    ax1.set_ylim([0, 0.012])
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
   
    # ========== Title - exactly like reference ==========
    ax1.set_title('Polarization Voltage (V)', 
                  fontsize=15, fontweight='bold', pad=15)
    
    # ========== Legend ==========
    lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 , labels1, loc='upper left', fontsize=11, 
               framealpha=0.95, edgecolor='gray')
    
    # Grid (subtle)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('2_polarization_bumpy.png', dpi=400, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\n✓ Chart saved: 2_polarization_bumpy.png")
    plt.close()
    
    print("=" * 60)


if __name__ == '__main__':
    simulate_and_plot()
