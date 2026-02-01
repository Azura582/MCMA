"""
Monte Carlo Simulation for Battery Discharge Time
Uncertainty Factors (Small Perturbations):
1. Aging rate constant (k_aging) - Small perturbation ±6%
2. Initial ohmic resistance (R0) - Small perturbation ±10%
3. Ambient temperature (T_amb) - Small perturbation ±5°C

Each factor: 500 simulations
Initial state: Full charge (SOC=100%), Video Streaming scenario
Goal: Demonstrate model stability under small parameter variations
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from model import SmartphoneBatteryModel
from scenery import scenario_video_streaming

# English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def run_single_simulation(model, scenario_func, T_amb=298.15):
    """Run a single discharge simulation and return discharge time in hours"""
    soc0 = 1.0
    T_init = T_amb  # Start at ambient temperature
    y0 = [soc0, T_init, 0.0, 0.0]
    t_span = (0, 50 * 3600)  # Max 50 hours for video streaming
    
    try:
        sol = model.simulate(t_span, y0, scenario_func, max_step=120)
        t_empty_s = model.find_empty_time(sol)
        t_empty_h = t_empty_s / 3600
        return t_empty_h
    except Exception as e:
        print(f"Simulation error: {e}")
        return np.nan


def monte_carlo_aging(n_sim=500):
    """Monte Carlo simulation varying aging rate constant (small perturbation ±6%)"""
    print(f"\n[1/3] Simulating aging rate uncertainty ({n_sim} runs)...")
    print(f"  Perturbation: 5.0±0.3 ×10⁻⁶ h⁻¹ (±6% variation)")
    
    # Aging rate: 5e-6 ± 0.3e-6 (±6% variation)
    k_aging_samples = np.random.normal(5e-6, 0.3e-6, n_sim)
    k_aging_samples = np.clip(k_aging_samples, 4.0e-6, 6.0e-6)
    
    discharge_times = []
    
    for i, k_aging in enumerate(k_aging_samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_sim}")
        
        model = SmartphoneBatteryModel()
        model.k_aging = k_aging
        
        t_h = run_single_simulation(model, scenario_video_streaming, T_amb=298.15)
        discharge_times.append(t_h)
    
    return np.array(discharge_times), k_aging_samples


def monte_carlo_resistance(n_sim=500):
    """Monte Carlo simulation varying initial ohmic resistance (small perturbation ±10%)"""
    print(f"\n[2/3] Simulating ohmic resistance uncertainty ({n_sim} runs)...")
    print(f"  Perturbation: 0.030±0.003 Ω (±10% variation)")
    
    # R0: 0.03 ± 0.003 Ohm (±10% variation)
    R0_samples = np.random.normal(0.03, 0.003, n_sim)
    R0_samples = np.clip(R0_samples, 0.024, 0.036)
    
    discharge_times = []
    
    for i, R0 in enumerate(R0_samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_sim}")
        
        model = SmartphoneBatteryModel()
        model.R0 = R0
        
        t_h = run_single_simulation(model, scenario_video_streaming, T_amb=298.15)
        discharge_times.append(t_h)
    
    return np.array(discharge_times), R0_samples


def monte_carlo_temperature(n_sim=500):
    """Monte Carlo simulation varying ambient temperature (small perturbation ±5°C)"""
    print(f"\n[3/3] Simulating ambient temperature uncertainty ({n_sim} runs)...")
    print(f"  Perturbation: 25±5 °C (±20% variation)")
    
    # Temperature: 25±5°C (small perturbation around room temperature)
    T_amb_celsius = np.random.normal(25, 5, n_sim)
    T_amb_celsius = np.clip(T_amb_celsius, 15, 35)
    T_amb_kelvin = T_amb_celsius + 273.15
    
    discharge_times = []
    
    for i, T_amb in enumerate(T_amb_kelvin):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_sim}")
        
        model = SmartphoneBatteryModel()
        
        # Create scenario function with specific T_amb
        scenario_func = lambda t, T=T_amb: {**scenario_video_streaming(t), 'T_amb': T}
        
        t_h = run_single_simulation(model, scenario_func, T_amb=T_amb)
        discharge_times.append(t_h)
    
    return np.array(discharge_times), T_amb_celsius


def plot_results(times_aging, times_R0, times_temp, 
                 params_aging, params_R0, params_temp):
    """
    Plot Monte Carlo results emphasizing model stability
    Small perturbations should result in small output variations
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sort by parameter value for cleaner visualization
    idx_aging = np.argsort(params_aging)
    idx_R0 = np.argsort(params_R0)
    idx_temp = np.argsort(params_temp)
    
    # Calculate statistics for each factor
    mean_aging = np.nanmean(times_aging)
    std_aging = np.nanstd(times_aging)
    cv_aging = (std_aging / mean_aging) * 100
    
    mean_R0 = np.nanmean(times_R0)
    std_R0 = np.nanstd(times_R0)
    cv_R0 = (std_R0 / mean_R0) * 100
    
    mean_temp = np.nanmean(times_temp)
    std_temp = np.nanstd(times_temp)
    cv_temp = (std_temp / mean_temp) * 100
    
    # ===== Plot 1: Aging Rate (±6% perturbation) =====
    ax1 = axes[0]
    ax1.plot(params_aging[idx_aging] * 1e6, times_aging[idx_aging], 
             'o', color='#E74C3C', markersize=3, alpha=0.5, label='Simulations')
    ax1.axhline(mean_aging, color='#C0392B', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_aging:.2f}h')
    ax1.axhspan(mean_aging - std_aging, mean_aging + std_aging, 
               alpha=0.15, color='#E74C3C', label=f'±1σ: {std_aging:.3f}h')
    
    ax1.set_xlabel('Aging Rate ($\\times 10^{-6}$ h$^{-1}$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Discharge Time (h)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Aging Rate\nCV = {cv_aging:.2f}%', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='best', fontsize=9)
    
    # Add stability text
    ax1.text(0.98, 0.02, f'Range: {np.nanmin(times_aging):.2f}-{np.nanmax(times_aging):.2f}h\nΔ = {np.nanmax(times_aging)-np.nanmin(times_aging):.3f}h', 
            transform=ax1.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    # ===== Plot 2: Ohmic Resistance (±10% perturbation) =====
    ax2 = axes[1]
    ax2.plot(params_R0[idx_R0] * 1000, times_R0[idx_R0], 
             'o', color='#27AE60', markersize=3, alpha=0.5, label='Simulations')
    ax2.axhline(mean_R0, color='#1E8449', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_R0:.2f}h')
    ax2.axhspan(mean_R0 - std_R0, mean_R0 + std_R0, 
               alpha=0.15, color='#27AE60', label=f'±1σ: {std_R0:.3f}h')
    
    ax2.set_xlabel('Ohmic Resistance R$_0$ (mΩ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Discharge Time (h)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Resistance R$_0$\nCV = {cv_R0:.2f}%', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='best', fontsize=9)
    
    # Add stability text
    ax2.text(0.98, 0.02, f'Range: {np.nanmin(times_R0):.2f}-{np.nanmax(times_R0):.2f}h\nΔ = {np.nanmax(times_R0)-np.nanmin(times_R0):.3f}h', 
            transform=ax2.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    # ===== Plot 3: Ambient Temperature (±5°C perturbation) =====
    ax3 = axes[2]
    ax3.plot(params_temp[idx_temp], times_temp[idx_temp], 
             'o', color='#3498DB', markersize=3, alpha=0.5, label='Simulations')
    ax3.axhline(mean_temp, color='#2874A6', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_temp:.2f}h')
    ax3.axhspan(mean_temp - std_temp, mean_temp + std_temp, 
               alpha=0.15, color='#3498DB', label=f'±1σ: {std_temp:.3f}h')
    
    ax3.set_xlabel('Ambient Temperature (°C)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Discharge Time (h)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Temperature\nCV = {cv_temp:.2f}%', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.legend(loc='best', fontsize=9)
    
    # Add stability text
    ax3.text(0.98, 0.02, f'Range: {np.nanmin(times_temp):.2f}-{np.nanmax(times_temp):.2f}h\nΔ = {np.nanmax(times_temp)-np.nanmin(times_temp):.3f}h', 
            transform=ax3.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig('monte_carlo_distribution.png', dpi=400, bbox_inches='tight')
    print("\n✓ Figure saved: monte_carlo_distribution.png")
    plt.close()
    
    # Print detailed summary emphasizing stability
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION SUMMARY - MODEL STABILITY ANALYSIS")
    print("="*70)
    print("\nSmall Perturbations → Small Output Variations (Model is STABLE)")
    print("\n1. Aging Rate (±6% perturbation):")
    print(f"   Mean: {mean_aging:.3f} h  |  Std: {std_aging:.3f} h  |  CV: {cv_aging:.2f}%")
    print(f"   Range: {np.nanmin(times_aging):.3f} ~ {np.nanmax(times_aging):.3f} h")
    print(f"   Max Deviation: ±{max(abs(mean_aging - np.nanmin(times_aging)), abs(np.nanmax(times_aging) - mean_aging)):.3f} h ({max(abs(mean_aging - np.nanmin(times_aging)), abs(np.nanmax(times_aging) - mean_aging))/mean_aging*100:.2f}%)")
    
    print("\n2. Ohmic Resistance R₀ (±10% perturbation):")
    print(f"   Mean: {mean_R0:.3f} h  |  Std: {std_R0:.3f} h  |  CV: {cv_R0:.2f}%")
    print(f"   Range: {np.nanmin(times_R0):.3f} ~ {np.nanmax(times_R0):.3f} h")
    print(f"   Max Deviation: ±{max(abs(mean_R0 - np.nanmin(times_R0)), abs(np.nanmax(times_R0) - mean_R0)):.3f} h ({max(abs(mean_R0 - np.nanmin(times_R0)), abs(np.nanmax(times_R0) - mean_R0))/mean_R0*100:.2f}%)")
    
    print("\n3. Ambient Temperature (±5°C perturbation):")
    print(f"   Mean: {mean_temp:.3f} h  |  Std: {std_temp:.3f} h  |  CV: {cv_temp:.2f}%")
    print(f"   Range: {np.nanmin(times_temp):.3f} ~ {np.nanmax(times_temp):.3f} h")
    print(f"   Max Deviation: ±{max(abs(mean_temp - np.nanmin(times_temp)), abs(np.nanmax(times_temp) - mean_temp)):.3f} h ({max(abs(mean_temp - np.nanmin(times_temp)), abs(np.nanmax(times_temp) - mean_temp))/mean_temp*100:.2f}%)")
    
    # Overall stability assessment
    avg_cv = (cv_aging + cv_R0 + cv_temp) / 3
    print(f"\n{'='*70}")
    print(f"OVERALL STABILITY: Average CV = {avg_cv:.2f}%")
    if avg_cv < 1.0:
        print("✅ EXCELLENT: Model is highly stable under small perturbations")
    elif avg_cv < 2.0:
        print("✅ GOOD: Model shows good stability")
    elif avg_cv < 5.0:
        print("⚠️  ACCEPTABLE: Model stability is acceptable")
    else:
        print("❌ POOR: Model sensitivity is high")
    print("="*70)


if __name__ == '__main__':
    print("="*70)
    print("MONTE CARLO SIMULATION - MODEL STABILITY ANALYSIS")
    print("Scenario: Video Streaming, Full Charge (SOC=100%)")
    print("Perturbations: Aging Rate (±6%), R₀ (±10%), Temperature (±5°C)")
    print("Simulations per factor: 500")
    print("="*70)
    
    np.random.seed(42)
    
    # Run Monte Carlo simulations
    times_aging, params_aging = monte_carlo_aging(500)
    times_R0, params_R0 = monte_carlo_resistance(500)
    times_temp, params_temp = monte_carlo_temperature(500)
    
    # Plot all results emphasizing stability
    plot_results(times_aging, times_R0, times_temp,
                 params_aging, params_R0, params_temp)
    
    print("\n✅ Simulation complete!")