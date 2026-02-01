"""
Monte Carlo Simulation for Battery Discharge Time
Uncertainty Factors:
1. Aging rate constant (k_aging)
2. Ambient temperature (T_amb) - Large variation
3. Initial ohmic resistance (R0)

Each factor: 500 simulations
Initial state: Full charge (SOC=100%), Idle scenario
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from model import SmartphoneBatteryModel

# English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def scenario_idle(t, T_amb=298.15):
    """Idle scenario - minimal power consumption"""
    return {
        'screen_on': False,
        'brightness': 0.0,
        'cpu_usage': 0.05,  # Minimal CPU
        'data_rate': 0.1,   # Background sync
        'gps_on': False,
        'T_amb': T_amb
    }


def run_single_simulation(model, T_amb=298.15):
    """Run a single discharge simulation and return discharge time in hours"""
    soc0 = 1.0
    T_init = T_amb  # Start at ambient temperature
    y0 = [soc0, T_init, 0.0, 0.0]
    t_span = (0, 200 * 3600)  # Max 200 hours for idle
    
    # Create scenario function with specific T_amb
    scenario_func = lambda t: scenario_idle(t, T_amb)
    
    try:
        sol = model.simulate(t_span, y0, scenario_func, max_step=120)
        t_empty_s = model.find_empty_time(sol)
        t_empty_h = t_empty_s / 3600
        return t_empty_h
    except Exception as e:
        print(f"Simulation error: {e}")
        return np.nan


def monte_carlo_aging(n_sim=500):
    """Monte Carlo simulation varying aging rate constant"""
    print(f"\n[1/3] Simulating aging rate uncertainty ({n_sim} runs)...")
    
    # Aging rate varies: mean=5e-6, std=2e-6
    k_aging_samples = np.random.normal(5e-6, 2e-6, n_sim)
    k_aging_samples = np.clip(k_aging_samples, 1e-7, 1e-5)
    
    discharge_times = []
    
    for i, k_aging in enumerate(k_aging_samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_sim}")
        
        model = SmartphoneBatteryModel()
        model.k_aging = k_aging
        
        t_h = run_single_simulation(model)
        discharge_times.append(t_h)
    
    return np.array(discharge_times), k_aging_samples


def monte_carlo_temperature(n_sim=500):
    """Monte Carlo simulation varying ambient temperature (large variation)"""
    print(f"\n[2/3] Simulating ambient temperature uncertainty ({n_sim} runs)...")
    
    # Temperature varies: -20°C to +45°C - LARGE variation
    T_amb_celsius = np.random.uniform(-20, 45, n_sim)
    T_amb_kelvin = T_amb_celsius + 273.15
    
    discharge_times = []
    
    for i, T_amb in enumerate(T_amb_kelvin):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_sim}")
        
        model = SmartphoneBatteryModel()
        t_h = run_single_simulation(model, T_amb=T_amb)
        discharge_times.append(t_h)
    
    return np.array(discharge_times), T_amb_celsius


def monte_carlo_resistance(n_sim=500):
    """Monte Carlo simulation varying initial ohmic resistance R0"""
    print(f"\n[3/3] Simulating ohmic resistance uncertainty ({n_sim} runs)...")
    
    # R0 varies: mean=0.03, std=0.01
    R0_samples = np.random.normal(0.03, 0.01, n_sim)
    R0_samples = np.clip(R0_samples, 0.005, 0.08)
    
    discharge_times = []
    
    for i, R0 in enumerate(R0_samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_sim}")
        
        model = SmartphoneBatteryModel()
        model.R0 = R0
        
        t_h = run_single_simulation(model)
        discharge_times.append(t_h)
    
    return np.array(discharge_times), R0_samples


def plot_results(times_aging, times_temp, times_R0, 
                 params_aging, params_temp, params_R0):
    """Plot all Monte Carlo results on one figure"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sort by parameter value for cleaner visualization
    idx_aging = np.argsort(params_aging)
    idx_temp = np.argsort(params_temp)
    idx_R0 = np.argsort(params_R0)
    
    # ===== Plot 1: Aging Rate =====
    ax1 = axes[0]
    ax1.plot(params_aging[idx_aging] * 1e6, times_aging[idx_aging], 
             'o-', color='#E74C3C', markersize=2, linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Aging Rate ($\\times 10^{-6}$ h$^{-1}$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Discharge Time (h)', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Aging Rate', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    mean_aging = np.nanmean(times_aging)
    std_aging = np.nanstd(times_aging)
    ax1.axhline(mean_aging, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_aging:.1f}h')
    ax1.legend(loc='best', fontsize=10)
    
    # ===== Plot 2: Ambient Temperature =====
    ax2 = axes[1]
    ax2.plot(params_temp[idx_temp], times_temp[idx_temp], 
             'o-', color='#3498DB', markersize=2, linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Ambient Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Discharge Time (h)', fontsize=12, fontweight='bold')
    ax2.set_title('Effect of Temperature', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    mean_temp = np.nanmean(times_temp)
    ax2.axhline(mean_temp, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_temp:.1f}h')
    ax2.legend(loc='best', fontsize=10)
    
    # ===== Plot 3: Ohmic Resistance =====
    ax3 = axes[2]
    ax3.plot(params_R0[idx_R0] * 1000, times_R0[idx_R0], 
             'o-', color='#27AE60', markersize=2, linewidth=0.5, alpha=0.7)
    ax3.set_xlabel('Ohmic Resistance R$_0$ (mΩ)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Discharge Time (h)', fontsize=12, fontweight='bold')
    ax3.set_title('Effect of Resistance', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    mean_R0 = np.nanmean(times_R0)
    ax3.axhline(mean_R0, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_R0:.1f}h')
    ax3.legend(loc='best', fontsize=10)
    
    #plt.suptitle('Monte Carlo Simulation: Battery Discharge Time Under Uncertainty', 
    #             fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_distribution.png', dpi=400, bbox_inches='tight')
    print("\n✓ Figure saved: monte_carlo_distribution.png")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION SUMMARY")
    print("="*70)
    print(f"\n1. Aging Rate: {np.nanmean(times_aging):.2f} ± {np.nanstd(times_aging):.2f} h")
    print(f"   Range: {np.nanmin(times_aging):.2f} ~ {np.nanmax(times_aging):.2f} h")
    print(f"\n2. Temperature: {np.nanmean(times_temp):.2f} ± {np.nanstd(times_temp):.2f} h")
    print(f"   Range: {np.nanmin(times_temp):.2f} ~ {np.nanmax(times_temp):.2f} h")
    print(f"\n3. Resistance R0: {np.nanmean(times_R0):.2f} ± {np.nanstd(times_R0):.2f} h")
    print(f"   Range: {np.nanmin(times_R0):.2f} ~ {np.nanmax(times_R0):.2f} h")
    print("="*70)


if __name__ == '__main__':
    print("="*70)
    print("MONTE CARLO SIMULATION FOR BATTERY DISCHARGE TIME")
    print("Scenario: Idle Mode, Full Charge (SOC=100%)")
    print("Factors: Aging Rate, Ambient Temperature, Ohmic Resistance")
    print("Simulations per factor: 500")
    print("="*70)
    
    np.random.seed(42)
    
    # Run Monte Carlo simulations
    times_aging, params_aging = monte_carlo_aging(500)
    times_temp, params_temp = monte_carlo_temperature(500)
    times_R0, params_R0 = monte_carlo_resistance(500)
    
    
    # Plot all results
    plot_results(times_aging, times_temp, times_R0,
                 params_aging, params_temp, params_R0)
    
    print("\nSimulation complete!")