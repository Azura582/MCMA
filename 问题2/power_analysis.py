"""
Power Consumption Analysis for 5 Scenarios
Nested Donut Chart (Ring Chart) Visualization
Each ring represents one scenario with different radii
"""
import numpy as np
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


def calculate_power_breakdown(model, scenario_func, duration_h=1.0):
    """
    Calculate power consumption breakdown for each component
    Returns: dict with power in Wh for each component
    """
    scenario = scenario_func(0)
    
    power = {
        'Screen': 0.0,
        'CPU': 0.0,
        'Network': 0.0,
        'GPS': 0.0,
        'Base': model.P_base * duration_h
    }
    
    # Screen power
    if scenario.get('screen_on', False):
        brightness = scenario.get('brightness', 0.5)
        P_screen = model.P_a * brightness * model.P_refresh * model.P_screen_square
        power['Screen'] = P_screen * duration_h
    
    # CPU power
    cpu_usage = scenario.get('cpu_usage', 0.0)
    P_cpu = model.P_cpu_idle + cpu_usage * model.P_cpu_B * (model.P_cpu_f ** 2)
    power['CPU'] = P_cpu * duration_h
    
    # Network power
    data_rate = scenario.get('data_rate', 0.0)
    P_net = model.P_net_idle + model.beta * data_rate
    power['Network'] = P_net * duration_h
    
    # GPS power
    if scenario.get('gps_on', False):
        power['GPS'] = model.P_gps * duration_h
    
    return power


def analyze_all_scenarios():
    """Analyze power breakdown for all 5 scenarios"""
    model = SmartphoneBatteryModel()
    
    scenarios = {
        'Video Streaming': scenario_video_streaming,
        'Gaming': scenario_gaming,
        'Navigation': scenario_navigation,
        'Idle': scenario_free,
        'Cold Weather': scenario_cold_weather
    }
    
    duration_h = 12.0
    results = {}
    
    for name, func in scenarios.items():
        power = calculate_power_breakdown(model, func, duration_h)
        results[name] = power
        
        total = sum(power.values())
        print(f"\n{name}:")
        print(f"  Screen:  {power['Screen']:.4f} Wh ({power['Screen']/total*100:.1f}%)" if total > 0 else "  Screen:  0 Wh")
        print(f"  CPU:     {power['CPU']:.4f} Wh ({power['CPU']/total*100:.1f}%)" if total > 0 else "  CPU:     0 Wh")
        print(f"  Network: {power['Network']:.4f} Wh ({power['Network']/total*100:.1f}%)" if total > 0 else "  Network: 0 Wh")
        print(f"  GPS:     {power['GPS']:.4f} Wh ({power['GPS']/total*100:.1f}%)" if total > 0 else "  GPS:     0 Wh")
        print(f"  Base:    {power['Base']:.4f} Wh ({power['Base']/total*100:.1f}%)" if total > 0 else "  Base:    0 Wh")
        print(f"  TOTAL:   {total:.4f} Wh")
    
    return results


def plot_nested_donut(results):
    """
    Create nested donut chart with 5 rings (one per scenario)
    Each ring at different radius
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Component colors
    colors = {
        'Screen': '#E74C3C',
        'CPU': '#3498DB',
        'Network': '#2ECC71',
        'GPS': '#F39C12',
        'Base': '#9B59B6'
    }
    
    components = ['Screen', 'CPU', 'Network', 'GPS', 'Base']
    
    # Scenario order (inside to outside) - sorted by total power
    scenario_totals = {name: sum(power.values()) for name, power in results.items()}
    scenario_order = sorted(scenario_totals.keys(), key=lambda x: scenario_totals[x])
    
    # Ring parameters
    width = 0.12
    start_radius = 0.25
    gap = 0.14
    
    # Create each ring
    for i, scenario_name in enumerate(scenario_order):
        power_data = results[scenario_name]
        inner_radius = start_radius + i * gap
        
        values = [power_data[comp] for comp in components]
        ring_colors = [colors[comp] for comp in components]
        
        total = sum(values)
        if total == 0:
            # For zero total, show equal segments in gray
            values = [1, 1, 1, 1, 1]
            ring_colors = ['#CCCCCC'] * 5
        
        # Create pie as donut ring - BLACK edge
        wedges, _ = ax.pie(
            values,
            radius=inner_radius + width,
            colors=ring_colors,
            wedgeprops=dict(width=width, edgecolor='black', linewidth=1.2),
            startangle=90,
            counterclock=False
        )
        
        # Add percentage labels on each segment
        # Calculate angles for each segment
        total_val = sum(values)
        cumsum = 0
        for j, (wedge, val) in enumerate(zip(wedges, values)):
            if val / total_val < 0.03:  # Skip very small segments (<3%)
                cumsum += val
                continue
            
            # Calculate midpoint angle of wedge
            pct = val / total_val
            mid_angle = 90 - (cumsum + val / 2) / total_val * 360  # degrees
            mid_rad = np.radians(mid_angle)
            
            # Position at middle of ring
            r = inner_radius + width / 2
            x_pct = r * np.cos(mid_rad)
            y_pct = r * np.sin(mid_rad)
            
            # Add percentage text
            ax.text(x_pct, y_pct, f'{pct*100:.0f}%',
                   ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.5))
            
            cumsum += val
        
        # Add scenario label - moved to LEFT side with connection line
        label_radius = inner_radius + width / 2
        
        # Position labels on the left side, stacked vertically
        x_label = -1.15  # Fixed x position on left
        y_label = 0.4 - i * 0.18  # Stack vertically
        
        # Connection point on the ring (left side)
        x_ring = -label_radius * 0.95
        y_ring = label_radius * 0.3
        
        # Draw connection line
        ax.plot([x_label + 0.12, x_ring], [y_label, y_ring], 
                color='#2C3E50', linewidth=1, linestyle='-', alpha=0.6)
        
        ax.annotate(
            f'{scenario_name}: {total:.1f} Wh',
            xy=(x_label, y_label),
            fontsize=10,
            fontweight='bold',
            ha='left',
            va='center',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=ring_colors[0], alpha=0.3, edgecolor='#2C3E50')
        )
    
    # Center text
    ax.text(0, 0, 'Power\nConsumption\n', 
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='circle,pad=0.5', facecolor='#ECF0F1', edgecolor='#2C3E50', linewidth=2))
    
    # Legend - closer to chart
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[comp], 
                                       edgecolor='black', linewidth=1.2, label=comp) 
                       for comp in components]
    ax.legend(handles=legend_elements, loc='upper right', 
              fontsize=12, title='Components', title_fontsize=13,
              bbox_to_anchor=(1.05, 0.95), frameon=True, 
              facecolor='white', edgecolor='#2C3E50')
    
    # Title
    #ax.set_title('Power Consumption',
    #            fontsize=16, fontweight='bold', pad=25)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('power_radar_chart.png', dpi=400, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("\n✓ Figure saved: power_radar_chart.png")
    plt.close()


if __name__ == '__main__':
    print("="*60)
    print("POWER CONSUMPTION ANALYSIS - 5 SCENARIOS")
    print("Components: Screen, CPU, Network, GPS, Base")
    print("Duration: 12 hour")
    print("="*60)
    
    results = analyze_all_scenarios()
    plot_nested_donut(results)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)