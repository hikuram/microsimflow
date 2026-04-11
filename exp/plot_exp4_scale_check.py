import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def extract_placement_time(log_str):
    """Extract placement time in seconds from the log string."""
    # Matches patterns like "rigidfiber(ID:4):12.3s"
    match = re.search(r'rigidfiber\(ID:\d+\):([\d\.]+)s', str(log_str))
    if match:
        return float(match.group(1))
    return 0.0

def extract_radius(recipe_str):
    """Extract radius value from the recipe string."""
    match = re.search(r'radius=(\d+)', str(recipe_str))
    if match:
        return int(match.group(1))
    return None

def extract_vf(recipe_str):
    """Extract volume fraction (Vf) from the recipe string."""
    parts = str(recipe_str).split(':')
    if len(parts) > 1:
        return float(parts[1])
    return None

def plot_scale_check():
    # Fallback to multiple possible CSV names
    csv_files = ["scale_check_results.csv", "exp4_scale_check_results.csv"]
    csv_file = next((f for f in csv_files if os.path.exists(f)), None)
    
    if not csv_file:
        print("Error: Scale check CSV file not found.")
        return

    df = pd.read_csv(csv_file)

    # Parse parameters from strings
    df['Radius'] = df['Recipe'].apply(extract_radius)
    df['Target_Vf'] = df['Recipe'].apply(extract_vf)
    df['Placement_Time_s'] = df['Placement_Logs'].apply(extract_placement_time)
    
    # Calculate average effective conductivity (assumes isotropic random orientation)
    df['Avg_Conductivity'] = df[['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz']].mean(axis=1)

    # Group by Size, Radius, and Target_Vf and calculate means across different seeds
    grouped = df.groupby(['Size', 'Radius', 'Target_Vf']).agg({
        'Avg_Conductivity': 'mean',
        'Placement_Time_s': 'mean',
        'chfem_Time_s': 'mean'
    }).reset_index()

    # Layout: 1x3 subplots for overview
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define visual encodings: Color for Size, Marker for Radius
    colors = {200: '#1f77b4', 300: '#ff7f0e', 400: '#2ca02c'}
    markers = {2: 'o', 3: 's', 4: '^'}

    # 1. Conductivity Plot (Physics Check)
    ax1 = axes[0]
    for (s, r), subset in grouped.groupby(['Size', 'Radius']):
        subset = subset.sort_values('Target_Vf')
        ax1.plot(subset['Target_Vf'] * 100, subset['Avg_Conductivity'],
                 marker=markers.get(r, 'x'), color=colors.get(s, 'k'),
                 linestyle='-', linewidth=2, markersize=7, 
                 label=f'Size={s}, r={r}')

    ax1.set_xlabel('Volume Fraction (%)', fontsize=12)
    ax1.set_ylabel('Effective Conductivity', fontsize=12)
    ax1.set_title('1. Effective Conductivity\n(RVE & Resolution Consistency)', fontsize=14)
    ax1.set_yscale('log') # Use log scale if values span multiple orders of magnitude
    ax1.grid(True, linestyle='--', alpha=0.7, which='both')
    ax1.legend(fontsize=10)

    # 2. Placement Time Plot (Algorithm Scaling Check)
    ax2 = axes[1]
    for (s, r), subset in grouped.groupby(['Size', 'Radius']):
        subset = subset.sort_values('Target_Vf')
        ax2.plot(subset['Target_Vf'] * 100, subset['Placement_Time_s'],
                 marker=markers.get(r, 'x'), color=colors.get(s, 'k'),
                 linestyle='--', linewidth=2, markersize=7)

    ax2.set_xlabel('Volume Fraction (%)', fontsize=12)
    ax2.set_ylabel('RSA Placement Time (s)', fontsize=12)
    ax2.set_title('2. Structure Generation Time\n(Numba Scaling Check)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. Solver Time Plot (GPU Scaling Check)
    ax3 = axes[2]
    for (s, r), subset in grouped.groupby(['Size', 'Radius']):
        subset = subset.sort_values('Target_Vf')
        ax3.plot(subset['Target_Vf'] * 100, subset['chfem_Time_s'],
                 marker=markers.get(r, 'x'), color=colors.get(s, 'k'),
                 linestyle='-.', linewidth=2, markersize=7)

    ax3.set_xlabel('Volume Fraction (%)', fontsize=12)
    ax3.set_ylabel('chfem Solver Time (s)', fontsize=12)
    ax3.set_title('3. Homogenization Solver Time\n(GPU Memory/Computation Check)', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Final adjustments
    plt.suptitle("Exp 4: Scale, Resolution, and Computational Cost Overview", fontsize=16, y=1.05)
    plt.tight_layout()
    
    out_img = "exp4_scale_check_plot.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {out_img}")
    plt.close()

if __name__ == "__main__":
    plot_scale_check()
