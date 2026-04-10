import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_exp5_morphology():
    csv_file = "exp5_morphology.csv"
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # Target volume fraction parsing for clean grouping
    df['Vf_A'] = df['PolymerA_Frac'].round(2)
    
    # Group by background type and volume fraction
    grouped = df.groupby(['BG_Type', 'Vf_A'])[['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz']].mean().reset_index()
    
    # Theoretical Limits (Rule of Mixtures)
    E_a, E_b = 10.0, 1000.0
    
    # Create smooth theoretical curves from Vf_A = 0.0 to 1.0
    v_a_smooth = np.linspace(0, 1.0, 100)
    v_b_smooth = 1.0 - v_a_smooth
    E_voigt = v_a_smooth * E_a + v_b_smooth * E_b
    E_reuss = (E_a * E_b) / (v_a_smooth * E_b + v_b_smooth * E_a)

    # Setup 1x3 subplots for X, Y, Z directions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    directions = [('chfem_Kxx', 'X-Direction'),
                  ('chfem_Kyy', 'Y-Direction'),
                  ('chfem_Kzz', 'Z-Direction')]
                  
    markers = {'lamellar': 's', 'cylinder': '^', 'gyroid': 'o', 'sea_island': 'D'}

    for i, (col, title) in enumerate(directions):
        ax = axes[i]
        
        # Plot theoretical bounds and fill area between them
        ax.plot(v_a_smooth, E_voigt, 'k--', label='Voigt (Upper Bound)', linewidth=1.5)
        ax.plot(v_a_smooth, E_reuss, 'k:', label='Reuss (Lower Bound)', linewidth=1.5)
        ax.fill_between(v_a_smooth, E_reuss, E_voigt, color='gray', alpha=0.1)

        # Plot simulation results for each morphology
        for bg in grouped['BG_Type'].unique():
            subset = grouped[grouped['BG_Type'] == bg].sort_values('Vf_A')
            ax.plot(subset['Vf_A'], subset[col], marker=markers.get(bg, 'x'), 
                    label=bg.capitalize(), markersize=8, linewidth=2)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Volume Fraction of Soft Phase A ($V_A$)", fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1100)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.set_ylabel("Effective Stiffness", fontsize=12)
        if i == 2:
            # Place legend outside the last plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=11)

    plt.suptitle("Tutorial Part 1: Morphology Anisotropy vs. Rule of Mixtures\n(Phase A: E=10.0, Phase B: E=1000.0)", 
                 fontsize=16, y=1.05)
    
    # Save the figure
    plt.savefig("exp5_morphology_plot.png", dpi=300, bbox_inches='tight')
    print("Saved exp5 plot as 'exp5_morphology_plot.png'")
    plt.close()

if __name__ == "__main__":
    plot_exp5_morphology()
