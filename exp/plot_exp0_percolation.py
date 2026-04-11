import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit

# Publication-quality font and style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
})

def percolation_power_law(v, k0, vc, t):
    """
    Percolation theory model: K = K0 * (Vf - Vc)^t  (for Vf > Vc)
    Adds a minute value 1e-6 for log-scale plotting when Vf falls below Vc.
    """
    return k0 * np.maximum(v - vc, 0)**t + 1e-6

def create_slice_montage(df, out_prefix="exp0_montage"):
    """Create a horizontal montage of representative slice images for each volume fraction"""
    # Group by the extracted filler radius
    for r, group in df.groupby('Radius'):
        # Get the first row (representative data) for each volume fraction
        representative_rows = group.sort_values('Total_Vf').groupby('Total_Vf').first().reset_index()
        
        n_images = len(representative_rows)
        if n_images == 0:
            continue
            
        fig, axes = plt.subplots(1, n_images, figsize=(3 * n_images, 4))
        if n_images == 1:
            axes = [axes]
            
        fig.suptitle(f'Microstructure Evolution (Radius = {r})', fontsize=18, fontweight='bold', y=0.98)
        
        for ax, (_, row) in zip(axes, representative_rows.iterrows()):
            img_path = f"{row['Basename']}_slice.png"
            vf_pct = row['Total_Vf'] * 100
            k_eff = row['K_eff']
            
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=12)
                
            # Add volume fraction (%) and conductivity to each image title
            ax.set_title(f"$V_f$ = {vf_pct:.1f}%\n$K$ = {k_eff:.1e}", fontsize=14)
            ax.axis('off')
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_name = f"{out_prefix}_r{r}.png"
        plt.savefig(out_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Created slice montage: {out_name}")

def main():
    csv_file = "exp0_percolation_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run exp0 first.")
        return

    # Extract filler fraction and radius from the Recipe string
    df['Total_Vf'] = df['Filler_Frac']
    df['Radius'] = df['Recipe'].str.extract(r'radius=(\d+)').astype(int)
    
    # Macroscopic electrical conductivity (assuming isotropy)
    df['K_eff'] = (df['chfem_Kxx'] + df['chfem_Kyy'] + df['chfem_Kzz']) / 3.0

    # Drop NaNs or missing values
    df_clean = df.dropna(subset=['Total_Vf', 'K_eff', 'Radius'])

    if len(df_clean) == 0:
        print("No valid data found to plot.")
        return

    # 1. Generate montage images
    print("\n--- Generating Montages ---")
    create_slice_montage(df_clean)

    # 2. Plot percolation curves
    print("\n--- Plotting Percolation Curves ---")
    plt.figure(figsize=(8, 6))
    
    # Define colors and markers for the 3 levels
    colors = {2: '#0072B2', 4: '#D55E00', 6: '#009E73'}
    markers = {2: 'o', 4: 's', 6: '^'}

    radii = sorted(df_clean['Radius'].unique())

    for r in radii:
        group = df_clean[df_clean['Radius'] == r]
        Vf_clean = group['Total_Vf'].values
        K_eff_clean = group['K_eff'].values
        
        c = colors.get(r, '#000000')
        m = markers.get(r, 'o')

        # Scatter plot for simulation data
        plt.scatter(Vf_clean * 100, K_eff_clean, color=c, marker=m, edgecolor='k', s=80, alpha=0.7, label=f'r={r} (Sim)', zorder=3)

        # Estimate percolation threshold (Vc) and critical exponent (t) via curve fitting
        try:
            popt, _ = curve_fit(
                percolation_power_law, Vf_clean, K_eff_clean,
                p0=[1000.0, 0.02, 1.5],
                bounds=([0.0, 0.0, 1.0], [np.inf, 0.2, 4.0]) 
            )
            k0_fit, vc_fit, t_fit = popt
            
            print(f"[Radius={r}] Fitted: K0={k0_fit:.2f}, Vc={vc_fit*100:.2f}%, t={t_fit:.2f}")

            # Smooth curve for plotting
            v_smooth = np.linspace(0, max(Vf_clean) * 1.1, 200)
            k_smooth = percolation_power_law(v_smooth, k0_fit, vc_fit, t_fit)
            
            formula_label = f'Fit (r={r}): $V_c={vc_fit*100:.1f}\\%$'
            plt.plot(v_smooth * 100, k_smooth, color=c, linestyle='--', linewidth=2.5, label=formula_label, zorder=2)
                     
        except RuntimeError:
            print(f"[Radius={r}] Curve fitting failed. Data might not show clear percolation yet.")

    plt.xlabel('Filler Volume Fraction (%)')
    plt.ylabel('Effective Conductivity (S/m)') 
    plt.title('Percolation Threshold Analysis by Filler Radius')
    
    plt.yscale('log')
    plt.ylim(bottom=1e-5) 
    plt.xlim(left=0)
    
    plt.grid(True, which="both", ls="--", alpha=0.5, zorder=1)
    plt.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    
    out_img = 'exp0_percolation_plot.png'
    plt.savefig(out_img, dpi=300)
    print(f"Plot saved as '{out_img}'")

if __name__ == "__main__":
    main()
