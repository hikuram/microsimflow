import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Publication-quality font and style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1.5,
})

def extract_shape_from_recipe(recipe_str):
    """Extract and format the base shape name from the recipe string."""
    base = recipe_str.split(':')[0]
    mapping = {
        'sphere': 'Sphere',
        'flake': 'Flake',
        'rigidfiber': 'Rigid Fiber',
        'staggered': 'Staggered'
    }
    return mapping.get(base, base.capitalize())

def main():
    csv_file = "exp7_shape_stretch_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run exp7 first.")
        return

    # Extract shape names from recipe configurations
    df['Shape'] = df['Recipe'].apply(extract_shape_from_recipe)
    
    # Filter rows that contain valid chfem solver results
    df_clean = df.dropna(subset=['Stretch_Ratio', 'chfem_Kxx']).copy()

    # Calculate relative conductivity (K / K0)
    # K0 is the baseline conductivity at Stretch_Ratio == 1.0
    df_clean['Kxx_rel'] = np.nan
    df_clean['Kyy_rel'] = np.nan
    df_clean['Kzz_rel'] = np.nan

    # Group by 'Shape' and 'Basename' to match initial and stretched states
    groups = df_clean.groupby(['Shape', df_clean['Basename'].str.extract(r'(seed\d+)', expand=False)])
    
    for name, group in groups:
        baseline = group[group['Stretch_Ratio'] == 1.0]
        if baseline.empty:
            continue
            
        kxx_0 = baseline['chfem_Kxx'].values[0]
        kyy_0 = baseline['chfem_Kyy'].values[0]
        kzz_0 = baseline['chfem_Kzz'].values[0]
        
        # Normalize conductivities against the unstretched baseline
        df_clean.loc[group.index, 'Kxx_rel'] = group['chfem_Kxx'] / kxx_0
        df_clean.loc[group.index, 'Kyy_rel'] = group['chfem_Kyy'] / kyy_0
        df_clean.loc[group.index, 'Kzz_rel'] = group['chfem_Kzz'] / kzz_0

    directions = [
        ('X-direction (Stretch Axis)', 'chfem_Kxx', 'Kxx_rel'),
        ('Y-direction (Transverse)', 'chfem_Kyy', 'Kyy_rel'),
        ('Z-direction (Transverse)', 'chfem_Kzz', 'Kzz_rel')
    ]

    # =========================================================================
    # 1. Plot Absolute Conductivity (Kxx) in the Stretch Direction
    # =========================================================================
    print("\n--- Plotting Absolute Kxx ---")
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_clean, x='Stretch_Ratio', y='chfem_Kxx', hue='Shape', style='Shape',
        markers=True, dashes=False, linewidth=2.5, markersize=9, 
        err_style="band", palette="colorblind"
    )
    
    plt.title('Absolute Conductivity ($K_{xx}$) vs. Stretch Ratio', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(r'Stretch Ratio ($\lambda$)')
    plt.ylabel(r'Conductivity ($K_{xx}$)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title='Filler Geometry', frameon=True)
    
    out_img_abs = "exp7_absolute_Kxx.png"
    plt.savefig(out_img_abs, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{out_img_abs}'")
    plt.close()

    # =========================================================================
    # 2. Plot Normalized Degradation Curves (X, Y, Z)
    # =========================================================================
    print("\n--- Plotting Normalized Degradation Curves (X, Y, Z) ---")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle('Relative Conductivity ($K / K_0$) Under Tension by Filler Shape', fontsize=16, fontweight='bold', y=1.05)

    for ax, (title, _, col_rel) in zip(axes, directions):
        sns.lineplot(
            data=df_clean, x='Stretch_Ratio', y=col_rel, hue='Shape', style='Shape',
            markers=True, dashes=False, linewidth=2.5, markersize=9, 
            err_style="band", palette="colorblind", ax=ax
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r'Stretch Ratio ($\lambda$)')
        ax.set_ylabel(r'Relative Conductivity ($K / K_0$)')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        if ax != axes[2]:
            ax.get_legend().remove()
        else:
            ax.legend(title='Filler Geometry', bbox_to_anchor=(1.05, 1), loc='upper left')

    out_img_rel = "exp7_relative_K.png"
    plt.savefig(out_img_rel, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{out_img_rel}'")
    plt.close()

if __name__ == "__main__":
    main()
