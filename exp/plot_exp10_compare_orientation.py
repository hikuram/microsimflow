import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
})

def create_montage_exp10(df, seed, vf, out_filename, scale_factor=2.0):
    df_seed = df[(df['Seed'] == seed) & (df['VF'] == vf)].copy()
    if df_seed.empty: return

    profiles = sorted(df_seed['Profile'].unique())
    kappas = sorted(df_seed['Kappa'].unique())
    
    col_max_native_w = {k: 0 for k in kappas}
    max_native_h = 0
    img_cache = {}

    for kappa in kappas:
        for prof in profiles:
            cell_data = df_seed[(df_seed['Profile'] == prof) & (df_seed['Kappa'] == kappa)]
            if not cell_data.empty:
                img_path = cell_data.iloc[0].get('Slice_Image', f"{cell_data.iloc[0]['Basename']}_slice.png")
                if pd.notna(img_path) and os.path.exists(str(img_path)):
                    with Image.open(str(img_path)) as img:
                        w, h = img.size
                        img_cache[(prof, kappa)] = (w, h, str(img_path))
                        col_max_native_w[kappa] = max(col_max_native_w[kappa], w)
                        max_native_h = max(max_native_h, h)

    if max_native_h == 0:
        print(f"No valid slice images found for seed {seed}, VF {vf}.")
        return

    row_h = int(max_native_h * scale_factor)
    col_widths = [int(col_max_native_w[k] * scale_factor) for k in kappas]

    cell_padding = 15
    margin_top = 100
    margin_left = 350
    label_h = 45

    total_w = margin_left + sum(col_widths) + cell_padding * len(col_widths) + 40
    total_h = margin_top + len(profiles) * (row_h + label_h + 20) + 40

    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)

    try:
        font_prop = fm.FontProperties(family=plt.rcParams['font.sans-serif'])
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        font = ImageFont.truetype(font_path, 22)
        font_header = ImageFont.truetype(font_path, 22)
        font_title = ImageFont.truetype(font_path, 34)
    except Exception as e:
        font = font_header = font_title = ImageFont.load_default()

    title_text = f"Exp10: Orientation Kappa (Seed {int(seed)}, VF={vf:.2f})"
    draw.text((total_w // 2, 20), title_text, fill="black", font=font_title, anchor="mt")

    # Kappa column headers
    current_x = margin_left
    for i, kappa in enumerate(kappas):
        draw.text((current_x + col_widths[i] // 2, margin_top - 20), f"Kappa = {kappa}", fill="black", font=font_header, anchor="md")
        current_x += col_widths[i] + cell_padding

    current_y = margin_top
    for prof in profiles:
        row_center_y = current_y + (row_h // 2)
        draw.text((margin_left - 30, row_center_y), prof, fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for i, kappa in enumerate(kappas):
            if (prof, kappa) in img_cache:
                w, h, img_path = img_cache[(prof, kappa)]
                target_w = int(w * scale_factor)
                target_h = int(h * scale_factor)
                
                with Image.open(img_path) as img:
                    img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    # Simple centering (no need for Poisson contraction representation)
                    paste_y = current_y + (row_h - target_h) // 2
                    paste_x = current_x + (col_widths[i] - target_w) // 2
                    canvas.paste(img_resized, (paste_x, paste_y))
                    
            current_x += col_widths[i] + cell_padding
        current_y += row_h + 20

    canvas.save(out_filename, quality=95)
    print(f"Saved montage: {out_filename}")


def plot_exp10():
    csv_file = "exp10_compare_orientation_results.csv"
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df['FileName'] = df['Basename'].apply(os.path.basename)
    df['Profile'] = df['FileName'].str.extract(r'exp\d+_(.*?)_vf')
    df['VF'] = df['FileName'].str.extract(r'_vf([0-9\.]+)').astype(float)
    df['Seed'] = df['FileName'].str.extract(r'_seed(\d+)').astype(int)
    
    # Extract Kappa from Recipe string, fallback to filename if missing
    # Fixed regex to properly capture negative kappa values
    if 'Recipe' in df.columns:
        df['Kappa'] = df['Recipe'].str.extract(r'kappa=([0-9\.-]+)').astype(float)
    else:
        df['Kappa'] = np.nan
    df['Kappa'] = df['Kappa'].fillna(df['FileName'].str.extract(r'_K([0-9\.-]+)')[0].astype(float))
        
    df['chfem_Txx'] = pd.to_numeric(df['chfem_Txx'], errors='coerce')
    df['chfem_Tyy'] = pd.to_numeric(df['chfem_Tyy'], errors='coerce')
    df['puma_Txx'] = pd.to_numeric(df['puma_Txx'], errors='coerce')
    df['puma_Tyy'] = pd.to_numeric(df['puma_Tyy'], errors='coerce')

    df['chfem_Anisotropy'] = df['chfem_Txx'] / df['chfem_Tyy']
    df['puma_Anisotropy'] = df['puma_Txx'] / df['puma_Tyy']

    print("\n--- Generating Exp10 Montages ---")
    seeds = sorted(df['Seed'].unique())
    for seed in seeds:
        max_vf = df[df['Seed'] == seed]['VF'].max()
        create_montage_exp10(df, seed=seed, vf=max_vf, out_filename=f"exp10_orientation_montage_seed{int(seed)}_vf{max_vf}.png")

    print("\n--- Plotting Exp10 Orientation Effects ---")
    global_max_vf = df['VF'].max()
    df_stable = df[df['VF'] == global_max_vf].copy()

    # 1. Plot Anisotropy Ratio
    df_melt_aniso = df_stable.melt(id_vars=['Kappa', 'Profile'], value_vars=['chfem_Anisotropy', 'puma_Anisotropy'], var_name='Solver', value_name='Anisotropy')
    df_melt_aniso = df_melt_aniso.dropna(subset=['Anisotropy'])

    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df_melt_aniso, x='Kappa', y='Anisotropy', hue='Profile', style='Solver', markers=True, dashes=True)
    
    # Use Log scale for Y-axis, SymLog for X-axis to include 0
    plt.yscale('log')
    plt.xscale('symlog', linthresh=5)
    plt.ylim(0.5, None)
    plt.axhline(1.0, color='gray', linestyle=':', label='Isotropic (1.0)')
    plt.title(f'Exp10: Corrected Anisotropy (Txx/Tyy) vs Kappa (VF={global_max_vf})')
    plt.xlabel('Orientation Concentration (Kappa)')
    plt.ylabel('Anisotropy Ratio (Txx / Tyy) [Log Scale]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('exp10_anisotropy_plot.png', dpi=300)
    print("Saved plot: exp10_anisotropy_plot.png")

    # 2. Dual plot for absolute conductivity (Txx, Tyy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    df_chfem_melt = df_stable.melt(id_vars=['Kappa', 'Profile'], value_vars=['chfem_Txx', 'chfem_Tyy'], var_name='Direction', value_name='Conductivity').dropna(subset=['Conductivity'])
    df_puma_melt = df_stable.melt(id_vars=['Kappa', 'Profile'], value_vars=['puma_Txx', 'puma_Tyy'], var_name='Direction', value_name='Conductivity').dropna(subset=['Conductivity'])

    sns.lineplot(data=df_chfem_melt, x='Kappa', y='Conductivity', hue='Profile', style='Direction', markers=True, ax=axes[0])
    axes[0].set_yscale('log')
    axes[0].set_xscale('symlog', linthresh=5)
    axes[0].set_title(f'chfem: Conductivity vs Kappa (VF={global_max_vf})')
    axes[0].set_xlabel('Orientation Concentration (Kappa)')
    axes[0].set_ylabel('Conductivity [Log Scale]')
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    sns.lineplot(data=df_puma_melt, x='Kappa', y='Conductivity', hue='Profile', style='Direction', markers=True, ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_xscale('symlog', linthresh=5)
    axes[1].set_title(f'puma: Conductivity vs Kappa (VF={global_max_vf})')
    axes[1].set_xlabel('Orientation Concentration (Kappa)')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig('exp10_conductivity_directions_plot.png', dpi=300)
    print("Saved plot: exp10_conductivity_directions_plot.png")

if __name__ == "__main__":
    plot_exp10()
