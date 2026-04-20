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

def create_montage_exp8(df, seed, out_filename, scale_factor=2.0):
    df_seed = df[df['Seed'] == seed].copy()
    if df_seed.empty: return

    profiles = sorted(df_seed['Profile'].unique())
    vfs = sorted(df_seed['VF'].unique())
    
    row_data = {}
    max_native_h = 0
    for prof in profiles:
        subset = df_seed[df_seed['Profile'] == prof].sort_values('VF')
        imgs = []
        for _, row in subset.iterrows():
            vf = row['VF']
            # Fallback to Basename if Slice_Image column is missing or broken
            img_path = row.get('Slice_Image', f"{row['Basename']}_slice.png")
            if pd.notna(img_path) and os.path.exists(str(img_path)):
                with Image.open(str(img_path)) as img:
                    w, h = img.size
                    imgs.append((vf, img_path, w, h))
                    max_native_h = max(max_native_h, h)
        if imgs:
            row_data[prof] = imgs

    if not row_data:
        print(f"No valid slice images found for seed {seed}.")
        return

    scaled_max_h = int(max_native_h * scale_factor)
    label_h = 45
    cell_padding = 20
    row_padding = 30
    margin_top = 100
    margin_left = 350

    max_row_width = 0
    for prof, imgs in row_data.items():
        row_width = sum([int(w * scale_factor) for _, _, w, _ in imgs]) + (cell_padding * (len(imgs) - 1))
        max_row_width = max(max_row_width, row_width)

    total_w = margin_left + max_row_width + 40
    total_h = margin_top + len(row_data) * (scaled_max_h + label_h + row_padding) + 20
    
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_prop = fm.FontProperties(family=plt.rcParams['font.sans-serif'])
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        font = ImageFont.truetype(font_path, 22)
        font_header = ImageFont.truetype(font_path, 22)
        font_title = ImageFont.truetype(font_path, 34)
    except Exception as e:
        print(f"Warning: Failed to load font ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()
    
    title_text = f"Exp8: Spheres Structure (Seed {int(seed)})"
    draw.text((total_w // 2, 20), title_text, fill="black", font=font_title, anchor="mt")

    current_y = margin_top
    for prof in profiles:
        if prof not in row_data: continue
        imgs = row_data[prof]
        
        row_center_y = current_y + label_h + (scaled_max_h // 2)
        draw.text((margin_left - 30, row_center_y), prof, fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for vf, img_path, w, h in imgs:
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            draw.text((current_x + target_w // 2, current_y + label_h - 10), f"VF={vf:.2f}", fill="black", font=font_header, anchor="md")
            
            with Image.open(img_path) as img:
                img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                paste_y = current_y + label_h + (scaled_max_h - target_h)
                canvas.paste(img_resized, (current_x, paste_y))
            current_x += target_w + cell_padding
        current_y += (scaled_max_h + label_h + row_padding)

    canvas.save(out_filename, quality=95)
    print(f"Saved montage: {out_filename}")


def plot_exp8():
    csv_file = "exp8_compare_spheres_results.csv"
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df['FileName'] = df['Basename'].apply(os.path.basename)
    df['Profile'] = df['FileName'].str.extract(r'exp\d+_(.*?)_vf')
    df['VF'] = df['FileName'].str.extract(r'_vf([0-9\.]+)').astype(float)
    df['Seed'] = df['FileName'].str.extract(r'_seed(\d+)').astype(int)
    
    df['chfem_Txx'] = pd.to_numeric(df['chfem_Txx'], errors='coerce')
    df['puma_Txx'] = pd.to_numeric(df['puma_Txx'], errors='coerce')

    print("\n--- Generating Exp8 Montages ---")
    seeds = sorted(df['Seed'].unique())
    for seed in seeds:
        create_montage_exp8(df, seed=seed, out_filename=f"exp8_spheres_montage_seed{int(seed)}.png")

    print("\n--- Plotting Exp8 Conductivity ---")
    # Melt dataframe to use seaborn's powerful style mapping
    df_melt = df.melt(id_vars=['VF', 'Profile'], value_vars=['chfem_Txx', 'puma_Txx'], var_name='Solver', value_name='Conductivity')
    df_melt = df_melt.dropna(subset=['Conductivity'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='VF', y='Conductivity', hue='Profile', style='Solver', markers=True, dashes=True)
    
    plt.yscale('log')
    plt.title('Exp8: Conductivity (Txx) vs Volume Fraction (VF) for Spheres')
    plt.xlabel('Volume Fraction (VF)')
    plt.ylabel('Conductivity (Txx) [Log Scale]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('exp8_spheres_conductivity_plot.png', dpi=300)
    print("Saved plot: exp8_spheres_conductivity_plot.png")

if __name__ == "__main__":
    plot_exp8()
