import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

plt.rcParams.update({
    'font.family': 'sans-serif', 
    'font.sans-serif':  ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'], 
    'axes.labelsize': 14, 
    'xtick.labelsize': 12, 
    'ytick.labelsize': 12, 
    'legend.fontsize': 12,
    'axes.linewidth': 1.5
})

def create_montage_exp3(df, out_filename="exp3_hybrid_montage.png", scale_factor=3.0):
    """
    Assemble a single montage from slice images.
    Rows: Random Seed
    Cols: Flake Volume Fraction (v_flk)
    """
    seeds = sorted(df['Seed'].unique())
    v_flks = sorted(df['v_flk'].unique())
    
    if not seeds or not v_flks: return

    # 1. Get the max native width/height to establish the grid
    img_cache = {}
    max_native_w = 0
    max_native_h = 0

    for s in seeds:
        for v in v_flks:
            # np.isclose or exact float match (assuming exact match works if extracted properly)
            cell_data = df[(df['Seed'] == s) & (np.isclose(df['v_flk'], v))]
            if not cell_data.empty:
                basename = cell_data.iloc[0]['Basename']
                img_path = f"{basename}_slice.png"
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        w, h = img.size
                        img_cache[(s, v)] = (w, h, img_path)
                        max_native_w = max(max_native_w, w)
                        max_native_h = max(max_native_h, h)

    if max_native_h == 0 or max_native_w == 0: 
        print("No valid slice images found. Skipping montage.")
        return

    # 2. Calculate canvas size and layout parameters
    row_h = int(max_native_h * scale_factor)
    col_w = int(max_native_w * scale_factor)

    cell_padding = 15
    margin_top = 80
    margin_left = 160

    try:
        font_prop = fm.FontProperties(family=plt.rcParams['font.sans-serif'])
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        
        font = ImageFont.truetype(font_path, 26)
        font_header = ImageFont.truetype(font_path, 22)
        font_title = ImageFont.truetype(font_path, 32)
    except Exception as e:
        print(f"Warning: Failed to load Matplotlib font for PIL ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()

    total_w = margin_left + (col_w * len(v_flks)) + (cell_padding * (len(v_flks) - 1)) + 30
    total_h = margin_top + (row_h * len(seeds)) + (cell_padding * (len(seeds) - 1)) + 30
    
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Draw main title
    draw.text((total_w // 2, 20), "Microstructure Evolution (Hybrid Filler)", fill="black", font=font_title, anchor="mt")

    # 3. Paste images
    current_y = margin_top
    for r_idx, s in enumerate(seeds):
        # Row Header
        draw.text((margin_left - 20, current_y + row_h // 2), f"Seed = {s}", 
                  fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for c_idx, v in enumerate(v_flks):
            # Column Header
            if r_idx == 0:
                header_text = f"Flake={v*100:.1f}%\nFiber={(0.08 - v)*100:.1f}%"
                draw.multiline_text((current_x + col_w // 2, margin_top - 15), header_text, 
                                    fill="black", font=font_header, anchor="md", align="center")
            
            # Find the closest matching v_flk in the cache
            closest_v = min([k[1] for k in img_cache.keys() if k[0] == s], key=lambda x: abs(x - v), default=None)

            if closest_v is not None and abs(closest_v - v) < 1e-4:
                w, h, img_path = img_cache[(s, closest_v)]
                with Image.open(img_path) as img:
                    target_w = int(w * scale_factor)
                    target_h = int(h * scale_factor)
                    img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    canvas.paste(img_resized, (current_x, current_y))
            else:
                # Blank placeholder if missing
                draw.rectangle([current_x, current_y, current_x + col_w, current_y + row_h], 
                               outline="gray", fill="#f0f0f0")
                draw.text((current_x + col_w//2, current_y + row_h//2), "N/A", 
                          fill="gray", font=font_header, anchor="mm")
            
            current_x += col_w + cell_padding
        current_y += row_h + cell_padding

    canvas.save(out_filename, quality=95)
    print(f"Saved montage: {out_filename}")

def main():
    csv_file = "exp3_hybrid_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return

    # Extract the volume fraction of flakes and seed
    df['v_flk'] = pd.to_numeric(df['Recipe'].str.extract(r'flake:([0-9\.]+)')[0], errors='coerce')
    df['Seed'] = pd.to_numeric(df['Basename'].str.extract(r'seed(\d+)')[0], errors='coerce')

    # Drop rows without parameters
    df = df.dropna(subset=['v_flk', 'Seed']).copy()
    df['Seed'] = df['Seed'].astype(int)

    if len(df) == 0:
        print("No valid parameter data found in CSV.")
        return

    # 1. Generate montage image
    print("\n--- Generating Montage ---")
    create_montage_exp3(df)

    # 2. Plot conductivity curves
    print("\n--- Plotting Conductivity Curves ---")
    if all(col in df.columns for col in ['chfem_Txx', 'chfem_Tyy', 'chfem_Tzz']):
        df['K_eff'] = (df['chfem_Txx'] + df['chfem_Tyy'] + df['chfem_Tzz']) / 3.0
    else:
        print("Warning: chfem results are missing. Plotting will be skipped.")
        return

    df_clean = df.dropna(subset=['K_eff']).sort_values('v_flk')
    if len(df_clean) == 0:
        print("No valid conductivity data found to plot.")
        return

    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(df_clean['v_flk'] * 100, df_clean['K_eff'], color='#009E73', alpha=0.6, label='Individual runs')

    # Connect the mean values with a line
    mean_df = df_clean.groupby('v_flk')['K_eff'].mean().reset_index()
    plt.plot(mean_df['v_flk'] * 100, mean_df['K_eff'], color='#000000', marker='o', markersize=8, linewidth=2.5, label='Mean K_eff (Total Vf=8%)')

    plt.xlabel('Flake Volume Fraction (%)')
    plt.ylabel('Effective Conductivity (S/m)')
    plt.title('Synergistic Effect of Fiber/Flake Hybrid')
    plt.yscale('log')
    plt.ylim(bottom=1e-5)
    
    # Adding a secondary axis (Twin Axis) at the top of the X-axis
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels([f"{8.0 - x:.1f}" for x in ax1.get_xticks()])
    ax2.set_xlabel('Fiber Volume Fraction (%)')

    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='lower center')
    plt.tight_layout()
    
    out_img = 'exp3_hybrid_plot.png'
    plt.savefig(out_img, dpi=300)
    print(f"Saved {out_img}")

if __name__ == "__main__":
    main()
