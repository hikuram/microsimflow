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

def create_montage_exp1(df, out_filename="exp1_agglom_montage.png", scale_factor=3.0):
    """
    Assemble a single montage from slice images.
    Rows: Random Seed
    Cols: Degree of Agglomeration (n)
    """
    seeds = sorted(df['Seed'].unique())
    ns = sorted(df['n'].unique())
    
    if not seeds or not ns: return

    # 1. Get the max native width/height to establish the grid
    img_cache = {}
    max_native_w = 0
    max_native_h = 0

    for s in seeds:
        for n in ns:
            cell_data = df[(df['Seed'] == s) & (df['n'] == n)]
            if not cell_data.empty:
                basename = cell_data.iloc[0]['Basename']
                img_path = f"{basename}_slice.png"
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        w, h = img.size
                        img_cache[(s, n)] = (w, h, img_path)
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

    total_w = margin_left + (col_w * len(ns)) + (cell_padding * (len(ns) - 1)) + 30
    total_h = margin_top + (row_h * len(seeds)) + (cell_padding * (len(seeds) - 1)) + 30
    
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Draw main title
    draw.text((total_w // 2, 20), "Microstructure by Agglomeration Level", fill="black", font=font_title, anchor="mt")

    # 3. Paste images
    current_y = margin_top
    for r_idx, s in enumerate(seeds):
        # Row Header
        draw.text((margin_left - 20, current_y + row_h // 2), f"Seed = {s}", 
                  fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for c_idx, n in enumerate(ns):
            # Column Header
            if r_idx == 0:
                header_text = f"num_fibers={n}"
                draw.text((current_x + col_w // 2, margin_top - 15), header_text, 
                          fill="black", font=font_header, anchor="md", align="center")
            
            if (s, n) in img_cache:
                w, h, img_path = img_cache[(s, n)]
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
    csv_file = "exp1_agglom_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return

    # Extract num_fibers and seed
    df['n'] = pd.to_numeric(df['Recipe'].str.extract(r'num_fibers=(\d+)')[0], errors='coerce')
    df['Seed'] = pd.to_numeric(df['Basename'].str.extract(r'seed(\d+)')[0], errors='coerce')
    
    # Drop rows missing base parameters
    df = df.dropna(subset=['n', 'Seed']).copy()
    df['n'] = df['n'].astype(int)
    df['Seed'] = df['Seed'].astype(int)

    if len(df) == 0:
        print("No valid parameter data found in CSV.")
        return

    # 1. Generate montage image (Runs even if solver failed)
    print("\n--- Generating Montage ---")
    create_montage_exp1(df, out_filename="exp1_agglom_montage.png")

    # 2. Plot conductivity curves (Strictly requires K_eff calculation results)
    print("\n--- Plotting Conductivity Curves ---")
    
    if all(col in df.columns for col in ['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz']):
        df['K_eff'] = (df['chfem_Kxx'] + df['chfem_Kyy'] + df['chfem_Kzz']) / 3.0
    else:
        print("Warning: chfem results are missing. Plotting will be skipped.")
        return

    df_clean = df.dropna(subset=['K_eff']).sort_values('n')

    if len(df_clean) == 0:
        print("No valid conductivity data found to plot (df_clean is empty).")
        return

    plt.figure(figsize=(8, 6))
    # Plot the variance for each seed as a scatter plot
    plt.scatter(df_clean['n'], df_clean['K_eff'], color='#D55E00', alpha=0.6, label='Individual runs')

    # Connect the mean values with a line
    mean_df = df_clean.groupby('n')['K_eff'].mean().reset_index()
    plt.plot(mean_df['n'], mean_df['K_eff'], color='#0072B2', marker='D', markersize=8, linewidth=2.5, label='Mean K_eff (Total Vf=8%)')

    plt.xlabel('Degree of Agglomeration (Fibers per Cluster, n)')
    plt.ylabel('Effective Conductivity (S/m)')
    plt.title('Network Breakdown by Agglomeration')
    plt.yscale('log')
    plt.ylim(bottom=1e-5)
    
    ns_unique = sorted(df_clean['n'].unique())
    plt.xticks(ns_unique)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_img = 'exp1_agglom_plot.png'
    plt.savefig(out_img, dpi=300)
    print(f"Saved {out_img}")

if __name__ == "__main__":
    main()
