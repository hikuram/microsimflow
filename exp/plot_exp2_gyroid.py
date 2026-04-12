import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.family': 'sans-serif', 
    'font.sans-serif':  ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'], 
    'axes.labelsize': 14, 
    'xtick.labelsize': 12, 
    'ytick.labelsize': 12, 
    'legend.fontsize': 12,
    'axes.linewidth': 1.5
})

def power_law(v, k0, vc, t):
    return k0 * np.maximum(v - vc, 0)**t + 1e-6

def create_montage_exp2(df, out_filename="exp2_gyroid_montage.png", scale_factor=3.0):
    """
    Assemble a single montage from slice images.
    Rows: Random Seed
    Images are packed left-to-right per Seed, with actual Vf printed above each image.
    """
    seeds = sorted(df['Seed'].unique())
    if not seeds: return

    # 1. Gather valid images and calculate row dimensions
    row_data = {}
    max_native_h = 0
    
    for s in seeds:
        subset = df[df['Seed'] == s].sort_values('Vf')
        imgs = []
        for _, row in subset.iterrows():
            vf = row['Vf']
            img_path = f"{row['Basename']}_slice.png"
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    w, h = img.size
                    imgs.append((vf, img_path, w, h))
                    max_native_h = max(max_native_h, h)
        if imgs:
            row_data[s] = imgs

    if not row_data:
        print("No valid slice images found. Skipping montage.")
        return

    # 2. Calculate canvas size and layout parameters
    scaled_max_h = int(max_native_h * scale_factor)
    
    label_h = 40       # Space reserved ABOVE the image for the Vf label
    cell_padding = 15  # Horizontal space between images
    row_padding = 20   # Vertical space between rows
    margin_top = 80
    margin_left = 160  # Wider margin for Seed labels

    # Find the maximum width across all rows to set canvas width
    max_row_width = 0
    for s, imgs in row_data.items():
        row_width = sum([int(w * scale_factor) for _, _, w, _ in imgs]) + (cell_padding * (len(imgs) - 1))
        max_row_width = max(max_row_width, row_width)

    total_w = margin_left + max_row_width + 40
    total_h = margin_top + len(row_data) * (scaled_max_h + label_h + row_padding) + 20
    
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Get the exact font path resolved by Matplotlib's rcParams
    try:
        font_prop = fm.FontProperties(family=plt.rcParams['font.sans-serif'])
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        
        font = ImageFont.truetype(font_path, 26)
        font_header = ImageFont.truetype(font_path, 22)
        font_title = ImageFont.truetype(font_path, 32)
    except Exception as e:
        print(f"Warning: Failed to load Matplotlib font for PIL ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()
    
    # Draw main title
    draw.text((total_w // 2, 20), "Microstructure Evolution (Gyroid)", fill="black", font=font_title, anchor="mt")

    # 3. Paste images
    current_y = margin_top
    for s in seeds:
        if s not in row_data:
            continue
            
        # Row Header (centered vertically relative to the image part)
        row_center_y = current_y + label_h + (scaled_max_h // 2)
        draw.text((margin_left - 20, row_center_y), f"Seed = {s}", 
                  fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for vf, img_path, w, h in row_data[s]:
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Draw Vf Label directly above the image
            vf_pct = vf * 100
            label_text = f"Vf={vf_pct:.1f}%"
            draw.text((current_x + target_w // 2, current_y + label_h - 10), label_text, 
                      fill="black", font=font_header, anchor="md", align="center")
            
            # Paste Image
            with Image.open(img_path) as img:
                img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                canvas.paste(img_resized, (current_x, current_y + label_h))
            
            current_x += target_w + cell_padding
            
        current_y += (scaled_max_h + label_h + row_padding)

    canvas.save(out_filename, quality=95)
    print(f"Saved montage: {out_filename}")


def main():
    csv_file = "exp2_gyroid_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return

    # Use pure Filler_Frac not including the interface phase
    df['Vf'] = df['Filler_Frac']
    
    # Extract Seed from Basename
    df['Seed'] = pd.to_numeric(df['Basename'].str.extract(r'seed(\d+)')[0], errors='coerce')

    # Drop rows without parameters
    df = df.dropna(subset=['Vf', 'Seed']).copy()
    df['Seed'] = df['Seed'].astype(int)

    if len(df) == 0:
        print("No valid parameter data found in CSV.")
        return

    # 1. Generate montage image
    print("\n--- Generating Montage ---")
    create_montage_exp2(df)

    # 2. Plot conductivity curves
    print("\n--- Plotting Conductivity Curves ---")
    if all(col in df.columns for col in ['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz']):
        df['K_eff'] = (df['chfem_Kxx'] + df['chfem_Kyy'] + df['chfem_Kzz']) / 3.0
    else:
        print("Warning: chfem results are missing. Plotting will be skipped.")
        return

    df_clean = df.dropna(subset=['K_eff'])
    if len(df_clean) == 0:
        print("No valid conductivity data found to plot.")
        return

    plt.figure(figsize=(8, 6))
    v_data = df_clean['Vf'].values
    k_data = df_clean['K_eff'].values

    plt.scatter(v_data * 100, k_data, color='#D55E00', s=80, edgecolor='k', label='Gyroid Data', zorder=3)

    try:
        popt, _ = curve_fit(power_law, v_data, k_data, p0=[1000.0, 0.01, 2.0], bounds=([0, 0, 1.0], [np.inf, 0.2, 4.0]))
        v_smooth = np.linspace(0, max(v_data) * 1.1, 200)
        plt.plot(v_smooth * 100, power_law(v_smooth, *popt), color='k', linestyle='--', linewidth=2.5, 
                 label=f'Fit: Vc={popt[1]*100:.2f}%, t={popt[2]:.2f}', zorder=2)
    except:
        print("Fit failed.")

    plt.xlabel('Filler Volume Fraction (%)') 
    plt.ylabel('Effective Conductivity (S/m)')
    plt.title('Double Percolation Threshold (Gyroid)')
    plt.yscale('log')
    plt.ylim(bottom=1e-5)
    plt.xlim(left=0)
    plt.grid(True, which="both", ls="--", alpha=0.5, zorder=1)
    plt.legend()
    plt.tight_layout()
    
    out_img = 'exp2_gyroid_plot.png'
    plt.savefig(out_img, dpi=300)
    print(f"Saved {out_img}")

if __name__ == "__main__":
    main()
