import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

# Set publication-quality font and style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
})

def create_montage_exp6(df, seed, out_filename="exp6_filler_montage.png", scale_factor=2.0):
    """
    Assemble a montage from slice images for a specific seed.
    Rows: Filler Type
    Images are packed left-to-right by Filler Fraction (Vf), with actual Vf printed above.
    """
    df_seed = df[df['Seed'] == seed].copy()
    if df_seed.empty:
        return

    filler_types = sorted(df_seed['Filler_Type'].unique())
    if not filler_types:
        return

    # 1. Gather valid images and calculate row dimensions
    row_data = {}
    max_native_h = 0
    
    for ftype in filler_types:
        subset = df_seed[df_seed['Filler_Type'] == ftype].sort_values('Filler_Frac')
        imgs = []
        for _, row in subset.iterrows():
            vf = row['Filler_Frac']
            img_path = f"{row['Basename']}_slice.png"
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    w, h = img.size
                    imgs.append((vf, img_path, w, h))
                    max_native_h = max(max_native_h, h)
        if imgs:
            row_data[ftype] = imgs

    if not row_data:
        print(f"No valid slice images found for seed {seed}.")
        return

    # 2. Calculate canvas size and layout parameters
    scaled_max_h = int(max_native_h * scale_factor)
    
    label_h = 45       # Space reserved ABOVE the image for the Vf label
    cell_padding = 20  # Horizontal space between images
    row_padding = 30   # Vertical space between rows
    margin_top = 100
    margin_left = 200  # Wide margin for Filler Type labels

    # Find the maximum width across all rows to set canvas width
    max_row_width = 0
    for ftype, imgs in row_data.items():
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
        font_title = ImageFont.truetype(font_path, 34)
    except Exception as e:
        print(f"Warning: Failed to load font ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()
    
    # Draw main title
    title_text = f"Filler Shape Reinforcement (Seed {int(seed)})"
    draw.text((total_w // 2, 20), title_text, fill="black", font=font_title, anchor="mt")

    # 3. Paste images
    current_y = margin_top
    for ftype in filler_types:
        if ftype not in row_data:
            continue
            
        imgs = row_data[ftype]
        
        # Row Header (Centered vertically relative to the image part)
        row_center_y = current_y + label_h + (scaled_max_h // 2)
        draw.text((margin_left - 30, row_center_y), ftype, 
                  fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for vf, img_path, w, h in imgs:
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Draw Vf Label directly above the image
            vf_pct = vf * 100
            label_text = f"Vf={vf_pct:.1f}%"
            draw.text((current_x + target_w // 2, current_y + label_h - 10), label_text, 
                      fill="black", font=font_header, anchor="md", align="center")
            
            # Paste Image (bottom-aligned if there are native height differences)
            with Image.open(img_path) as img:
                img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                paste_y = current_y + label_h + (scaled_max_h - target_h)
                canvas.paste(img_resized, (current_x, paste_y))
            
            current_x += target_w + cell_padding
            
        current_y += (scaled_max_h + label_h + row_padding)

    canvas.save(out_filename, quality=95)
    print(f"Saved visual montage: {out_filename}")


def plot_exp6_filler():
    # Identify the result CSV
    csv_file = "exp6_filler_results.csv"
    if not os.path.exists(csv_file):
        csv_file = "exp6_filler.csv"
        
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # Extract filler type from the Recipe string
    df['Filler_Type'] = df['Recipe'].apply(lambda x: str(x).split(':')[0].capitalize())
    
    # Extract Seed from Basename
    if 'Basename' in df.columns:
        df['Seed'] = pd.to_numeric(df['Basename'].str.extract(r'seed(\d+)')[0], errors='coerce')
    else:
        df['Seed'] = 1

    # Ensure necessary columns are present
    df = df.dropna(subset=['Filler_Type', 'Filler_Frac', 'Seed']).copy()

    if df.empty:
        print("Dataframe is empty after parsing.")
        return

    # --- 1. Generate Visual Montage per Seed ---
    print("\n--- Generating Montages ---")
    seeds = sorted(df['Seed'].unique())
    for seed in seeds:
        out_name = f"exp6_montage_seed{int(seed)}.png"
        create_montage_exp6(df, seed=seed, out_filename=out_name, scale_factor=2.0)

    # --- 2. Plotting Physics Validation ---
    print("\n--- Plotting Stiffness Comparison ---")
    
    # Check if conductivity/stiffness solver results are available
    has_chfem = all(col in df.columns for col in ['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz'])
    
    if not has_chfem:
        print("Warning: Solver results are missing. Plotting skipped.")
        return

    # Calculate average stiffness across XYZ directions (random orientation assumes isotropy)
    df['Avg_Stiffness'] = df[['chfem_Kxx', "chfem_Kyy", "chfem_Kzz"]].mean(axis=1)
    
    # Clean NaNs before plotting
    df_plot = df.dropna(subset=['Avg_Stiffness'])
    if df_plot.empty:
        print("No valid stiffness data to plot.")
        return

    grouped = df_plot.groupby(['Filler_Type', 'Filler_Frac'])['Avg_Stiffness'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    
    markers = {'Sphere': 'o', 'Flake': 's', 'Rigidfiber': '^'}
    
    for filler_type in grouped['Filler_Type'].unique():
        subset = grouped[grouped['Filler_Type'] == filler_type].sort_values('Filler_Frac')
        ax.plot(subset['Filler_Frac'] * 100, subset['Avg_Stiffness'], 
                marker=markers.get(filler_type, 'x'), label=filler_type, markersize=8, linewidth=2)

    ax.set_xlabel("Filler Volume Fraction (%)")
    ax.set_ylabel("Average Effective Stiffness")
    ax.set_title("Tutorial Part 2: Filler Shape Reinforcement Effect\n(Random Orientation)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_img = "exp6_filler_plot.png"
    plt.savefig(out_img, dpi=300)
    print(f"Saved statistical plot: {out_img}")
    plt.close()

if __name__ == "__main__":
    plot_exp6_filler()
