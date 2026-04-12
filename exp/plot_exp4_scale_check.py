import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import re
from PIL import Image, ImageDraw, ImageFont

def extract_recipe_params(recipe_str):
    """
    Extract vf, length, and radius from the recipe string:
    format: "rigidfiber:{vf}:length={length}:radius={radius}"
    """
    # Regex to capture float for vf, and integers for length/radius
    match = re.search(r'rigidfiber:([\d\.]+):length=(\d+):radius=(\d+)', str(recipe_str))
    if match:
        vf = float(match.group(1))
        length = int(match.group(2))
        radius = int(match.group(3))
        return vf, length, radius
    return None, None, None

def extract_placement_time(log_str):
    """Extract placement time in seconds from the log string."""
    match = re.search(r'\(ID:\d+\):([\d\.]+)s', str(log_str))
    if match:
        return float(match.group(1))
    return 0.0

def create_montage_exp4(df, seed, out_filename, scale_factor=1.5):
    """
    Assemble a montage from slice images for a specific seed.
    Rows: Grouped by (Grid_Size, Length, Radius)
    Columns: Sorted by Target_Vf
    Relative physical sizes are maintained using the scale_factor.
    """
    df_seed = df[df['Seed'] == seed].copy()
    if df_seed.empty:
        return

    # Define rows by unique combinations of resolution and filler geometry
    # Sorting ensures consistent row order (Box Size -> Length -> Radius)
    row_keys = sorted(
        df_seed[['Grid_Size', 'Length', 'Radius']].drop_duplicates().values.tolist(),
        key=lambda x: (x[0], x[1], x[2])
    )
    
    row_data = {}
    for (grid, l, r) in row_keys:
        subset = df_seed[
            (df_seed['Grid_Size'] == grid) & 
            (df_seed['Length'] == l) & 
            (df_seed['Radius'] == r)
        ].sort_values('Target_Vf')
        
        imgs = []
        max_h_row = 0
        for _, row in subset.iterrows():
            vf = row['Target_Vf']
            img_path = f"{row['Basename']}_slice.png"
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    w, h = img.size
                    imgs.append((vf, l, r, img_path, w, h))
                    max_h_row = max(max_h_row, h)
        if imgs:
            row_data[(grid, l, r)] = (imgs, max_h_row)

    if not row_data:
        print(f"No valid slice images found for seed {seed}.")
        return

    # Layout dimensions
    label_h = 60       # Space for column header (Vf/L/R)
    cell_padding = 20  # Horizontal gap
    row_padding = 40   # Vertical gap
    margin_top = 100
    margin_left = 250  # Left margin for (Grid/L/R) row labels

    # Calculate canvas dimensions
    max_row_width = 0
    total_h = margin_top
    for (grid, l, r), (imgs, max_native_h) in row_data.items():
        scaled_h = int(max_native_h * scale_factor)
        row_w = sum([int(w * scale_factor) for _, _, _, _, w, _ in imgs]) + (cell_padding * (len(imgs) - 1))
        max_row_width = max(max_row_width, row_w)
        total_h += (scaled_h + label_h + row_padding)

    total_w = margin_left + max_row_width + 40
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Font setup
    try:
        font_prop = fm.FontProperties(family=plt.rcParams.get('font.sans-serif', ['sans-serif']))
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        font = ImageFont.truetype(font_path, 24)
        font_header = ImageFont.truetype(font_path, 20)
        font_title = ImageFont.truetype(font_path, 34)
    except Exception as e:
        print(f"Warning: Failed to load font ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()
    
    # Title
    title_text = f"Scale Consistency Analysis (Seed {int(seed)})"
    draw.text((total_w // 2, 20), title_text, fill="black", font=font_title, anchor="mt")

    # Paste images and draw labels
    current_y = margin_top
    for (grid, l, r) in row_keys:
        if (grid, l, r) not in row_data:
            continue
            
        imgs, max_native_h = row_data[(grid, l, r)]
        scaled_max_h = int(max_native_h * scale_factor)
        
        # Row Header Label (Resolution and Geometry)
        row_center_y = current_y + label_h + (scaled_max_h // 2)
        row_label = f"Grid: {grid}\nL={l}, R={r}"
        draw.multiline_text((margin_left - 30, row_center_y), row_label, 
                            fill="black", font=font, anchor="rm", align="right")
        
        current_x = margin_left
        for vf, fl, fr, img_path, w, h in imgs:
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Column Header Label (Vf, Length, Radius)
            header_text = f"Vf={vf:.2f}\nL={fl}, R={fr}"
            draw.multiline_text((current_x + target_w // 2, current_y + label_h - 10), header_text, 
                                fill="black", font=font_header, anchor="md", align="center")
            
            # Bottom-align images within the row to emphasize volume/box size differences
            with Image.open(img_path) as img:
                img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                paste_y = current_y + label_h + (scaled_max_h - target_h)
                canvas.paste(img_resized, (current_x, paste_y))
            
            current_x += target_w + cell_padding
            
        current_y += (scaled_max_h + label_h + row_padding)

    canvas.save(out_filename, quality=95)
    print(f"Saved montage: {out_filename}")

def main():
    csv_file = "exp4_scale_check_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # Parse parameters from recipe string
    params = df['Recipe'].apply(lambda x: pd.Series(extract_recipe_params(x)))
    df[['Target_Vf', 'Length', 'Radius']] = params
    
    # RSA performance metric
    df['Placement_Time_s'] = df['Placement_Logs'].apply(extract_placement_time)
    
    # Extract Seed from Basename (e.g., ...seed1...)
    if 'Basename' in df.columns:
        df['Seed'] = pd.to_numeric(df['Basename'].str.extract(r'seed(\d+)')[0], errors='coerce')
    else:
        df['Seed'] = 1

    # Cleanup invalid data
    df = df.dropna(subset=['Grid_Size', 'Target_Vf', 'Length', 'Radius', 'Seed']).copy()

    # 1. Montage Generation (Split by Seed)
    print("\n--- Generating Montages (English Labels) ---")
    seeds = sorted(df['Seed'].unique())
    for seed in seeds:
        out_name = f"exp4_montage_seed{int(seed)}.png"
        create_montage_exp4(df, seed=seed, out_filename=out_name, scale_factor=1.5)

    # 2. Scientific Plotting
    print("\n--- Plotting Scaling Metrics ---")
    
    # Conductivity calculation (check if chfem outputs exist)
    has_chfem = all(col in df.columns for col in ['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz'])
    if has_chfem:
        df['Avg_K'] = df[['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz']].mean(axis=1)

    # Average metrics across seeds for the graph
    agg_dict = {'Placement_Time_s': 'mean'}
    if has_chfem:
        agg_dict['Avg_K'] = 'mean'
        if 'chfem_Time_s' in df.columns:
            agg_dict['chfem_Time_s'] = 'mean'
            
    grouped = df.groupby(['Grid_Size', 'Length', 'Radius', 'Target_Vf']).agg(agg_dict).reset_index()

    # Create summary plot (1x3 panel)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color/Marker settings
    unique_grids = grouped['Grid_Size'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_grids)))
    color_map = dict(zip(unique_grids, colors))

    # 1. Physics Check: Effective Conductivity
    ax1 = axes[0]
    if has_chfem:
        for (grid, l, r), subset in grouped.groupby(['Grid_Size', 'Length', 'Radius']):
            subset = subset.sort_values('Target_Vf')
            ax1.plot(subset['Target_Vf'] * 100, subset['Avg_K'],
                     marker='o', color=color_map.get(grid, 'k'),
                     linestyle='-', linewidth=2, label=f'{grid}, L={int(l)}')
        ax1.set_yscale('log')
    ax1.set_xlabel('Volume Fraction (%)')
    ax1.set_ylabel('Effective Conductivity')
    ax1.set_title('1. Homogenization Consistency')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=8)

    # 2. Algorithm Performance: RSA Scaling
    ax2 = axes[1]
    for (grid, l, r), subset in grouped.groupby(['Grid_Size', 'Length', 'Radius']):
        subset = subset.sort_values('Target_Vf')
        ax2.plot(subset['Target_Vf'] * 100, subset['Placement_Time_s'],
                 marker='s', color=color_map.get(grid, 'k'), linestyle='--')
    ax2.set_xlabel('Volume Fraction (%)')
    ax2.set_ylabel('Placement Time (s)')
    ax2.set_title('2. Structure Generation Cost')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Solver Performance: GPU/Memory Scaling
    ax3 = axes[2]
    if 'chfem_Time_s' in grouped.columns:
        for (grid, l, r), subset in grouped.groupby(['Grid_Size', 'Length', 'Radius']):
            subset = subset.sort_values('Target_Vf')
            ax3.plot(subset['Target_Vf'] * 100, subset['chfem_Time_s'],
                     marker='^', color=color_map.get(grid, 'k'), linestyle='-.')
    ax3.set_xlabel('Volume Fraction (%)')
    ax3.set_ylabel('Solver Time (s)')
    ax3.set_title('3. Solver Performance Check')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Exp 4: Resolution & Size Sensitivity Analysis", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig("exp4_scale_check_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
