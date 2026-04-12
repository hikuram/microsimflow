import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import re
from PIL import Image, ImageDraw, ImageFont

def extract_params(recipe_str):
    """
    Extract Vf, Length, and Radius from the recipe string.
    Expected format: f"rigidfiber:{vf}:length={length}:radius={radius}"
    """
    # Extract Vf (Target Volume Fraction)
    vf_match = re.search(r'rigidfiber:([0-9\.]+):', str(recipe_str))
    vf = float(vf_match.group(1)) if vf_match else None
    
    # Extract Length
    len_match = re.search(r'length=([0-9\.]+)', str(recipe_str))
    length = float(len_match.group(1)) if len_match else None
    
    # Extract Radius
    rad_match = re.search(r'radius=([0-9\.]+)', str(recipe_str))
    radius = float(rad_match.group(1)) if rad_match else None
    
    return vf, length, radius

def extract_placement_time(log_str):
    """Extract placement time in seconds from the log string."""
    match = re.search(r'\(ID:\d+\):([\d\.]+)s', str(log_str))
    return float(match.group(1)) if match else 0.0

def create_montage_exp4(df, seed, out_filename, scale_factor=1.5):
    """
    Assemble a montage from slice images for a specific seed.
    Rows: Combination of (Grid_Size, Length, Radius)
    Columns: Volume Fraction (Vf)
    Maintains relative physical scaling of the microstructure boxes.
    """
    df_seed = df[df['Seed'] == seed].copy()
    if df_seed.empty:
        return

    # Define unique row keys based on spatial scale and filler geometry
    # Sorting by Grid_Size (numerical part), then Length, then Radius
    row_keys = sorted(
        df_seed[['Grid_Size', 'Length', 'Radius']].drop_duplicates().values.tolist(),
        key=lambda x: (int(re.search(r'(\d+)', str(x[0])).group(1)), x[1], x[2])
    )
    
    row_data = {}
    for (grid, l, r) in row_keys:
        subset = df_seed[
            (df_seed['Grid_Size'] == grid) & 
            (df_seed['Length'] == l) & 
            (df_seed['Radius'] == r)
        ].sort_values('Vf')
        
        imgs = []
        max_h_row = 0
        for _, row in subset.iterrows():
            vf = row['Vf']
            img_path = f"{row['Basename']}_slice.png"
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    w, h = img.size
                    imgs.append((vf, img_path, w, h))
                    max_h_row = max(max_h_row, h)
        if imgs:
            row_data[(grid, l, r)] = (imgs, max_h_row)

    if not row_data:
        print(f"No valid slice images found for seed {seed}.")
        return

    # Layout Parameters
    label_h = 45       # Margin for Vf labels above images
    cell_padding = 20  # Horizontal space between images
    row_padding = 30   # Vertical space between rows
    margin_top = 100
    margin_left = 250  # Wide margin for detailed row labels (Grid, L, R)

    # Calculate total canvas dimensions
    max_row_width = 0
    total_h = margin_top
    for (grid, l, r), (imgs, max_native_h) in row_data.items():
        scaled_h = int(max_native_h * scale_factor)
        row_w = sum([int(w * scale_factor) for _, _, w, _ in imgs]) + (cell_padding * (len(imgs) - 1))
        max_row_width = max(max_row_width, row_w)
        total_h += (scaled_h + label_h + row_padding)

    total_w = margin_left + max_row_width + 40
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts from Matplotlib properties
    try:
        font_prop = fm.FontProperties(family=plt.rcParams.get('font.sans-serif', ['sans-serif']))
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        font = ImageFont.truetype(font_path, 24)
        font_header = ImageFont.truetype(font_path, 20)
        font_title = ImageFont.truetype(font_path, 34)
    except Exception as e:
        print(f"Warning: Failed to load font ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()
    
    # Draw Montage Title
    title_text = f"Scale and Resolution Check (Seed {seed})"
    draw.text((total_w // 2, 20), title_text, fill="black", font=font_title, anchor="mt")

    # Paste images and draw labels
    current_y = margin_top
    for (grid, l, r) in row_keys:
        if (grid, l, r) not in row_data:
            continue
            
        imgs, max_native_h = row_data[(grid, l, r)]
        scaled_max_h = int(max_native_h * scale_factor)
        
        # Row Header Label: Display Grid Size and Filler Geometry
        row_center_y = current_y + label_h + (scaled_max_h // 2)
        row_label = f"Grid: {grid}\nL={int(l)}, R={int(r)}"
        draw.multiline_text((margin_left - 30, row_center_y), row_label, 
                            fill="black", font=font, anchor="rm", align="right")
        
        current_x = margin_left
        for vf, img_path, w, h in imgs:
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Draw individual Vf labels above each microstructure slice
            label_text = f"Vf={vf*100:.1f}%"
            draw.text((current_x + target_w // 2, current_y + label_h - 10), label_text, 
                      fill="black", font=font_header, anchor="md", align="center")
            
            # Bottom-align images within the row to emphasize scale differences
            with Image.open(img_path) as img:
                img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                paste_y = current_y + label_h + (scaled_max_h - target_h)
                canvas.paste(img_resized, (current_x, paste_y))
            
            current_x += target_w + cell_padding
            
        current_y += (scaled_max_h + label_h + row_padding)

    canvas.save(out_filename, quality=95)
    print(f"Saved montage: {out_filename}")


def plot_scale_check():
    # Identify the result CSV
    csv_files = ["scale_check_results.csv", "exp4_scale_check_results.csv"]
    csv_file = next((f for f in csv_files if os.path.exists(f)), None)
    
    if not csv_file:
        print("Error: Scale check CSV not found.")
        return

    df = pd.read_csv(csv_file)

    # Parse detailed parameters from the recipe string
    params = df['Recipe'].apply(lambda x: pd.Series(extract_params(x), index=['Vf', 'Length', 'Radius']))
    df = pd.concat([df, params], axis=1)
    
    df['Placement_Time_s'] = df['Placement_Logs'].apply(extract_placement_time)
    
    # Extract Seed from Basename for grouping
    if 'Basename' in df.columns:
        df['Seed'] = pd.to_numeric(df['Basename'].str.extract(r'seed(\d+)')[0], errors='coerce')
    else:
        df['Seed'] = 0

    # Ensure necessary columns are present and clean
    df = df.dropna(subset=['Grid_Size', 'Vf', 'Length', 'Radius', 'Seed']).copy()

    # --- 1. Montage Generation per Seed ---
    print("\n--- Generating Visual Montages ---")
    seeds = sorted(df['Seed'].unique())
    for seed in seeds:
        out_name = f"exp4_montage_seed{int(seed)}.png"
        create_montage_exp4(df, seed=seed, out_filename=out_name, scale_factor=1.5)

    # --- 2. Scaling Metrics Plotting ---
    print("\n--- Plotting Performance and Physics Metrics ---")
    
    # Check if conductivity solver results are available
    has_chfem = all(col in df.columns for col in ['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz'])
    if has_chfem:
        df['Avg_Conductivity'] = df[['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz']].mean(axis=1)
    else:
        df['Avg_Conductivity'] = np.nan

    # Aggregate means across seeds for plotting
    agg_dict = {'Placement_Time_s': 'mean'}
    if has_chfem:
        agg_dict['Avg_Conductivity'] = 'mean'
        if 'chfem_Time_s' in df.columns:
            agg_dict['chfem_Time_s'] = 'mean'
            
    grouped = df.groupby(['Grid_Size', 'Length', 'Radius', 'Vf']).agg(agg_dict).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color mapping for different spatial scales (Grid_Size)
    # Using the numerical box size as the key
    unique_sizes = sorted(grouped['Grid_Size'].unique(), key=lambda x: int(re.search(r'(\d+)', x).group(1)))
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors = {size: color_list[i % len(color_list)] for i, size in enumerate(unique_sizes)}
    markers = ['o', 's', '^', 'D']

    # Subplot 1: Physics Validation (Effective Conductivity)
    ax1 = axes[0]
    if has_chfem and not grouped['Avg_Conductivity'].isna().all():
        for i, ((grid, l, r), subset) in enumerate(grouped.groupby(['Grid_Size', 'Length', 'Radius'])):
            subset = subset.sort_values('Vf')
            ax1.plot(subset['Vf'] * 100, subset['Avg_Conductivity'],
                     marker=markers[i % len(markers)], color=colors.get(grid, 'k'),
                     linestyle='-', linewidth=2, markersize=7, 
                     label=f"{grid}, L={int(l)}")
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, "Solver data missing", ha='center', va='center')
        
    ax1.set_xlabel('Volume Fraction (%)')
    ax1.set_ylabel('Effective Conductivity')
    ax1.set_title('1. Physics Consistency')
    ax1.grid(True, linestyle='--', alpha=0.7, which='both')
    ax1.legend(fontsize=9, ncol=2)

    # Subplot 2: Algorithmic Scaling (Structure Generation Time)
    ax2 = axes[1]
    for i, ((grid, l, r), subset) in enumerate(grouped.groupby(['Grid_Size', 'Length', 'Radius'])):
        subset = subset.sort_values('Vf')
        ax2.plot(subset['Vf'] * 100, subset['Placement_Time_s'],
                 marker=markers[i % len(markers)], color=colors.get(grid, 'k'),
                 linestyle='--', linewidth=2, markersize=7)
    ax2.set_xlabel('Volume Fraction (%)')
    ax2.set_ylabel('RSA Placement Time (s)')
    ax2.set_title('2. RSA Generation Cost')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Subplot 3: Numerical Scaling (Solver Execution Time)
    ax3 = axes[2]
    if 'chfem_Time_s' in grouped.columns and not grouped['chfem_Time_s'].isna().all():
        for i, ((grid, l, r), subset) in enumerate(grouped.groupby(['Grid_Size', 'Length', 'Radius'])):
            subset = subset.sort_values('Vf')
            ax3.plot(subset['Vf'] * 100, subset['chfem_Time_s'],
                     marker=markers[i % len(markers)], color=colors.get(grid, 'k'),
                     linestyle='-.', linewidth=2, markersize=7)
    else:
        ax3.text(0.5, 0.5, "Solver time data missing", ha='center', va='center')
    ax3.set_xlabel('Volume Fraction (%)')
    ax3.set_ylabel('Solver Time (s)')
    ax3.set_title('3. Solver Computation Cost')
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle("Exp 4: Comprehensive Scaling and Consistency Analysis", fontsize=16, y=1.05)
    plt.tight_layout()
    
    out_img = "exp4_scale_check_plot.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Saved statistical summary: {out_img}")
    plt.close()

if __name__ == "__main__":
    plot_scale_check()
