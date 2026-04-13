import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from PIL import Image, ImageDraw, ImageFont

# Set high-quality plotting style
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

def create_clean_montage_exp7(df, seed=11, out_filename="exp7_montage_clean.png", legend_file="common_legend.png", scale_factor=3.0):
    """
    Assemble a montage from pre-cleaned, marginless slice images.
    Strictly maintains a 1:1 physical scaling factor (X and Y).
    Column widths adapt strictly to the stretched width, avoiding artificial horizontal gaps.
    Row heights are fixed to the unstretched height, using top-blank space to visualize Poisson contraction.
    """
    df_seed = df[df['Basename'].str.contains(f"seed{seed}")].copy()
    if df_seed.empty:
        print(f"No data found for seed {seed}. Skipping montage.")
        return

    shapes = sorted(df_seed['Shape'].unique())
    stretches = sorted(df_seed['Stretch_Ratio'].unique())
    if not shapes or not stretches: return

    # 1. Get the maximum native width for each stretch ratio and the global maximum height (reference height)
    col_max_native_w = {s: 0 for s in stretches}
    max_native_h = 0
    img_cache = {}

    for stretch in stretches:
        for shape in shapes:
            cell_data = df_seed[(df_seed['Shape'] == shape) & (df_seed['Stretch_Ratio'] == stretch)]
            if not cell_data.empty:
                img_path = cell_data.iloc[0]['Slice_Image']
                if pd.notna(img_path) and os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        w, h = img.size
                        img_cache[(shape, stretch)] = (w, h, img_path)
                        col_max_native_w[stretch] = max(col_max_native_w[stretch], w)
                        max_native_h = max(max_native_h, h)

    if max_native_h == 0: return

    # 2. Calculate canvas size and layout parameters
    row_h = int(max_native_h * scale_factor) # Row height is fixed to the maximum height (unstretched state)
    col_widths = [int(col_max_native_w[s] * scale_factor) for s in stretches] # Column widths tightly fit the stretched width

    cell_padding = 15
    margin_top = 100
    margin_left = 160
    legend_pad = 30

    # Get the exact font path resolved by Matplotlib's rcParams
    try:
        # Create properties based on the current Matplotlib sans-serif setting
        font_prop = fm.FontProperties(family=plt.rcParams['font.sans-serif'])
        # Ask Matplotlib to find the absolute path to this font file
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        
        font = ImageFont.truetype(font_path, 26)
        font_header = ImageFont.truetype(font_path, 26)
        font_title = ImageFont.truetype(font_path, 28)
    except Exception as e:
        print(f"Warning: Failed to load Matplotlib font for PIL ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()

    try:
        with Image.open(legend_file) as leg:
            legend_w, legend_h = leg.size
            legend_img = leg.copy()
    except FileNotFoundError:
        legend_w, legend_h, legend_img = 0, 0, None

    total_w = margin_left + sum(col_widths) + (cell_padding * (len(stretches) - 1)) + legend_pad + legend_w + 20
    total_h = margin_top + (row_h * len(shapes)) + (cell_padding * (len(shapes) - 1)) + 20
    
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Draw main title
    draw.text((total_w // 2, 20), "Microstructure Deformation by Filler Shape", 
              fill="black", font=font_title, anchor="mt")

    # 3. Paste images (bottom-aligned, no wasted horizontal space)
    current_y = margin_top
    for r_idx, shape in enumerate(shapes):
        # Row Header
        draw.text((margin_left - 20, current_y + row_h // 2), shape, 
                  fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for c_idx, stretch in enumerate(stretches):
            col_w = col_widths[c_idx]
            native_w, native_h = col_max_native_w[stretch], max_native_h
            
            # Column Header
            if r_idx == 0 and native_w > 0:
                header_text = f"Stretch = {stretch} ({native_w}x{native_h})"
                draw.multiline_text((current_x + col_w // 2, margin_top - 15), header_text, 
                                    fill="black", font=font_header, anchor="md", align="center")
            
            if (shape, stretch) in img_cache:
                w, h, img_path = img_cache[(shape, stretch)]
                with Image.open(img_path) as img:
                    # Resize both dimensions using the exact same scale factor
                    target_w = int(w * scale_factor)
                    target_h = int(h * scale_factor)
                    img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    
                    # X-axis: Center within the column width (tightly fitted)
                    paste_x = current_x + (col_w - target_w) // 2
                    # Y-axis: Bottom-align to visualize Poisson contraction as blank space at the top
                    paste_y = current_y + (row_h - target_h)
                    
                    canvas.paste(img_resized, (paste_x, paste_y))
            
            current_x += col_w + cell_padding
        current_y += row_h + cell_padding

    # 4. Paste the common legend
    if legend_img:
        legend_x = margin_left + sum(col_widths) + (cell_padding * (len(stretches) - 1)) + legend_pad
        legend_y = margin_top + (current_y - margin_top) // 2 - legend_h // 2
        canvas.paste(legend_img, (legend_x, max(margin_top, legend_y)))

    canvas.save(out_filename, quality=95)
    print(f"Saved physically scaled deformation montage: {out_filename}")

def main():
    csv_file = "exp7_shape_stretch_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df['Shape'] = df['Recipe'].apply(extract_shape_from_recipe)
    
    # Filter valid rows with simulation results
    df_clean = df.dropna(subset=['Stretch_Ratio', 'chfem_Txx']).copy()

    # Calculate Relative Conductivity (K / K_0)
    df_clean['Txx_rel'] = np.nan
    df_clean['Tyy_rel'] = np.nan
    df_clean['Tzz_rel'] = np.nan

    groups = df_clean.groupby(['Shape', df_clean['Basename'].str.extract(r'(seed\d+)', expand=False)])
    
    for name, group in groups:
        baseline = group[group['Stretch_Ratio'] == 1.0]
        if baseline.empty: continue
            
        kxx_0 = baseline['chfem_Txx'].values[0]
        kyy_0 = baseline['chfem_Tyy'].values[0]
        kzz_0 = baseline['chfem_Tzz'].values[0]
        
        df_clean.loc[group.index, 'Txx_rel'] = group['chfem_Txx'] / kxx_0
        df_clean.loc[group.index, 'Tyy_rel'] = group['chfem_Tyy'] / kyy_0
        df_clean.loc[group.index, 'Tzz_rel'] = group['chfem_Tzz'] / kzz_0

    # --- Plotting Part ---
    
    # 1. Absolute Txx Plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_clean, x='Stretch_Ratio', y='chfem_Txx', hue='Shape', style='Shape',
        markers=True, dashes=False, linewidth=2.5, markersize=9, palette="colorblind"
    )
    plt.title('Absolute Conductivity ($K_{xx}$) vs. Stretch Ratio', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(r'Stretch Ratio ($\lambda$)')
    plt.ylabel(r'Conductivity ($K_{xx}$)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("exp7_absolute_Txx.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Relative K Plots (X, Y, Z directions)
    directions = [
        ('X-direction (Stretch Axis)', 'Txx_rel'),
        ('Y-direction (Transverse)', 'Tyy_rel'),
        ('Z-direction (Transverse)', 'Tzz_rel')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle('Relative Conductivity ($K / K_0$) by Filler Shape', fontsize=16, fontweight='bold', y=1.05)
    for ax, (title, col_rel) in zip(axes, directions):
        sns.lineplot(
            data=df_clean, x='Stretch_Ratio', y=col_rel, hue='Shape', style='Shape',
            markers=True, dashes=False, linewidth=2.5, markersize=9, palette="colorblind", ax=ax
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r'Stretch Ratio ($\lambda$)')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if ax != axes[2]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else: ax.legend(title='Filler Geometry', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig("exp7_relative_K.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Montage Generation ---
    # Extract all unique seed numbers from the original unfiltered dataframe (df).
    # This ensures montages are generated even if the solver was skipped (--solver skip).
    seeds_in_data = df['Basename'].str.extract(r'seed(\d+)')[0].dropna().unique()
    
    if len(seeds_in_data) > 0:
        for s in seeds_in_data:
            seed_num = int(s)
            out_name = f"exp7_montage_seed{seed_num}_clean.png"
            # Pass the original 'df' instead of 'df_clean' to guarantee all slice images are found
            create_clean_montage_exp7(df, seed=seed_num, out_filename=out_name)
    else:
        print("Warning: Could not extract seed numbers from Basename. Skipping montage generation.")

if __name__ == "__main__":
    main()
