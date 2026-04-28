import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

def create_montage_exp5(df, out_filename="exp5_morphology_montage.png", scale_factor=2.0):
    """
    Assemble a single montage from slice images representing different morphologies.
    Rows: Background Type (BG_Type)
    Columns: Polymer A Fraction (Vf_A)
    Since there is no seed splitting, it picks the first available image for each condition.
    """
    # Use rounded PolymerA_Frac for clean column grouping
    df_clean = df.copy()
    df_clean['Vf_A'] = df_clean['PolymerA_Frac'].round(2)
    
    bg_types = sorted(df_clean['BG_Type'].dropna().unique())
    if not bg_types:
        return

    # 1. Gather representative images and calculate row dimensions
    row_data = {}
    max_native_h = 0
    
    for bg in bg_types:
        # Get all unique volume fractions for this morphology
        subset = df_clean[df_clean['BG_Type'] == bg].sort_values('Vf_A')
        
        # Group by Vf_A to ensure we only pick one representative image per fraction
        imgs = []
        for vf, group in subset.groupby('Vf_A'):
            for _, row in group.iterrows():
                img_path = f"{row['Basename']}_slice.png"
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        w, h = img.size
                        imgs.append((vf, img_path, w, h))
                        max_native_h = max(max_native_h, h)
                    break # Stop after finding the first valid image for this Vf_A
                    
        if imgs:
            row_data[bg] = imgs

    if not row_data:
        print("No valid slice images found. Skipping montage.")
        return

    # 2. Calculate layout parameters
    scaled_max_h = int(max_native_h * scale_factor)
    
    label_h = 45       # Margin for Vf labels above images
    cell_padding = 20  # Horizontal space between images
    row_padding = 30   # Vertical space between rows
    margin_top = 100
    margin_left = 220  # Wide margin for morphology labels

    # Calculate total canvas width and height
    max_row_width = 0
    total_h = margin_top
    for bg, imgs in row_data.items():
        row_w = sum([int(w * scale_factor) for _, _, w, _ in imgs]) + (cell_padding * (len(imgs) - 1))
        max_row_width = max(max_row_width, row_w)
        total_h += (scaled_max_h + label_h + row_padding)

    total_w = margin_left + max_row_width + 40
    
    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # 3. Load Matplotlib fonts for PIL
    try:
        font_prop = fm.FontProperties(family=plt.rcParams.get('font.sans-serif', ['sans-serif']))
        font_path = fm.findfont(font_prop, fallback_to_default=True)
        font = ImageFont.truetype(font_path, 26)
        font_header = ImageFont.truetype(font_path, 22)
        font_title = ImageFont.truetype(font_path, 34)
    except Exception as e:
        print(f"Warning: Failed to load font ({e}). Using default.")
        font = font_header = font_title = ImageFont.load_default()
    
    # Draw Main Title
    title_text = "Phase Morphology Evolution"
    draw.text((total_w // 2, 20), title_text, fill="black", font=font_title, anchor="mt")

    # 4. Paste images and draw labels
    current_y = margin_top
    for bg in bg_types:
        if bg not in row_data:
            continue
            
        imgs = row_data[bg]
        
        # Draw Row Header (Morphology Name)
        row_center_y = current_y + label_h + (scaled_max_h // 2)
        draw.text((margin_left - 30, row_center_y), bg.capitalize(), 
                  fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for vf, img_path, w, h in imgs:
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Draw Vf Label directly above the image
            vf_pct = vf * 100
            label_text = f"Vf_A={vf_pct:.0f}%"
            draw.text((current_x + target_w // 2, current_y + label_h - 10), label_text, 
                      fill="black", font=font_header, anchor="md", align="center")
            
            # Paste resized image (bottom-aligned if heights differ)
            with Image.open(img_path) as img:
                img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                paste_y = current_y + label_h + (scaled_max_h - target_h)
                canvas.paste(img_resized, (current_x, paste_y))
            
            current_x += target_w + cell_padding
            
        current_y += (scaled_max_h + label_h + row_padding)

    canvas.save(out_filename, quality=95)
    print(f"Saved visual montage: {out_filename}")


def plot_exp5_morphology():
    csv_file = "exp5_morphology_results.csv"
    # Fallback to older name if _results is missing
    if not os.path.exists(csv_file):
        csv_file = "exp5_morphology.csv"
        
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # Target volume fraction parsing for clean grouping
    df['Vf_A'] = df['PolymerA_Frac'].round(2)
    
    # Ensure dataset is clean before plotting
    if df.empty:
        print("Dataframe is empty.")
        return

    # --- 1. Generate Visual Montage ---
    print("\n--- Generating Phase Morphology Montage ---")
    create_montage_exp5(df, out_filename="exp5_morphology_montage.png", scale_factor=2.0)
    
    # --- 2. Plotting Physics Validation ---
    print("\n--- Plotting Stiffness Comparison ---")
    
    # Verify solver columns exist
    if not all(col in df.columns for col in ['chfem_Txx', 'chfem_Tyy', 'chfem_Tzz']):
        print("Warning: Solver results (chfem_Txx, etc.) are missing. Plotting skipped.")
        return
        
    # Group by background type and volume fraction
    grouped = df.dropna(subset=['chfem_Txx', 'chfem_Tyy', 'chfem_Tzz']).groupby(['BG_Type', 'Vf_A'])[['chfem_Txx', 'chfem_Tyy', 'chfem_Tzz']].mean().reset_index()
    
    # Theoretical Limits (Rule of Mixtures)
    E_a, E_b = 1.0, 100.0
    
    # Create smooth theoretical curves from Vf_A = 0.0 to 1.0
    v_a_smooth = np.linspace(0, 1.0, 100)
    v_b_smooth = 1.0 - v_a_smooth
    E_voigt = v_a_smooth * E_a + v_b_smooth * E_b
    E_reuss = (E_a * E_b) / (v_a_smooth * E_b + v_b_smooth * E_a)

    # Setup 1x3 subplots for X, Y, Z directions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    directions = [('chfem_Txx', 'X-Direction'),
                  ('chfem_Tyy', 'Y-Direction'),
                  ('chfem_Tzz', 'Z-Direction')]
                  
    markers = {'lamellar': 's', 'cylinder': '^', 'gyroid': 'o', 'sea_island': 'D'}

    for i, (col, title) in enumerate(directions):
        ax = axes[i]
        
        # Plot theoretical bounds and fill area between them
        ax.plot(v_a_smooth, E_voigt, 'k--', label='Voigt (Upper Bound)', linewidth=1.5)
        ax.plot(v_a_smooth, E_reuss, 'k:', label='Reuss (Lower Bound)', linewidth=1.5)
        ax.fill_between(v_a_smooth, E_reuss, E_voigt, color='gray', alpha=0.1)

        # Plot simulation results for each morphology
        for bg in grouped['BG_Type'].unique():
            subset = grouped[grouped['BG_Type'] == bg].sort_values('Vf_A')
            ax.plot(subset['Vf_A'], subset[col], marker=markers.get(bg, 'x'), 
                    label=bg.capitalize(), markersize=8, linewidth=2)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Volume Fraction of Soft Phase A ($V_A$)", fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 110)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.set_ylabel("Effective Stiffness [MPa assumed]", fontsize=12)
        if i == 2:
            # Place legend outside the last plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=11)

    plt.suptitle(
       "Exp 5: Morphology Anisotropy vs. Rule of Mixtures\n"
       "(Polymer A: E=1.0, nu=0.35; Polymer B: E=100.0, nu=0.30)",
       fontsize=16,
       y=1.05,
    )
    
    # Save the figure
    out_img = "exp5_morphology_plot.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Saved statistical plot: {out_img}")
    plt.close()

if __name__ == "__main__":
    plot_exp5_morphology()
