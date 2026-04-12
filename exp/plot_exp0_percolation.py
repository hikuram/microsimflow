import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import curve_fit

# Publication-quality font and style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
})

def percolation_power_law(v, k0, vc, t):
    """
    Percolation theory model: K = K0 * (Vf - Vc)^t  (for Vf > Vc)
    Adds a minute value 1e-6 for log-scale plotting when Vf falls below Vc.
    """
    return k0 * np.maximum(v - vc, 0)**t + 1e-6

def create_clean_montage_exp0(df, seed=1, out_filename="exp0_montage_clean.png", scale_factor=3.0):
    """
    Assemble a montage from pre-cleaned, marginless slice images.
    Images are packed left-to-right per Radius, with actual Vf printed above each image.
    """
    df_seed = df[df['Basename'].str.contains(f"seed{seed}")].copy()
    if df_seed.empty:
        return

    radii = sorted(df_seed['Radius'].unique())
    if not radii: return

    # 1. Gather valid images and calculate row dimensions
    row_data = {}
    max_native_h = 0
    
    for r in radii:
        subset = df_seed[df_seed['Radius'] == r].sort_values('Total_Vf')
        imgs = []
        for _, row in subset.iterrows():
            vf = row['Total_Vf']
            img_path = f"{row['Basename']}_slice.png"
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    w, h = img.size
                    imgs.append((vf, img_path, w, h))
                    max_native_h = max(max_native_h, h)
        if imgs:
            row_data[r] = imgs

    if not row_data:
        print(f"No valid slice images found for seed {seed}.")
        return

    # 2. Calculate canvas size and layout parameters
    scaled_max_h = int(max_native_h * scale_factor)
    
    label_h = 40       # Space reserved ABOVE the image for the Vf label
    cell_padding = 15  # Horizontal space between images
    row_padding = 20   # Vertical space between rows
    margin_top = 80
    margin_left = 180  # Wider margin for Radius labels

    # Find the maximum width across all rows to set canvas width
    max_row_width = 0
    for r, imgs in row_data.items():
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
    title_text = f"Microstructure Evolution (Seed {seed})"
    draw.text((total_w // 2, 20), title_text, fill="black", font=font_title, anchor="mt")

    # 3. Paste images
    current_y = margin_top
    for r in radii:
        if r not in row_data:
            continue
            
        # Row Header (centered vertically relative to the image part)
        row_center_y = current_y + label_h + (scaled_max_h // 2)
        draw.text((margin_left - 20, row_center_y), f"Radius = {r}", 
                  fill="black", font=font, anchor="rm")
        
        current_x = margin_left
        for vf, img_path, w, h in row_data[r]:
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
    print(f"Saved cleanly formatted montage: {out_filename}")


def main():
    csv_file = "exp0_percolation_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run exp0 first.")
        return

    # Extract filler fraction and radius from the Recipe string
    df['Total_Vf'] = df['Filler_Frac']
    df['Radius'] = pd.to_numeric(df['Recipe'].str.extract(r'radius=(\d+)')[0], errors='coerce')
    
    # Drop rows where we couldn't at least parse the base parameters
    df = df.dropna(subset=['Total_Vf', 'Radius']).copy()
    df['Radius'] = df['Radius'].astype(int)

    if len(df) == 0:
        print("No valid parameter data found in CSV.")
        return

    # 1. Generate montage images (Runs even if solver failed and K_eff is missing)
    print("\n--- Generating Montages ---")
    seeds_in_data = df['Basename'].str.extract(r'seed(\d+)')[0].dropna().unique()
    
    if len(seeds_in_data) > 0:
        for s in seeds_in_data:
            seed_num = int(s)
            out_name = f"exp0_montage_seed{seed_num}_clean.png"
            create_clean_montage_exp0(df, seed=seed_num, out_filename=out_name)
    else:
        print("Warning: Could not extract seed numbers from Basename. Skipping montage generation.")

    # 2. Plot percolation curves (Strictly requires K_eff calculation results)
    print("\n--- Plotting Percolation Curves ---")
    
    # Calculate Macroscopic electrical conductivity
    if all(col in df.columns for col in ['chfem_Kxx', 'chfem_Kyy', 'chfem_Kzz']):
        df['K_eff'] = (df['chfem_Kxx'] + df['chfem_Kyy'] + df['chfem_Kzz']) / 3.0
    else:
        print("Warning: chfem results are missing. Plotting will be skipped.")
        return

    # df_clean is only used for Matplotlib
    df_clean = df.dropna(subset=['K_eff']).copy()

    if len(df_clean) == 0:
        print("No valid conductivity data found to plot (df_clean is empty).")
        return

    plt.figure(figsize=(8, 6))
    
    # Define colors and markers for the levels
    colors = {2: '#0072B2', 4: '#D55E00', 6: '#009E73'}
    markers = {2: 'o', 4: 's', 6: '^'}

    radii = sorted(df_clean['Radius'].unique())

    for r in radii:
        group = df_clean[df_clean['Radius'] == r]
        Vf_clean = group['Total_Vf'].values
        K_eff_clean = group['K_eff'].values
        
        c = colors.get(r, '#000000')
        m = markers.get(r, 'o')

        # Scatter plot for simulation data
        plt.scatter(Vf_clean * 100, K_eff_clean, color=c, marker=m, edgecolor='k', s=80, alpha=0.7, label=f'r={r} (Sim)', zorder=3)

        # Estimate percolation threshold (Vc) and critical exponent (t) via curve fitting
        try:
            popt, _ = curve_fit(
                percolation_power_law, Vf_clean, K_eff_clean,
                p0=[1000.0, 0.02, 1.5],
                bounds=([0.0, 0.0, 1.0], [np.inf, 0.2, 4.0]) 
            )
            k0_fit, vc_fit, t_fit = popt
            
            print(f"[Radius={r}] Fitted: K0={k0_fit:.2f}, Vc={vc_fit*100:.2f}%, t={t_fit:.2f}")

            # Smooth curve for plotting
            v_smooth = np.linspace(0, max(Vf_clean) * 1.1, 200)
            k_smooth = percolation_power_law(v_smooth, k0_fit, vc_fit, t_fit)
            
            formula_label = f'Fit (r={r}): $V_c={vc_fit*100:.1f}\\%$'
            plt.plot(v_smooth * 100, k_smooth, color=c, linestyle='--', linewidth=2.5, label=formula_label, zorder=2)
                     
        except RuntimeError:
            print(f"[Radius={r}] Curve fitting failed. Data might not show clear percolation yet.")
        except Exception as e:
            print(f"[Radius={r}] Curve fitting error: {e}")

    plt.xlabel('Filler Volume Fraction (%)')
    plt.ylabel('Effective Conductivity (S/m)') 
    plt.title('Percolation Threshold Analysis by Filler Radius')
    
    plt.yscale('log')
    plt.ylim(bottom=1e-5) 
    plt.xlim(left=0)
    
    plt.grid(True, which="both", ls="--", alpha=0.5, zorder=1)
    plt.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    
    out_img = 'exp0_percolation_plot.png'
    plt.savefig(out_img, dpi=300)
    print(f"Plot saved as '{out_img}'")

if __name__ == "__main__":
    main()
