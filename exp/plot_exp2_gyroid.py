import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif':  ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'], 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

def power_law(v, k0, vc, t):
    return k0 * np.maximum(v - vc, 0)**t + 1e-6

df = pd.read_csv("exp2_gyroid_results.csv")
# Use pure Filler_Frac not including the interface phase
df['Vf'] = df['Filler_Frac']
df['K_eff'] = (df['chfem_Kxx'] + df['chfem_Kyy'] + df['chfem_Kzz']) / 3.0
df_clean = df.dropna(subset=['Vf', 'K_eff'])

plt.figure(figsize=(8, 6))
v_data = df_clean['Vf'].values
k_data = df_clean['K_eff'].values

plt.scatter(v_data * 100, k_data, color='#D55E00', s=80, edgecolor='k', label='Gyroid Data')

try:
    popt, _ = curve_fit(power_law, v_data, k_data, p0=[1000.0, 0.01, 2.0], bounds=([0, 0, 1.0], [np.inf, 0.2, 4.0]))
    v_smooth = np.linspace(0, max(v_data) * 1.1, 200)
    plt.plot(v_smooth * 100, power_law(v_smooth, *popt), color='k', linestyle='--', linewidth=2.5, 
             label=f'Fit: Vc={popt[1]*100:.2f}%, t={popt[2]:.2f}')
except:
    print("Fit failed.")

plt.xlabel('Filler Volume Fraction (%)') 
plt.ylabel('Effective Conductivity (S/m)')
plt.title('Double Percolation Threshold (Gyroid)')
plt.yscale('log')
plt.ylim(bottom=1e-5)
plt.xlim(left=0)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('exp2_gyroid_plot.png', dpi=300)
print("Saved exp2_gyroid_plot.png")
