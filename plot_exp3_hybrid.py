import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif':  ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'], 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

df = pd.read_csv("exp3_hybrid_results.csv")
# Extract the volume fraction of flakes from the recipe string for the horizontal axis
df['v_flk'] = df['Recipe'].str.extract(r'flake:([0-9\.]+)').astype(float)
df['K_eff'] = (df['chfem_Kxx'] + df['chfem_Kyy'] + df['chfem_Kzz']) / 3.0
df_clean = df.dropna(subset=['K_eff']).sort_values('v_flk')

plt.figure(figsize=(8, 6))
# Scatter plot
plt.scatter(df_clean['v_flk'] * 100, df_clean['K_eff'], color='#009E73', alpha=0.6, label='Individual runs')

# Connect the mean values with a line
mean_df = df_clean.groupby('v_flk')['K_eff'].mean().reset_index()
plt.plot(mean_df['v_flk'] * 100, mean_df['K_eff'], color='#000000', marker='o', markersize=8, linewidth=2.5, label='Mean K_eff (Total Vf=7%)')

plt.xlabel('Flake Volume Fraction (%)')
plt.ylabel('Effective Conductivity (S/m)')
plt.title('Synergistic Effect of Fiber/Flake Hybrid')
plt.yscale('log')
plt.ylim(bottom=1e-5)
# Adding a secondary axis (Twin Axis) at the top of the X-axis to show the fiber volume fraction makes it more academic
ax1 = plt.gca()
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f"{7.0 - x:.1f}" for x in ax1.get_xticks()])
ax2.set_xlabel('Fiber Volume Fraction (%)')

ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(loc='lower center')
plt.tight_layout()
plt.savefig('exp3_hybrid_plot.png', dpi=300)
print("Saved exp3_hybrid_plot.png")
