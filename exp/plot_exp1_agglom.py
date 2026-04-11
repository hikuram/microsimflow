import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif':  ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'], 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

df = pd.read_csv("exp1_agglom_results.csv")
# Extract num_fibers from the recipe
df['n'] = df['Recipe'].str.extract(r'num_fibers=(\d+)').astype(int)
df['K_eff'] = (df['chfem_Kxx'] + df['chfem_Kyy'] + df['chfem_Kzz']) / 3.0
df_clean = df.dropna(subset=['K_eff']).sort_values('n')

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
plt.xticks(ns := sorted(df_clean['n'].unique()))
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('exp1_agglom_plot.png', dpi=300)
print("Saved exp1_agglom_plot.png")
