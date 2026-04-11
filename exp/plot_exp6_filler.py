import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_exp6_filler():
    csv_file = "exp6_filler.csv"
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # Extract filler type from the Recipe string
    df['Filler_Type'] = df['Recipe'].apply(lambda x: x.split(':')[0].capitalize())
    
    # Calculate average stiffness across XYZ directions (random orientation assumes isotropy)
    df['Avg_Stiffness'] = df[['chfem_Kxx', "chfem_Kyy", "chfem_Kzz"]].mean(axis=1)
    
    grouped = df.groupby(['Filler_Type', 'Filler_Frac'])['Avg_Stiffness'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    
    markers = {'Sphere': 'o', 'Flake': 's', 'Rigidfiber': '^'}
    
    for filler_type in grouped['Filler_Type'].unique():
        subset = grouped[grouped['Filler_Type'] == filler_type]
        ax.plot(subset['Filler_Frac'] * 100, subset['Avg_Stiffness'], 
                marker=markers.get(filler_type, 'x'), label=filler_type, markersize=8, linewidth=2)

    ax.set_xlabel("Filler Volume Fraction (%)")
    ax.set_ylabel("Average Effective Stiffness")
    ax.set_title("Tutorial Part 2: Filler Shape Reinforcement Effect\n(Random Orientation)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("exp6_filler_plot.png", dpi=300)
    print("Saved exp6 plot as 'exp6_filler_plot.png'")
    plt.close()

if __name__ == "__main__":
    plot_exp6_filler()
