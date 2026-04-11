import subprocess
import os
import numpy as np

def get_next_result_dir(base_name="result_exp0_"):
    """Generate sequential directory names (e.g., result_exp0_01) automatically."""
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp0_percolation_results.csv"  # Accumulated in the root directory
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    # Sweep volume fractions from 0.02 to 0.30 in increments of 0.02
    vfs = np.round(np.arange(0.02, 0.32, 0.02), 2)
    
    # 2 levels of filler radius (e.g., changing the aspect ratio if length is fixed)
    radii = [2, 4, 6]
    seeds = [1, 2, 3]

    print(f"Created output directory: {out_dir}")
    print(f"Starting Exp0: Standard Percolation Sweep. All metrics appended to '{csv_log}'")
    
    for r in radii:
        for vf in vfs:
            for seed in seeds:
                # Using a fixed length of 30 to observe the effect of the radius/aspect ratio
                recipe = f"rigidfiber:{vf}:length=30:radius={r}"
                basename = os.path.join(out_dir, f"perc_r{r}_vf{vf:.2f}_seed{seed}")
                
                cmd = [
                    "python3", "run_pipeline.py",
                    "--size", "100",
                    "--bg_type", "single",
                    "--physics_mode", "electrical",
                    "--solver", "chfem",
                    "--basename", basename,
                    "--csv_log", csv_log,
                    "--seed", str(seed),
                    "--recipe"
                ] + recipe.split()
                
                subprocess.run(cmd)

if __name__ == "__main__":
    run()
