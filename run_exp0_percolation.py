import subprocess
import os
import numpy as np

def run():
    csv_log = "exp0_percolation_results.csv"
    out_dir = "result_exp0"
    os.makedirs(out_dir, exist_ok=True)
    
    # Sweep volume fractions from 0.01 to 0.10 in increments of 0.01
    vfs = np.round(np.arange(0.01, 0.11, 0.01), 2)
    
    # 2 levels of filler radius (e.g., changing the aspect ratio if length is fixed)
    radii = [2, 4, 6]
    seeds = [1, 2, 3]

    print(f"Starting Exp0: Standard Percolation Sweep -> {csv_log}")
    
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
