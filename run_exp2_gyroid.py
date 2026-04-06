import subprocess
import os
import numpy as np

def run():
    csv_log = "exp2_gyroid_results.csv"
    out_dir = "result_exp2"
    os.makedirs(out_dir, exist_ok=True)
    
    # From 0.01 to 0.10 in increments of 0.01
    vfs = np.round(np.arange(0.01, 0.11, 0.01), 2)
    seeds = [1, 2, 3]

    print(f"Starting Exp2: Gyroid Percolation Sweep -> {csv_log}")
    for vf in vfs:
        for seed in seeds:
            recipe = f"rigidfiber:{vf}:length=60:radius=2"
            basename = os.path.join(out_dir, f"gyroid_vf{vf:.2f}_seed{seed}")
            
            cmd = [
                "python3", "run_pipeline.py",
                "--size", "200",
                "--bg_type", "gyroid",
                "--physics_mode", "electrical",
                "--solver", "chfem",
                "--recipe", recipe,
                "--basename", basename,
                "--csv_log", csv_log,
                "--seed", str(seed)
            ]
            subprocess.run(cmd)

if __name__ == "__main__":
    run()
