import subprocess
import os
import numpy as np

def get_next_result_dir(base_name="result_exp2_"):
    """Generate sequential directory names automatically."""
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp2_gyroid_results.csv"
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    # From 0.01 to 0.10 in increments of 0.01
    vfs = np.round(np.arange(0.01, 0.11, 0.01), 2)
    seeds = [1, 2, 3]

    print(f"Created output directory: {out_dir}")
    print(f"Starting Exp2: Gyroid Percolation Sweep. All metrics appended to '{csv_log}'")
    for vf in vfs:
        for seed in seeds:
            recipe = f"rigidfiber:{vf}:length=60:radius=2"
            basename = os.path.join(out_dir, f"gyroid_vf{vf:.2f}_seed{seed}")
            
            cmd = [
                "python3", "-m", "run_pipeline",
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
