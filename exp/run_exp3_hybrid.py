import subprocess
import os
import numpy as np

def get_next_result_dir(base_name="result_exp3_"):
    """Generate sequential directory names automatically."""
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp3_hybrid_results.csv"
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    total_vf = 0.08
    seeds = [1, 2, 3]

    print(f"Created output directory: {out_dir}")
    print(f"Starting Exp3: Hybrid Synergy Sweep. All metrics appended to '{csv_log}'")
    
    # Decrease v_fib from 0.08 to 0.00 in increments of 0.01
    for v_fib in np.round(np.arange(0.08, -0.01, -0.01), 2):
        v_flk = round(total_vf - v_fib, 2)
        
        for seed in seeds:
            # Combine the two recipes for fibers and flakes
            recipe = f"rigidfiber:{v_fib}:length=60:radius=2 flake:{v_flk}:radius=15:thickness=2"
            basename = os.path.join(out_dir, f"hybrid_fib{v_fib:.2f}_flk{v_flk:.2f}_seed{seed}")
            
            cmd = [
                "python3", "-m", "run_pipeline",
                "--size", "200",
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
