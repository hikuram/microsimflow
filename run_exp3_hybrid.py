import subprocess
import os
import numpy as np

def run():
    csv_log = "exp3_hybrid_results.csv"
    out_dir = "result_exp3"
    os.makedirs(out_dir, exist_ok=True)
    
    total_vf = 0.07
    seeds = [1, 2, 3]

    print(f"Starting Exp3: Hybrid Synergy Sweep -> {csv_log}")
    # Decrease v_fib from 0.07 to 0.00 in increments of 0.01
    for v_fib in np.round(np.arange(0.07, -0.01, -0.01), 2):
        v_flk = round(total_vf - v_fib, 2)
        
        for seed in seeds:
            # Combine the two recipes for fibers and flakes (if volume fraction is 0, placement is skipped)
            recipe = f"rigidfiber:{v_fib}:length=60:radius=2 flake:{v_flk}:radius=15:thickness=2"
            basename = os.path.join(out_dir, f"hybrid_fib{v_fib:.2f}_flk{v_flk:.2f}_seed{seed}")
            
            cmd = [
                "python3", "run_pipeline.py",
                "--size", "200",
                "--bg_type", "single",
                "--physics_mode", "electrical",
                "--solver", "chfem",
                "--basename", basename,
                "--csv_log", csv_log,
                "--seed", str(seed),
                "--recipe"
            ] + recipe.split()  # <- Changed here: split string into a list and combine
            
            subprocess.run(cmd)

if __name__ == "__main__":
    run()
