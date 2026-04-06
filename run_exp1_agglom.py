import subprocess
import os

def run():
    csv_log = "exp1_agglom_results.csv"
    out_dir = "result_exp1"
    os.makedirs(out_dir, exist_ok=True)
    
    ns = [1, 2, 3, 5, 10, 15, 20, 25]
    seeds = [1, 2, 3]

    print(f"Starting Exp1: Agglomeration Sweep -> {csv_log}")
    for n in ns:
        for seed in seeds:
            # Bending parameters to reproduce the agglomeration of flexible fibers
            recipe = f"agglomerate:0.08:num_fibers={n}:length=60:radius=2:max_bend_deg=90:max_total_bends=10"
            basename = os.path.join(out_dir, f"agglom_n{n}_seed{seed}")
            
            cmd = [
                "python3", "run_pipeline.py",
                "--size", "200",
                "--bg_type", "single",
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
