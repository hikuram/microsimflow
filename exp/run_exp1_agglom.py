import subprocess
import os

def get_next_result_dir(base_name="result_exp1_"):
    """Generate sequential directory names automatically."""
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp1_agglom_results.csv"
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    ns = [1, 2, 3, 5, 10, 15, 20, 25]
    seeds = [1, 2, 3]

    print(f"Created output directory: {out_dir}")
    print(f"Starting Exp1: Agglomeration Sweep. All metrics appended to '{csv_log}'")
    for n in ns:
        for seed in seeds:
            # Bending parameters to reproduce the agglomeration of flexible fibers
            recipe = f"agglomerate:0.08:num_fibers={n}:length=60:radius=2:max_bend_deg=90:max_total_bends=10"
            basename = os.path.join(out_dir, f"agglom_n{n}_seed{seed}")
            
            cmd = [
                "python3", "-m", "run_pipeline",
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
