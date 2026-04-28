import subprocess
import os
import numpy as np

def get_next_result_dir(base_name="exp6_"):
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp6_filler.csv"
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    # Mechanics default properties.
    prop_A = "1.0 0.35"
    prop_B = "1.0 0.35"
    prop_inter2 = "10.0 0.30"
    prop_inter = "100.0 0.25"
    prop_filler = "1000.0_0.20"
    
    # Volume fractions to sweep (1% to 9%)
    vfs = np.round(np.arange(0.01, 0.11, 0.02), 2)
    seeds = [1, 2]

    # Filler recipes with different aspect ratios
    filler_configs = {
        "Sphere": "sphere:{vf}:radius=6:prop={prop}",
        "Flake": "flake:{vf}:radius=10:thickness=2:prop={prop}",
        "Fiber": "rigidfiber:{vf}:length=30:radius=2:prop={prop}"
    }

    print(f"Created output directory: {out_dir}")
    print(f"Starting Tutorial Part 2: Pure Filler Shape Effect. Logging to '{csv_log}'")
    
    for name, recipe_template in filler_configs.items():
        for vf in vfs:
            for seed in seeds:
                recipe = recipe_template.format(vf=vf, prop=prop_filler)
                basename = os.path.join(out_dir, f"filler_{name}_vf{vf:.2f}_seed{seed}")
                
                cmd = [
                    "python3", "-m", "run_pipeline",
                    "--size", "150",
                    "--bg_type", "single",
                    "--physics_mode", "mechanics",
                    "--solver", "chfem",
                    "--prop_A", prop_A,
                    "--prop_B", prop_B,
                    "--prop_inter2", prop_inter2,
                    "--prop_inter", prop_inter,
                    "--basename", basename,
                    "--csv_log", csv_log,
                    "--seed", str(seed),
                    "--recipe"
                ] + recipe.split()
                
                subprocess.run(cmd)

if __name__ == "__main__":
    run()
