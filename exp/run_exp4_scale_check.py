import subprocess
import os

def get_next_result_dir(base_name="result_exp4_"):
    """Generate sequential directory names automatically."""
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp4_scale_check_results.csv"
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    # Pattern definition: (box_size, length, radius)
    configs = [
        # Size 200 (1 pattern)
        (200, 60, 2),
        
        # Size 300 (2 patterns)
        (300, 60, 2),
        (300, 90, 3),
        
        # Size 400 (3 patterns)
        (400, 60, 2),
        (400, 90, 3),
        (400, 120, 4)
    ]
    
    # Coarse sweep to grasp trends
    vfs = [0.03, 0.06, 0.09, 0.12]
    seeds = [1, 2, 3]

    print(f"Created output directory: {out_dir}")
    print(f"Starting Scale & Box Size Check. All metrics appended to '{csv_log}'")
    
    for size, length, radius in configs:
        for vf in vfs:
            for seed in seeds:
                recipe = f"rigidfiber:{vf}:length={length}:radius={radius}"
                
                # Format output filename so conditions can be distinguished
                basename = os.path.join(out_dir, f"scale_s{size}_L{length}_r{radius}_vf{vf:.2f}_seed{seed}")
                
                cmd = [
                    "python3", "-m", "run_pipeline",
                    "--size", str(size),
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
