import subprocess
import os

def run():
    csv_log = "scale_check_results.csv"
    out_dir = "result_scale_check"
    os.makedirs(out_dir, exist_ok=True)
    
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
    vfs = [0.01, 0.03, 0.05, 0.07]
    seeds = [1, 2, 3]

    print(f"Starting Scale & Box Size Check -> {csv_log}")
    
    for size, length, radius in configs:
        for vf in vfs:
            for seed in seeds:
                recipe = f"rigidfiber:{vf}:length={length}:radius={radius}"
                
                # Format output filename so conditions can be distinguished
                basename = os.path.join(out_dir, f"scale_s{size}_L{length}_r{radius}_vf{vf:.2f}_seed{seed}")
                
                cmd = [
                    "python3", "run_pipeline.py",
                    "--size", str(size),
                    "--bg_type", "single",
                    "--physics_mode", "electrical",
                    "--solver", "chfem",
                    "--basename", basename,
                    "--csv_log", csv_log,
                    "--seed", str(seed),
                    "--recipe"
                ] + recipe.split()  # Expand space-separated recipe into a list and combine
                
                print(f"\nRunning: Size={size}, L={length}, r={radius}, Vf={vf}, Seed={seed}")
                subprocess.run(cmd)

if __name__ == "__main__":
    run()
