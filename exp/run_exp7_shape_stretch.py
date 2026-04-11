import subprocess
import os
import numpy as np

def get_next_result_dir(base_name="result_exp7_"):
    """Generate sequential directory names to prevent overwriting results."""
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp7_shape_stretch_results.csv"
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    # Stretch ratios to evaluate (from unstretched to 50% elongation)
    stretch_ratios = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
    
    # Random seeds for statistical sampling
    seeds = [1, 2, 3]
    
    # Shape recipes (adjusted to achieve comparable initial conductivities)
    recipes = {
        "Sphere": f"sphere:{0.28}:radius=3",
        "Flake": f"flake:{0.16}:radius=12:thickness=2",
        "RigidFiber": f"rigidfiber:{0.16}:length=40:radius=2",
        "Staggered": f"staggered:{0.22}:radius=12:layer_thickness=2:min_layers=1:max_layers=3:max_offset_pct=20"
    }

    print(f"Created output directory: {out_dir}")
    print(f"Starting Exp7: Shape & Stretch Dependency. Metrics appended to '{csv_log}'")
    
    for shape_name, recipe in recipes.items():
        for seed in seeds:
            basename = os.path.join(out_dir, f"exp7_{shape_name}_seed{seed}")
            
            cmd = [
                "python3", "run_pipeline.py",
                "--size", "150",
                "--physics", "electrical",
                "--recipe", recipe,
                "--stretch", *stretch_ratios,
                "--seed", str(seed),
                "--basename", basename,
                "--csv_log", csv_log
            ]
            
            print(f"\nRunning: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {shape_name} (Seed {seed}): {e}")

if __name__ == "__main__":
    run()
