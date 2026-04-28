import subprocess
import os

def get_next_result_dir(base_name="exp5_"):
    i = 1
    while True:
        dir_name = f"{base_name}{i:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def run():
    csv_log = "exp5_morphology.csv"
    out_dir = get_next_result_dir()
    os.makedirs(out_dir)
    
    # Sweep volume fraction of Phase A (Soft phase)
    phaseA_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Define elastic properties: "Bulk/Young's_Modulus Poisson's_Ratio"
    # Soft phase (E=1.0), Hard phase (E=100.0)
    prop_A = "1.0 0.35"
    prop_B = "100.0 0.30"
    
    # Morphologies to compare
    morphologies = ["lamellar", "cylinder", "gyroid", "sea_island"]
    seeds = [1]

    print(f"Created output directory: {out_dir}")
    print(f"Starting Tutorial Part 1: Morphology Effect (Mechanics). Logging to '{csv_log}'")
    
    for bg in morphologies:
        for phaseA_ratio in phaseA_ratios:
            for seed in seeds:
                basename = os.path.join(out_dir, f"morph_{bg}_vfA{phaseA_ratio:.1f}_seed{seed}")
                
                cmd = [
                    "python3", "-m", "run_pipeline",
                    "--size", "150",
                    "--bg_type", bg,
                    "--phaseA_ratio", str(phaseA_ratio),
                    "--physics_mode", "mechanics",
                    "--solver", "chfem",
                    "--prop_A", prop_A,
                    "--prop_B", prop_B,
                    "--basename", basename,
                    "--csv_log", csv_log,
                    "--seed", str(seed),
                    "--recipe", "none:0" # Dummy recipe
                ]
                subprocess.run(cmd)

if __name__ == "__main__":
    run()
