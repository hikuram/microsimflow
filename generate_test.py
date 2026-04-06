import subprocess
import os

def run_tests():
    csv_log = "test_suite_results.csv"
    out_dir = "result_test"
    os.makedirs(out_dir, exist_ok=True)
    
    # Common settings reflecting recent tuning (contrast ratio 10^8)
    # Polymer: 1e-4, Interphase (tunnel): 1e0, Filler (default): 1e4
    common_args = [
        "--size", "150",             # Set slightly smaller for testing (change to 200 etc. if necessary)
        "--physics_mode", "electrical",
        "--solver", "chfem",         # or "both"
        "--csv_log", csv_log,
        "--prop_A", "1e-4",
        "--prop_B", "1e-4",
        "--prop_inter", "1e0"
    ]

    # Definition of test cases
    tests = [
        {
            "name": "test01_gyroid_hybrid",
            "bg_type": "gyroid",
            "desc": "Double percolation with fiber/flake hybrid",
            "recipe": "rigidfiber:0.03:length=60:radius=2 flake:0.02:radius=15:thickness=2",
            "seed": "1"
        },
        {
            "name": "test02_agglom_single",
            "bg_type": "single",
            "desc": "Agglomeration network breakdown in single phase",
            # New agglomerate logic (bilateral growth/bending)
            "recipe": "agglomerate:0.05:num_fibers=5:length=60:radius=2:max_bend_deg=90:max_total_bends=10",
            "seed": "42"
        },
        {
            "name": "test03_complex_bcc",
            "bg_type": "bcc",
            "desc": "Complex morphology (BCC) with adaptive fibers, flakes, and spheres",
            # Renamed from adfiber to adaptfiber
            "recipe": "adaptfiber:0.02:length=90:radius=2 flake:0.02:radius=10:thickness=2 sphere:0.01:radius=5",
            "seed": "99"
        }
    ]

    print(f"Starting Comprehensive Pipeline Test Suite -> {csv_log}")

    for t in tests:
        basename = os.path.join(out_dir, t["name"])
        print(f"\n========================================================")
        print(f"Running Test : {t['name']}")
        print(f"Description  : {t['desc']}")
        print(f"Background   : {t['bg_type']}")
        
        # Construct command and expand recipe using split()
        cmd = [
            "python3", "run_pipeline.py",
            "--bg_type", t["bg_type"],
            "--basename", basename,
            "--seed", t["seed"]
        ] + common_args + ["--recipe"] + t["recipe"].split()
        
        subprocess.run(cmd)
        
    print("\nAll tests completed successfully.")

if __name__ == "__main__":
    run_tests()
