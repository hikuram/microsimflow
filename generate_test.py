import random

# Variations used for testing
bg_types = ["single", "gyroid", "sea_island", "island_sea", "lamellar", "cylinder", "bcc"]
physics_modes = ["thermal", "electrical", "mechanics", "permeability"]
phaseA_ratios = [0.3, 0.5, 0.7]
base_rotations = ["0", "1", "2"]
stretch_options = ["1.0", "1.2", "1.5"]
stretch_axes = ["X", "Y", "Z"]

# Diverse recipes for edge cases and complex structural interactions
recipes = [
    # 1. Rigid fiber with strong anisotropy (diagonal alignment)
    "rigidfiber:0.04:length=60:radius=2:mean_dir=1,1,0:kappa=5.0",
    # 2. Flakes with strong Z-axis alignment
    "flake:0.04:radius=12:thickness=2:mean_dir=0,0,1:kappa=10.0",
    # 3. Highly flexible fibers with multiple bends
    "flexfiber:0.04:length=70:radius=2:max_bend_deg=60:max_total_bends=15",
    # 4. Agglomerate (clustered fibers)
    "agglomerate:0.04:num_fibers=4:length=50:radius=2:max_bend_deg=45",
    # 5. [NEW] Staggered (brick-and-mortar) structure
    "staggered:0.05:radius=15:layer_thickness=2:min_layers=2:max_layers=4",
    # 6. [NEW] Irregular cross-section fiber (bean shape)
    "irregfiber:0.04:length=50:shape=bean:radius=5:ratio=0.5",
    # 7. [NEW] Irregular cross-section fiber (C-shape)
    "irregfiber:0.04:length=50:shape=c-shape:radius=5:ratio=0.7",
    # 8. [NEW] Simple spheres (for affine deformation tracking test)
    "sphere:0.05:radius=6",
    # 9. Hybrid: Flakes + Rigid fibers
    "flake:0.02:radius=15:thickness=2 rigidfiber:0.02:length=60:radius=2",
    # 10. Advanced Hybrid: Staggered structure + Agglomerates
    "staggered:0.03:radius=12 agglomerate:0.02:num_fibers=3:length=50:radius=2"
]

# Initialize the Python script string
py_script = 'import subprocess\nimport os\n\n'
py_script += 'def run_tests():\n'
py_script += '    csv_log = "test_suite_results.csv"\n'
py_script += '    out_dir = "result_test"\n'
py_script += '    os.makedirs(out_dir, exist_ok=True)\n\n'
py_script += '    print("Starting Advanced Randomized Test Suite (10 configurations)...")\n\n'

# Generate 10 random test cases
for i in range(1, 11):
    bg = random.choice(bg_types)
    mode = random.choice(physics_modes)
    ratio = random.choice(phaseA_ratios)
    rot = random.choice(base_rotations)
    stretch = random.choice(stretch_options)
    s_axis = random.choice(stretch_axes)
    recipe = random.choice(recipes)
    
    basename = f"test_{i:02d}_{bg}_{mode}"
    
    py_script += f'    print("\\n========================================")\n'
    py_script += f'    print("Running Test {i}/10: [BG: {bg}] [Mode: {mode}] [Stretch: {stretch} ({s_axis})]")\n'
    py_script += f'    basename = os.path.join(out_dir, "{basename}")\n'
    py_script += f'    recipe = "{recipe}"\n'
    
    # Run at size=100 for faster test completion
    py_script += '    cmd = [\n'
    py_script += '        "python3", "-m", "run_pipeline",\n'
    py_script += '        "--size", "100",\n'
    py_script += f'        "--bg_type", "{bg}",\n'
    py_script += f'        "--physics_mode", "{mode}",\n'
    py_script += f'        "--phaseA_ratio", "{ratio}",\n'
    py_script += f'        "--base_rotation", "{rot}",\n'
    py_script += f'        "--stretch_ratios", "{stretch}",\n'
    py_script += f'        "--stretch_axis", "{s_axis}",\n'
    py_script += '        "--solver", "chfem",\n' # Use chfem for quick testing
    py_script += '        "--basename", basename,\n'
    py_script += '        "--csv_log", csv_log,\n'
    py_script += '        "--recipe"\n'
    py_script += '    ] + recipe.split()\n'
    py_script += '    subprocess.run(cmd)\n\n'

py_script += 'if __name__ == "__main__":\n'
py_script += '    run_tests()\n'

# Output as a Python script
with open("run_tests.py", "w") as f:
    f.write(py_script)

print("Generated 'run_tests.py' with 10 advanced randomized test cases.")
