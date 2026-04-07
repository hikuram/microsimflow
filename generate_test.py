import random

# Variations used for testing
bg_types = ["single", "gyroid", "sea_island", "island_sea", "lamellar", "cylinder", "bcc"]
physics_modes = ["thermal", "electrical", "mechanics"]
phaseA_ratios = [0.3, 0.5, 0.7]

# Moderate filler combination recipes
recipes = [
    "flake:0.04:radius=10:thickness=2",
    "adaptfiber:0.04:length=60:radius=2:max_total_bends=3",
    "flexfiber:0.04:length=60:radius=2",
    "agglomerate:0.04:num_fibers=3:length=60:radius=2",
    "flake:0.02:radius=15:thickness=2 adaptfiber:0.02:length=60:radius=2",
    "flake:0.02:radius=15:thickness=2 agglomerate:0.02:num_fibers=5:length=60:radius=2"
]

# Initialize the Python script string
py_script = 'import subprocess\nimport os\n\n'
py_script += 'def run_tests():\n'
py_script += '    csv_log = "test_suite_results.csv"\n'
py_script += '    out_dir = "result_test"\n'
py_script += '    os.makedirs(out_dir, exist_ok=True)\n\n'
py_script += '    print("Starting Randomized Test Suite (10 configurations)...")\n\n'

# Generate 10 random test cases
for i in range(1, 11):
    bg = random.choice(bg_types)
    mode = random.choice(physics_modes)
    ratio = random.choice(phaseA_ratios)
    recipe = random.choice(recipes)
    
    basename = f"test_{i:02d}_{bg}_{mode}"
    
    py_script += f'    print("\\n========================================")\n'
    py_script += f'    print("Running Test {i}/10: [BG: {bg}] [Mode: {mode}]")\n'
    py_script += f'    basename = os.path.join(out_dir, "{basename}")\n'
    py_script += f'    recipe = "{recipe}"\n'
    
    # Run at size=100 for faster test completion
    py_script += '    cmd = [\n'
    py_script += '        "python3", "run_pipeline.py",\n'
    py_script += '        "--size", "100",\n'
    py_script += f'        "--bg_type", "{bg}",\n'
    py_script += f'        "--physics_mode", "{mode}",\n'
    py_script += f'        "--phaseA_ratio", "{ratio}",\n'
    py_script += '        "--solver", "chfem",\n'
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

print("Generated 'run_tests.py' with 10 random test cases.")
