import argparse
import time
import csv
import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Import from micro_builder
from micro_builder import (
    set_random_seed,
    build_single_phase_grid,
    build_tpms_grid_with_target_ratio,
    build_sea_island_grid,
    build_island_sea_grid,
    build_lamellar_grid,
    build_cylinder_hex_grid,
    build_bcc_grid,
    place_fillers_hybrid,
    place_adaptive_fibers,
    get_flake_mask,
    get_sphere_mask,
    get_rigid_cylinder_mask,
    get_flexible_fiber_mask,
    get_agglomerate_mask,
    get_staggered_flakes_mask,
    finalize_microstructure,
    export_chfem_inputs,
    export_visualization_vti,
    summarize_phase_fractions,
    apply_background_deformation,
    render_deformed_fillers
)

def parse_chfem_log(log_path):
    """Extract tensor components and computation time (sum of X, Y, Z) from chfem log"""
    kxx, kyy, kzz = None, None, None
    total_time = 0.0
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        mat_match = re.search(r'Homogenized Constitutive Matrix.*?:[^\n]*\n(.*?)\n-{10,}', content, re.DOTALL)
        if mat_match:
            lines = [line.strip() for line in mat_match.group(1).strip().split('\n')]
            kxx = float(lines[0].split()[0])
            kyy = float(lines[1].split()[1])
            kzz = float(lines[2].split()[2])

        # Match 'Elapsed time:' or 'Elapsed time (total):' and sum them all up
        time_matches = re.findall(r'Elapsed time(?: \(total\))?:\s*([\d\.eE\+\-]+)', content)
        if time_matches:
            total_time = sum(float(t) for t in time_matches)
    except Exception as e:
        print(f"chfem log parsing error: {e}")

    return kxx, kyy, kzz, total_time

def run_puma_laplace(final_grid, voxel_size, physics_mode, cond_map):
    """Solve the Laplace equation (thermal/electrical conduction) using PuMA's Python API"""
    try:
        import pumapy as puma
    except ImportError:
        print("Error: pumapy module not found. Cannot run PuMA solver.")
        return None, None, None, 0.0

    if physics_mode == 'mechanics':
        print("Mechanics mode is currently routed to chfem only in this wrapper. Skipping PuMA.")
        return None, None, None, 0.0

    # Transpose dimensions: NumPy (Z, Y, X) -> PuMA Workspace (X, Y, Z)
    ws = puma.Workspace.from_array(final_grid.transpose(2, 1, 0))
    ws.voxel_length = voxel_size

    puma_cond_map = puma.IsotropicConductivityMap()
    for phase_id, cond_val in cond_map.items():
        puma_cond_map.add_material((int(phase_id), int(phase_id)), float(cond_val))

    print("\n--- Running PuMA Solver ---")
    t0 = time.time()

    try:
        # Compute for each XYZ direction (specify periodic boundary conditions with side_bc='p')
        print("Computing X direction...")
        res_x = puma.compute_thermal_conductivity(ws, puma_cond_map, direction='x', side_bc='p', solver_type='cg')
        kxx = res_x[0] if isinstance(res_x, tuple) else res_x
        
        print("Computing Y direction...")
        res_y = puma.compute_thermal_conductivity(ws, puma_cond_map, direction='y', side_bc='p', solver_type='cg')
        kyy = res_y[0] if isinstance(res_y, tuple) else res_y
        
        print("Computing Z direction...")
        res_z = puma.compute_thermal_conductivity(ws, puma_cond_map, direction='z', side_bc='p', solver_type='cg')
        kzz = res_z[0] if isinstance(res_z, tuple) else res_z
        
        total_time = time.time() - t0
        print(f"PuMA computation completed in {total_time:.2f}s")
        
        return kxx, kyy, kzz, total_time

    except Exception as e:
        print(f"PuMA encountered an error during computation: {e}")
        return None, None, None, 0.0

def save_thumbnail_png(grid, filename, phase_labels=None):
    """
    Save the Z-axis center slice as a pure, margin-less PNG for lightweight storage and montage assembly.
    Colorbars and axes are completely removed to reduce file size.
    
    phase_labels: Phase definitions in dictionary format {ID: 'Label'}
    """
    if phase_labels is None:
        phase_labels = {0: 'Polymer A', 1: 'Polymer B', 2: 'Secondary Inter', 3: 'Primary Inter', 4: 'Filler'}
        
    # Calculate dynamic settings to maintain consistent colors across all images
    ids = list(phase_labels.keys())
    num_phases = len(ids)
    vmin = min(ids) - 0.5
    vmax = max(ids) + 0.5
    custom_cmap = plt.get_cmap('viridis', num_phases)

    z_mid = grid.shape[0] // 2
    slice_img = grid[z_mid, :, :]
    
    # Save the pure 2D array directly to a PNG file without any Matplotlib figure overhead
    plt.imsave(filename, slice_img, cmap=custom_cmap, vmin=vmin, vmax=vmax, origin='upper')
    print(f"Saved clean thumbnail: {filename}")

def update_pvd_file(pvd_filepath, dataset_records):
    """
    Creates or updates a ParaView Data (.pvd) file to group VTI files as a time-series.
    This ensures ParaView correctly handles varying grid extents during deformation.
    """
    with open(pvd_filepath, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for timestep, filepath in dataset_records:
            # Extract basename for the 'file' attribute to ensure relative path portability
            rel_filename = os.path.basename(filepath)
            f.write(f'    <DataSet timestep="{timestep}" group="" part="0" file="{rel_filename}"/>\n')
            
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

def export_common_legend(phase_labels, filename="common_legend.png"):
    """
    Exports a standalone legend image to be shared across all montages.
    Skips generation if the file already exists to save time.
    """
    if os.path.exists(filename):
        return
        
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
        'axes.labelsize': 12, 'ytick.labelsize': 12, 
        'savefig.bbox': 'tight'
    })

    ids = list(phase_labels.keys())
    labels = list(phase_labels.values())
    num_phases = len(ids)
    
    vmin = min(ids) - 0.5
    vmax = max(ids) + 0.5
    custom_cmap = plt.get_cmap('viridis', num_phases)

    fig, ax = plt.subplots(figsize=(1.5, 4)) 
    ax.axis('off') # Hide the dummy axes
    
    # Create a dummy ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, ticks=ids, fraction=1.0, pad=0.0)
    cbar.ax.set_yticklabels(labels)
    cbar.ax.set_title("Phase ID", pad=15, fontweight='bold')

    plt.savefig(filename, dpi=200, transparent=False, facecolor='white')
    plt.close(fig)
    print(f"Saved common legend: {filename}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--voxel_size", type=float, default=1e-8)
    parser.add_argument("--bg_type", type=str, default="gyroid",
                        choices=["single", "gyroid", "sea_island", "island_sea", "lamellar", "cylinder", "bcc"])
    parser.add_argument("--phaseA_ratio", type=float, default=0.57)
    parser.add_argument("--recipe", nargs='+', required=True)
    parser.add_argument("--basename", type=str, default="model")
    parser.add_argument("--csv_log", type=str, default="comparison_results.csv")
    parser.add_argument("--physics_mode", type=str, default="thermal",
                        choices=["thermal", "electrical", "mechanics"])
    parser.add_argument("--solver", type=str, default="both",
                        choices=["chfem", "puma", "both", "skip"],
                        help="Solver for homogenisation. Use 'skip' to only build the model.")
    parser.add_argument("--prop_A", type=str, default=None, help="Property for Polymer A")
    parser.add_argument("--prop_B", type=str, default=None, help="Property for Polymer B")
    parser.add_argument("--prop_inter2", type=str, default=None, help="Property for Secondary Contact/Tunnel phase")
    parser.add_argument("--prop_inter", type=str, default=None, help="Property for Primary Contact/Tunnel phase")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--tunnel_radius", type=int, default=2, 
                        help="Radius in voxels for nearest-neighbor (tunneling) detection")
    parser.add_argument("--contact_radius", type=int, default=1, 
                        help="Radius in voxels for primary contact interface thickness")
    parser.add_argument("--stretch_ratios", type=float, nargs='+', default=[1.0], 
                        help="List of stretch ratios lambda along X-axis")
    parser.add_argument("--poisson_ratio", type=float, default=0.4, 
                        help="Poisson's ratio nu for transverse compression")
    return parser.parse_args()

def main():
    args = parse_args()
    # --- Fix the generator if a seed is specified ---
    if args.seed is not None:
        set_random_seed(args.seed)
    
    # Set default physical properties according to the physics mode
    if args.physics_mode == 'thermal':
        prop_A = args.prop_A or "0.3"
        prop_B = args.prop_B or "0.3"
        prop_secondary_inter = args.prop_inter2 or "30.0" # Safe fallback even if not present in grid
        prop_primary_inter = args.prop_inter or "30.0"
        default_filler = "300.0"
    elif args.physics_mode == 'electrical':
        prop_A = args.prop_A or "1e-4"
        prop_B = args.prop_B or "1e-4"
        prop_secondary_inter = args.prop_inter2 or "1e-3"
        prop_primary_inter = args.prop_inter or "1e-1"
        default_filler = "1e4"
    else: # mechanics
        prop_A = args.prop_A or "3.0 1.0"
        prop_B = args.prop_B or "3.0 1.0"
        prop_secondary_inter = args.prop_inter2 or "10.0 3.0"
        prop_primary_inter = args.prop_inter or "15.0 5.0"
        default_filler = "100.0 50.0"

    # Count valid recipes
    valid_recipes = [r for r in args.recipe if float(r.split(':')[1]) > 0]
    
    # Fixed ID assignment based on new micro_builder logic
    secondary_inter_id = 2
    primary_inter_id = 3
    filler_start_id = 4
    current_filler_id = filler_start_id
    
    # Initialize property mapping dictionary
    prop_map = {
        0: prop_A, 
        1: prop_B, 
        secondary_inter_id: prop_secondary_inter,
        primary_inter_id: prop_primary_inter
    }
    
    print(f"--- Pipeline Start: {args.basename} ({args.physics_mode} mode, bg: {args.bg_type}, solver: {args.solver}) ---")
    
    build_log = f"{args.basename}_build.log"
    with open(build_log, "w", encoding="utf-8") as f:
        f.write(f"--- Build Log for {args.basename} ---\n")
        f.write(f"Size: {args.size}^3, Voxel: {args.voxel_size}m, BG: {args.bg_type}\n")
        f.write(f"Mode: {args.physics_mode}, Solver: {args.solver}\n")
        f.write(f"Recipe: {' '.join(args.recipe)}\n")

    step_logs = []

    # 1. Branch for background generation
    t0 = time.time()
    if args.bg_type == "single":
        _, tpms_grid, _, actual_phaseA = build_single_phase_grid(args.size)
    elif args.bg_type == "lamellar":
        _, tpms_grid, _, actual_phaseA = build_lamellar_grid(args.size, wavelength=10, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "cylinder":
        _, tpms_grid, _, actual_phaseA = build_cylinder_hex_grid(args.size, wavelength=15, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "bcc":
        _, tpms_grid, _, actual_phaseA = build_bcc_grid(args.size, wavelength=15, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "sea_island":
        _, tpms_grid, _, actual_phaseA = build_sea_island_grid(args.size, island_radius=8, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "island_sea":
        _, tpms_grid, _, actual_phaseA = build_island_sea_grid(args.size, island_radius=8, target_phaseA_ratio=args.phaseA_ratio)
    else:
        # Gyroid
        _, tpms_grid, _, actual_phaseA = build_tpms_grid_with_target_ratio(args.size, wavelength=10, target_phaseA_ratio=args.phaseA_ratio)
        
    # Shell counter for electrical/mechanics mode (pass None for thermal)
    comp_grid = np.zeros((args.size, args.size, args.size), dtype=np.uint8)
    shell_count_grid = np.zeros_like(comp_grid) if args.physics_mode in ['electrical', 'mechanics'] else None
    step_logs.append(f"BG({args.bg_type}):{time.time() - t0:.1f}s")

    # Global placement registry for the reference configuration
    placement_registry = []

    # 2. Filler placement based on recipe
    for step in valid_recipes:
        parts = step.split(':')
        f_type = parts[0]
        f_vol = float(parts[1])

        opts = {}
        for p in parts[2:]:
            k, v = p.split('=')
            if k == 'prop': 
                opts[k] = v.replace('_', ' ')
            else: 
                opts[k] = float(v) if '.' in v else int(v)

        prop_map[current_filler_id] = opts.pop('prop', default_filler)
        protrusion_coef = opts.pop('protrusion_coef', 0.0025)
        t_step = time.time()
        
        hybrid_args = {
            'comp_grid': comp_grid,
            'tpms_grid': tpms_grid,
            'target_vol_frac': f_vol,
            'protrusion_coef': protrusion_coef,
            'log_file': build_log,
            'physics_mode': args.physics_mode,
            'shell_count_grid': shell_count_grid,
            'filler_id': current_filler_id,
            'inter_id': primary_inter_id, 
            'tunnel_radius': args.tunnel_radius,
            'placement_registry': placement_registry
        }
        
        if f_type == "flake":
            kwargs = {'radius': opts.get('radius', 10), 'thickness': opts.get('thickness', 2)}
            place_fillers_hybrid(filler_func=get_flake_mask, kwargs=kwargs, desc="Flake", **hybrid_args)
                                 
        elif f_type == "sphere":
            kwargs = {'radius': opts.get('radius', 5)}
            place_fillers_hybrid(filler_func=get_sphere_mask, kwargs=kwargs, desc="Sphere", **hybrid_args)
                                 
        elif f_type == "rigidfiber":
            kwargs = {'length': opts.get('length', 60), 'radius': opts.get('radius', 2)}
            place_fillers_hybrid(filler_func=get_rigid_cylinder_mask, kwargs=kwargs, desc="Rigid Fiber", **hybrid_args)
            
        elif f_type == "adaptfiber":
            # NOTE: adaptfiber is explicitly excluded from affine deformation support. 
            # It writes directly to comp_grid and is only evaluated at lambda=1.0.
            place_adaptive_fibers(comp_grid=comp_grid, tpms_grid=tpms_grid, target_vol_frac=f_vol,
                                  length=opts.get('length', 90), radius=opts.get('radius', 2),
                                  max_bend_deg=opts.get('max_bend_deg', 45), max_total_bends=opts.get('max_total_bends', 5),
                                  min_backbone_ratio=opts.get('min_backbone_ratio', 0.9), max_protrusion_ratio=protrusion_coef,
                                  log_file=build_log, physics_mode=args.physics_mode, shell_count_grid=shell_count_grid,
                                  filler_id=current_filler_id, inter_id=primary_inter_id)
                                  
        elif f_type == "flexfiber":
            kwargs = {
                'length': opts.get('length', 90), 'radius': opts.get('radius', 2),
                'max_bend_deg': opts.get('max_bend_deg', 90), 'max_total_bends': opts.get('max_total_bends', 10)
            }
            place_fillers_hybrid(filler_func=get_flexible_fiber_mask, kwargs=kwargs, desc="Flexible Fibers", **hybrid_args)
                                 
        elif f_type == "agglomerate":
            kwargs = {
                'num_fibers': opts.get('num_fibers', 5), 'length': opts.get('length', 90), 'radius': opts.get('radius', 2),
                'max_bend_deg': opts.get('max_bend_deg', 90), 'max_total_bends': opts.get('max_total_bends', 10)
            }
            desc_str = f"Agglomerate(n={int(opts.get('num_fibers', 5))})"
            place_fillers_hybrid(filler_func=get_agglomerate_mask, kwargs=kwargs, desc=desc_str, **hybrid_args)
                                 
        elif f_type == "staggered":
            kwargs = {
                'radius': opts.get('radius', 15), 'layer_thickness': opts.get('layer_thickness', 2),
                'min_layers': opts.get('min_layers', 1), 'max_layers': opts.get('max_layers', 4),
                'max_offset_pct': opts.get('max_offset_pct', 30)
            }
            place_fillers_hybrid(filler_func=get_staggered_flakes_mask, kwargs=kwargs, desc="Staggered Flakes", **hybrid_args)

        step_logs.append(f"{f_type}(ID:{current_filler_id}):{time.time() - t_step:.1f}s")
        current_filler_id += 1

    # =========================================================================
    # 3. One-Shot Stretching & Evaluation Loop
    # =========================================================================
    
    pvd_records = []
    pvd_filename = f"{args.basename}.pvd"
    
    # Ensure apply_background_deformation and render_deformed_fillers are imported at the top of run_pipeline.py!
    
    for stretch in args.stretch_ratios:
        print(f"\n{'='*50}")
        print(f"--- Processing Stretch Ratio: {stretch} ---")
        print(f"{'='*50}")
        
        current_basename = f"{args.basename}_L{stretch:.2f}"
        current_step_logs = step_logs.copy()
        t_def = time.time()
        
        # Deform ONLY the continuous polymer matrix
        current_tpms = apply_background_deformation(tpms_grid, stretch, args.poisson_ratio)
        
        # Initialize empty spaces for rigid fillers
        current_comp = np.zeros_like(current_tpms)
        current_shell = np.zeros_like(current_tpms) if args.physics_mode in ['electrical', 'mechanics'] else None
        
        # Redraw all rigid fillers using kinematic transformations
        render_deformed_fillers(
            placement_registry, 
            base_shape=(args.size, args.size, args.size), 
            stretch_ratio=stretch, 
            poisson_ratio=args.poisson_ratio, 
            is_thermal=(args.physics_mode == 'thermal'), 
            comp_grid=current_comp, 
            shell_count_grid=current_shell, 
            tunnel_radius=args.tunnel_radius
        )
        
        current_step_logs.append(f"Deformation({stretch}):{time.time() - t_def:.1f}s")

        # 4. Integration of final structure and creation of metadata
        final_grid = finalize_microstructure(
            current_comp, current_tpms, current_shell, args.physics_mode, 
            secondary_inter_id=secondary_inter_id, 
            primary_inter_id=primary_inter_id, 
            filler_start_id=filler_start_id,
            contact_radius=args.contact_radius
        )
        
        phase_stats = summarize_phase_fractions(
            final_grid, secondary_inter_id=secondary_inter_id, primary_inter_id=primary_inter_id, filler_start_id=filler_start_id
        )
        
        phase_labels = {0: 'Polymer A', 1: 'Polymer B', secondary_inter_id: 'Secondary Inter', primary_inter_id: 'Primary Inter'}
        for i in range(filler_start_id, current_filler_id):
            phase_labels[i] = f'Filler {i-filler_start_id+1}' if current_filler_id > filler_start_id + 1 else 'Filler'
        
        # Export the common legend once
        export_common_legend(phase_labels)
        
        metadata = {
            "Basename": current_basename,
            "Grid_Size": f"{final_grid.shape[2]}x{final_grid.shape[1]}x{final_grid.shape[0]}",
            "Voxel_Size_m": str(args.voxel_size),
            "BG_Type": args.bg_type,
            "Physics_Mode": args.physics_mode,
            "Solver": args.solver,
            "Recipe": " ".join(args.recipe),
            "PolymerA_Frac": f"{phase_stats['polymer_a_fraction']:.4f}",
            "PolymerB_Frac": f"{phase_stats['polymer_b_fraction']:.4f}",
            "Secondary_Inter_Frac": f"{phase_stats['secondary_interface_fraction']:.4f}",
            "Primary_Inter_Frac": f"{phase_stats['primary_interface_fraction']:.4f}",
            "Filler_Frac": f"{phase_stats['filler_total_fraction']:.4f}",
            "Stretch_Ratio": str(stretch),
            "Poisson_Ratio": str(args.poisson_ratio) if stretch != 1.0 else "N/A",
        }
        
        # Export VTI for visualization
        vti_filename = f"{current_basename}.vti"
        export_visualization_vti(final_grid, vti_filename, voxel_size=args.voxel_size, metadata=metadata)
        # --- Update PVD collection after each VTI export ---
        # Use the stretch ratio as the timestep value
        pvd_records.append((stretch, vti_filename))
        update_pvd_file(pvd_filename, pvd_records)
        print(f"Updated ParaView collection: {pvd_filename}")

        slice_filename = f"{current_basename}_slice.png"
        save_thumbnail_png(final_grid, slice_filename, phase_labels)

        # 5. Execute computational solvers and aggregate results
        chfem_time = chfem_kxx = chfem_kyy = chfem_kzz = ""
        puma_time = puma_kxx = puma_kyy = puma_kzz = ""

        if args.solver in ["chfem", "both"]:
            export_chfem_inputs(final_grid, current_basename, voxel_size=args.voxel_size, physics_mode=args.physics_mode, prop_map=prop_map)
            log_file = f"{current_basename}_metrics.txt"
            subprocess.run(["chfem_exec", f"{current_basename}.nf", f"{current_basename}.raw", "-m", log_file])
            ckx, cky, ckz, ctime = parse_chfem_log(log_file)
            if ckx is not None:
                chfem_time, chfem_kxx, chfem_kyy, chfem_kzz = f"{ctime:.2f}", ckx, cky, ckz

        if args.solver in ["puma", "both"]:
            cond_map = {k: float(v) for k, v in prop_map.items()}
            pkx, pky, pkz, ptime = run_puma_laplace(final_grid, args.voxel_size, args.physics_mode, cond_map)
            if pkx is not None:
                puma_time, puma_kxx, puma_kyy, puma_kzz = f"{ptime:.2f}", pkx, pky, pkz

        file_exists = os.path.isfile(args.csv_log)
        with open(args.csv_log, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Basename", "Grid_Size", "Voxel_Size_m", "BG_Type", "Mode", "Solver", "Recipe", 
                    "Stretch_Ratio", "Poisson_Ratio",
                    "PolymerA_Frac", "PolymerB_Frac", "Secondary_Inter_Frac", "Primary_Inter_Frac", "Filler_Frac", "Placement_Logs", 
                    "Slice_Image",
                    "chfem_Time_s", "chfem_Kxx", "chfem_Kyy", "chfem_Kzz",
                    "puma_Time_s", "puma_Kxx", "puma_Kyy", "puma_Kzz"
                ])
                
            writer.writerow([
                current_basename, f"{final_grid.shape[2]}x{final_grid.shape[1]}x{final_grid.shape[0]}", args.voxel_size, args.bg_type, args.physics_mode, args.solver, " ".join(args.recipe),
                stretch, args.poisson_ratio if stretch != 1.0 else "N/A",
                f"{phase_stats['polymer_a_fraction']:.4f}",
                f"{phase_stats['polymer_b_fraction']:.4f}",
                f"{phase_stats['secondary_interface_fraction']:.4f}", 
                f"{phase_stats['primary_interface_fraction']:.4f}", 
                f"{phase_stats['filler_total_fraction']:.4f}",
                " | ".join(current_step_logs),
                slice_filename,
                chfem_time, chfem_kxx, chfem_kyy, chfem_kzz,
                puma_time, puma_kxx, puma_kyy, puma_kzz
            ])

if __name__ == "__main__":
    main()
