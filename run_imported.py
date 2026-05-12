#!/usr/bin/env python3
"""
Microsimflow Analysis: Imported Model Pipeline
Loads exported models (.raw/.nf) and executes structure analysis and physical solvers (chfem, PuMA).
Ensures consistent evaluation between digital twin and experimental data.
"""

import os
import sys
import json
import numpy as np
import argparse
import csv
import subprocess

# --- Import from Microsimflow internal modules ---
from micro_builder import (
    summarize_phase_fractions,
    compute_structure_metrics,
    export_chfem_inputs,
    export_visualization_vti
)
# Advanced PoreSpy metrics module (implemented in Phase B)
from micro_builder.postprocess import compute_advanced_metrics 

from pipeline.io_csv import structure_metrics_to_csv_fields, parse_chfem_log
from pipeline.solver_puma import run_puma_elasticity, run_puma_laplace
from pipeline.viz_export import save_thumbnail_png

def parse_args():
    parser = argparse.ArgumentParser(description="Microsimflow - Analysis Pipeline for Imported Real-Image Models")
    parser.add_argument("--import_path", type=str, required=True, 
                        help="Base path to imported model without extension (e.g., imported_models/sample_pbc)")
    parser.add_argument("--solver", type=str, choices=['chfem', 'puma', 'both', 'skip'], default='chfem', help="Solvers to execute")
    parser.add_argument("--physics_mode", type=str, choices=['thermal', 'electrical', 'mechanics', 'permeability'], default='thermal')
    parser.add_argument("--void_phases", type=int, nargs='+', default=[0], help="Phase IDs to be treated as void (fluid) in permeability mode")
    parser.add_argument("--csv_log", type=str, default="imported_results.csv", help="CSV log file path")
    parser.add_argument("--advanced_metrics", action="store_true", 
                        help="Enable PoreSpy advanced morphological analysis (SSA, Sphericity, Autocorrelation)")
    parser.add_argument("--skip_vti", action="store_true", help="Do not export VTI file")
    parser.add_argument("--pbc_pad", type=int, default=20, help="Padding voxels for PBC-aware morphology metrics")
    
    # Property override arguments (consistent with run_pipeline.py)
    parser.add_argument("--prop_A", type=str, default=None, help="Property for Polymer (Phase 0/1)")
    parser.add_argument("--prop_inter2", type=str, default=None, help="Property for Secondary Interface (Phase 2)")
    parser.add_argument("--prop_inter", type=str, default=None, help="Property for Primary Interface (Phase 3)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Metadata and Grid Binary
    nf_path = f"{args.import_path}_meta.nf"
    raw_path = f"{args.import_path}_final.raw"
    
    if not os.path.exists(nf_path) or not os.path.exists(raw_path):
        print(f"  ! Error: Metadata (.nf) or Binary (.raw) missing at: {args.import_path}")
        sys.exit(1)
        
    with open(nf_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    grid_shape = tuple(meta['grid_size'])
    voxel_size = meta['voxel_size_m']
    
    print(f"--- Analysis Started: {os.path.basename(args.import_path)} ---")
    print(f"  -> Source: {meta.get('source_image', 'N/A')}")
    print(f"  -> Pattern: {meta.get('interface_pattern', 'N/A')} | PBC: {meta.get('pbc_enforced', False)}")
    print(f"  -> Grid: {grid_shape[2]}x{grid_shape[1]}x{grid_shape[0]}, Voxel: {voxel_size:.2e} m")
    
    final_grid = np.fromfile(raw_path, dtype=np.uint8).reshape(grid_shape)
    
    # Constant Phase ID Mapping
    primary_inter_id = 3
    secondary_inter_id = 2
    filler_start_id = 4
    
    # --- Setup Physical Property Map ---
    # Default values follow run_pipeline.py logic
    if args.physics_mode == 'mechanics':
        prop_A = args.prop_A or "1.0 0.35"
        prop_B = prop_A
        prop_secondary_inter = args.prop_inter2 or "10.0 0.30"
        prop_primary_inter = args.prop_inter or "100.0 0.25"
        default_filler = "1000.0 0.20"
    elif args.physics_mode == 'electrical':
        prop_A = args.prop_A or "1e-4"
        prop_B = prop_A
        prop_secondary_inter = args.prop_inter2 or "1e-3"
        prop_primary_inter = args.prop_inter or "1e-1"
        default_filler = "1e4"
    else: # Default: thermal
        prop_A = args.prop_A or "0.3"
        prop_B = prop_A
        prop_secondary_inter = args.prop_inter2 or "3.0"
        prop_primary_inter = args.prop_inter or "30.0"
        default_filler = "300.0"

    prop_map = {
        0: prop_A, 1: prop_B, 
        secondary_inter_id: prop_secondary_inter,
        primary_inter_id: prop_primary_inter,
        filler_start_id: default_filler
    }
    
    # 2. Extract Basic Phase Fractions
    phase_stats = summarize_phase_fractions(
        final_grid, secondary_inter_id=secondary_inter_id, 
        primary_inter_id=primary_inter_id, filler_start_id=filler_start_id
    )
    
    # 3. Extract Structure Metrics
    print("  -> Computing connectivity and structure metrics...")
    structure_metrics = compute_structure_metrics(
        final_grid, primary_inter_id=primary_inter_id, 
        secondary_inter_id=secondary_inter_id, filler_start_id=filler_start_id
    )
    
    # Optional advanced morphology analysis via PoreSpy
    if args.advanced_metrics:
        adv = compute_advanced_metrics(
            final_grid, voxel_size, filler_start_id=filler_start_id, pbc_pad=args.pbc_pad
        )
        structure_metrics.update(adv)
        
    metric_fields = structure_metrics_to_csv_fields(structure_metrics)
    
    # 4. Generate Visualizations (VTI and 2D Thumbnail)
    if not args.skip_vti:
        vti_path = f"{args.import_path}.vti"
        export_visualization_vti(final_grid, vti_path, voxel_size=voxel_size)
        print(f"  -> Exported VTI: {vti_path}")
        
    slice_path = f"{args.import_path}_slice.png"
    save_thumbnail_png(final_grid, slice_path)

    # 5. Solver Execution
    chfem_time, puma_time = "", ""
    chfem_results, puma_results = [""] * 6, [""] * 6
    
    if args.physics_mode == 'permeability':
        print(f"  -> Binarizing grid for permeability mode (Void phases: {args.void_phases})")
        export_grid = np.zeros_like(final_grid)
        for vp in args.void_phases:
            export_grid[final_grid == vp] = 1
        export_prop_map = {0: "0.0", 1: "1.0"} 
    else:
        export_grid = final_grid
        export_prop_map = prop_map

    if args.solver != 'skip':
        # Re-export solver inputs to ensure metadata/properties are in sync
        export_chfem_inputs(
            export_grid, args.import_path, voxel_size=voxel_size, 
            physics_mode=args.physics_mode, prop_map=export_prop_map
        )
    
        # A. Run chfem solver
        if args.solver in ['chfem', 'both']:
            print(f"  -> Running chfem solver ({args.physics_mode})...")
            log_file = f"{args.import_path}_metrics.txt"
            chfem_cmd = ["chfem_exec", f"{args.import_path}.nf", f"{args.import_path}.raw", "-m", log_file]
            subprocess.run(chfem_cmd)
            
            res_diag, ctime = parse_chfem_log(log_file)
            if res_diag[0] != "":
                chfem_time, chfem_results = f"{ctime:.2f}", res_diag
                
        # B. Run PuMA solver
        if args.solver in ['puma', 'both']:
            print(f"  -> Running PuMA solver ({args.physics_mode})...")
            if args.physics_mode == 'mechanics':
                puma_res_raw, ptime = run_puma_elasticity(final_grid, voxel_size, prop_map)
                if puma_res_raw[0] is not None:
                    puma_time, puma_results = f"{ptime:.2f}", puma_res_raw
            elif args.physics_mode == 'permeability':
                puma_res_raw, ptime = run_puma_permeability(export_grid, voxel_size, solid_cutoff=(0, 0))
                if puma_res_raw[0] is not None:
                    puma_time, puma_results = f"{ptime:.2f}", puma_res_raw
            else:
                # Conduction solver requires single-scalar cond_map
                cond_map = {k: float(v.split()[0]) for k, v in prop_map.items()}
                pkx, pky, pkz, ptime = run_puma_laplace(
                    final_grid, voxel_size, args.physics_mode, cond_map
                )
                if pkx is not None:
                    puma_time = f"{ptime:.2f}"
                    puma_results = [pkx, pky, pkz, "", "", ""]

    # 6. Logging Results to CSV
    file_exists = os.path.isfile(args.csv_log)
    with open(args.csv_log, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Generate header for imported results log
            writer.writerow([
                "Basename", "Grid_Size", "Voxel_Size_m", "Source_Image", "Pattern", "PBC", "Mode",
                "Polymer_Frac", "Secondary_Inter_Frac", "Primary_Inter_Frac", "Filler_Frac",
                "Contact_Ratio", "Tunneling_Ratio", "Connectivity_Ratio",
                "Specific_Surface_Area", "Mean_Sphericity", "Local_Thickness_Mean", "Autocorrelation_Length",
                "chfem_Time_s", "chfem_Txx", "chfem_Tyy", "chfem_Tzz", "chfem_Tyz", "chfem_Tzx", "chfem_Txy",
                "puma_Time_s", "puma_Txx", "puma_Tyy", "puma_Tzz", "puma_Tyz", "puma_Tzx", "puma_Txy"
            ])
        
        writer.writerow([
            os.path.basename(args.import_path), 
            f"{grid_shape[2]}x{grid_shape[1]}x{grid_shape[0]}", 
            voxel_size,
            meta.get('source_image', 'N/A'),
            meta.get('interface_pattern', 'N/A'),
            meta.get('pbc_enforced', False),
            args.physics_mode,
            f"{phase_stats['polymer_a_fraction'] + phase_stats['polymer_b_fraction']:.4f}",
            f"{phase_stats['secondary_interface_fraction']:.4f}",
            f"{phase_stats['primary_interface_fraction']:.4f}",
            f"{phase_stats['filler_total_fraction']:.4f}",
            metric_fields["Contact_Ratio"], metric_fields["Tunneling_Ratio"], metric_fields["Connectivity_Ratio"],
            metric_fields.get("Specific_Surface_Area", ""), 
            metric_fields.get("Mean_Sphericity", ""),
            metric_fields.get("Local_Thickness_Mean", ""), 
            metric_fields.get("Autocorrelation_Length", ""),
            chfem_time, *chfem_results,
            puma_time, *puma_results
        ])

    print(f"--- Process Complete. Results saved to {args.csv_log} ---")

if __name__ == "__main__":
    main()
