import csv
import os
import shutil
import subprocess

import numpy as np

from micro_builder import compute_structure_metrics, export_chfem_inputs
from .io_csv import (
    ensure_structure_metric_columns,
    parse_chfem_log,
    parse_nf_properties,
    structure_metrics_to_csv_fields,
)
from .solver_puma import run_puma_elasticity, run_puma_laplace

def run_recalculation_mode(args):
    """Executes the recalculation pipeline using existing .raw and .nf files"""
    if not os.path.exists(args.csv_log):
        print(f"Error: CSV file '{args.csv_log}' not found for recalculation.")
        return

    # 1. Create a backup of the original CSV
    backup_path = args.csv_log + ".backup"
    shutil.copy2(args.csv_log, backup_path)
    print(f"--- Recalculation Mode Started ---")
    print(f"Backup created at: {backup_path}")

    # Fallback properties from command-line arguments (used if .nf is missing or new IDs found)
    if args.physics_mode == 'thermal':
        fallback_props = {0: "0.3", 1: "0.3", 2: "3.0", 3: "30.0", 4: "300.0"}
    elif args.physics_mode == 'electrical':
        fallback_props = {0: "1e-4", 1: "1e-4", 2: "1e-3", 3: "1e-1", 4: "1e4"}
    else:  # mechanics
        fallback_props = {0: "1.0 0.35", 1: "1.0 0.35", 2: "10.0 0.30", 3: "100.0 0.25", 4: "1000.0 0.20"}

    # Load the original CSV into memory.
    # The recalculation loop updates only refreshed values, then rewrites the full CSV after each row.
    with open(args.csv_log, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    header, rows = ensure_structure_metric_columns(header, rows)

    # Required metadata columns for identifying files and grid dimensions
    try:
        idx_basename = header.index("Basename")
        idx_grid_size = header.index("Grid_Size")
        idx_voxel_size = header.index("Voxel_Size_m")
        idx_mode = header.index("Mode")
    except ValueError as e:
        print(f"Error: CSV is missing required columns ({e})")
        return

    # 2. Iterate through each model in the CSV
    for row_idx, row in enumerate(rows):
        basename = row[idx_basename]
        grid_size_str = row[idx_grid_size]
        voxel_size = float(row[idx_voxel_size])
        mode = row[idx_mode]

        print(f"\nRecalculating: {basename}")

        raw_file = f"{basename}.raw"
        nf_file = f"{basename}.nf"

        if not os.path.exists(raw_file):
            print(f"  Skipping: {raw_file} not found.")
            continue

        # Restore grid to determine unique Phase IDs
        dim_parts = grid_size_str.split('x')
        nx, ny, nz = int(dim_parts[0]), int(dim_parts[1]), int(dim_parts[2])
        final_grid = np.fromfile(raw_file, dtype=np.uint8).reshape((nz, ny, nx))

        # 3. Resolve Property Map (Task 5 Fix: Preserve heterogeneity)
        prop_map = parse_nf_properties(nf_file)

        if args.overwrite_props or not prop_map:
            print("  Updating properties from command-line arguments...")
            cli_overrides = {0: args.prop_A, 1: args.prop_B, 2: args.prop_inter2, 3: args.prop_inter}
            unique_ids = np.unique(final_grid)

            for uid in unique_ids:
                # Priority 1: User-specified CLI override
                if uid in cli_overrides and cli_overrides[uid] is not None:
                    prop_map[uid] = cli_overrides[uid]
                # Priority 2: Keep existing .nf property if present (protects multiple filler IDs)
                elif uid in prop_map:
                    continue
                # Priority 3: Hard fallback for missing data
                else:
                    if uid in fallback_props:
                        prop_map[uid] = fallback_props[uid]
                    elif uid >= 4:
                        prop_map[uid] = fallback_props[4]

            # Re-export .nf for chfem with corrected properties
            export_chfem_inputs(final_grid, basename, voxel_size, mode, prop_map)

        # 4. Compute lightweight structure metrics directly from final_grid
        if args.skip_structure_metrics:
            structure_metrics = {
                'contact_ratio': 0.0, 'tunneling_ratio': 0.0, 'connectivity_ratio': 0.0,
                'n_contact_voxels': 0, 'n_tunnel_voxels': 0, 'n_filler_voxels': 0,
                'n_conductive_candidate_voxels': 0, 'n_largest_cluster_voxels': 0, 'n_conductive_clusters': 0,
            }
        else:
            structure_metrics = compute_structure_metrics(final_grid)
        metric_fields = structure_metrics_to_csv_fields(structure_metrics)

        # 5. Execute solvers and collect results (Task 4: Universal T-notation)
        chfem_time = ""
        chfem_results = [""] * 6
        puma_time = ""
        puma_results = [""] * 6

        if args.solver in ["chfem", "both"]:
            log_file = f"{basename}_metrics.txt"
            subprocess.run(["chfem_exec", nf_file, raw_file, "-m", log_file])
            res_diag, ctime = parse_chfem_log(log_file)
            if res_diag[0] != "":
                chfem_time, chfem_results = f"{ctime:.2f}", res_diag

        if args.solver in ["puma", "both"]:
            if mode == 'mechanics':
                puma_results, ptime = run_puma_elasticity(final_grid, voxel_size, prop_map)
                if puma_results[0] is not None:
                    puma_time = f"{ptime:.2f}"
            else:
                cond_map = {k: float(v.split()[0]) for k, v in prop_map.items()}
                pkx, pky, pkz, ptime = run_puma_laplace(final_grid, voxel_size, mode, cond_map)
                if pkx is not None:
                    puma_time = f"{ptime:.2f}"
                    puma_results = [pkx, pky, pkz, "", "", ""]

        # 6. Update the row with new metrics
        try:
            row[header.index("chfem_Time_s")] = chfem_time
            row[header.index("puma_Time_s")] = puma_time

            # Map results to universal Txx, Tyy, Tzz, Tyz, Tzx, Txy columns
            comp_suffixes = ["xx", "yy", "zz", "yz", "zx", "xy"]
            for i, suffix in enumerate(comp_suffixes):
                col_c = f"chfem_T{suffix}"
                col_p = f"puma_T{suffix}"
                if col_c in header:
                    row[header.index(col_c)] = chfem_results[i]
                if col_p in header:
                    row[header.index(col_p)] = puma_results[i]

            for col_name, value in metric_fields.items():
                if col_name in header:
                    row[header.index(col_name)] = value
        except ValueError:
            pass

        rows[row_idx] = row

        # Persist the current CSV state after each updated row.
        # This preserves progressive saving without rebuilding from an empty file.
        with open(args.csv_log, mode='w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerows(rows)
            out_f.flush()
            os.fsync(out_f.fileno())

    print(f"\n--- Recalculation Complete. Saved to {args.csv_log} ---")
