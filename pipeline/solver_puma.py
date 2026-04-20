import time

import numpy as np

# PuMA backend settings
# Keep these as in-script settings instead of exposing additional CLI arguments.
PUMA_ELASTICITY_METHOD = "fe"
PUMA_ELASTICITY_FV_SOLVER = "bicgstab"
PUMA_ELASTICITY_FE_SOLVER = "minres"
PUMA_ELASTICITY_SIDE_BC = "p"
PUMA_ELASTICITY_MAXITER = 20000

def get_puma_elasticity_namespace(puma):
    """Select the PuMA namespace that exposes elasticity helpers."""
    if hasattr(puma, 'experimental') and hasattr(puma.experimental, 'ElasticityMap'):
        return puma.experimental
    return puma


def parse_mechanics_property(prop_val):
    """Parse a mechanics property string as Young's modulus and Poisson's ratio."""
    parts = str(prop_val).split()
    if len(parts) != 2:
        raise ValueError(f"Mechanics property must be specified as 'E nu': {prop_val}")
    return float(parts[0]), float(parts[1])


def run_puma_laplace(final_grid, voxel_size, physics_mode, cond_map):
    """Solve the Laplace equation (thermal/electrical conduction) using PuMA's Python API"""
    try:
        import pumapy as puma
    except ImportError:
        print("Error: pumapy module not found. Cannot run PuMA solver.")
        return None, None, None, 0.0

    if physics_mode == 'mechanics':
        print("Mechanics mode requires the PuMA elasticity wrapper, not the Laplace wrapper. Skipping PuMA.")
        return None, None, None, 0.0

    # Transpose dimensions: NumPy (Z, Y, X) -> PuMA Workspace (X, Y, Z)
    ws = puma.Workspace.from_array(final_grid.transpose(2, 1, 0))
    ws.voxel_length = voxel_size

    puma_cond_map = puma.IsotropicConductivityMap()
    for phase_id, cond_val in cond_map.items():
        puma_cond_map.add_material((int(phase_id), int(phase_id)), float(cond_val))
    puma_side_bc = "p"
    puma_solver_type = "cg"
    puma_maxiter = 20000

    print("\n--- Running PuMA Solver ---")
    t0 = time.time()

    try:
        # Compute for each XYZ direction (specify periodic boundary conditions with side_bc='p')
        print("Computing X direction...")
        res_x = puma.compute_thermal_conductivity(
            ws, puma_cond_map, direction='x',
            side_bc=puma_side_bc, solver_type=puma_solver_type, maxiter=puma_maxiter
        )
        k_eff_x = res_x[0] if isinstance(res_x, tuple) else res_x
        txx = k_eff_x[0]

        print("Computing Y direction...")
        res_y = puma.compute_thermal_conductivity(
            ws, puma_cond_map, direction='y',
            side_bc=puma_side_bc, solver_type=puma_solver_type, maxiter=puma_maxiter
        )
        k_eff_y = res_y[0] if isinstance(res_y, tuple) else res_y
        tyy = k_eff_y[1]

        print("Computing Z direction...")
        res_z = puma.compute_thermal_conductivity(
            ws, puma_cond_map, direction='z',
            side_bc=puma_side_bc, solver_type=puma_solver_type, maxiter=puma_maxiter
        )
        k_eff_z = res_z[0] if isinstance(res_z, tuple) else res_z
        tzz = k_eff_z[2]

        total_time = time.time() - t0
        print(f"PuMA computation completed in {total_time:.2f}s")

        return txx, tyy, tzz, total_time

    except Exception as e:
        print(f"PuMA encountered an error during computation: {e}")
        return None, None, None, 0.0


def run_puma_elasticity(final_grid, voxel_size, prop_map):
    """Solve homogenized elasticity using PuMA's Python API."""
    try:
        import pumapy as puma
    except ImportError:
        print("Error: pumapy module not found. Cannot run PuMA solver.")
        return [None] * 6, 0.0


    puma_elast = get_puma_elasticity_namespace(puma)

    grid_c = final_grid.transpose(2, 1, 0).astype(np.uint16).copy(order='C')
    ws = puma.Workspace.from_array(grid_c)
    ws.voxel_length = voxel_size

    puma_elast_map = puma_elast.ElasticityMap()
    ws_unique_ids = np.unique(ws.matrix)
    print(f"\n[PuMA Elasticity] Workspace unique IDs: {ws_unique_ids}")

    for uid in ws_unique_ids:
        uid_int = int(uid)

        if uid_int not in prop_map:
            raise KeyError(
                f"Phase ID {uid_int} is missing in prop_map. "
                f"Please define the mechanics property as 'E nu' for this phase before running PuMA elasticity."
            )

        prop_val = prop_map[uid_int]
        young_modulus, poisson_ratio = parse_mechanics_property(prop_val)
        puma_elast_map.add_isotropic_material((uid_int, uid_int), young_modulus, poisson_ratio)
        print(f"  -> Mapped ID {uid_int}: E={young_modulus:.2f}, nu={poisson_ratio:.3f}")

    directions = ['x', 'y', 'z', 'yz', 'xz', 'xy']
    effective_matrix = np.zeros((6, 6), dtype=float)
    total_time = 0.0

    solver_type = PUMA_ELASTICITY_FE_SOLVER if PUMA_ELASTICITY_METHOD == 'fe' else PUMA_ELASTICITY_FV_SOLVER

    print("\n--- Running PuMA Elasticity Solver ---")
    print(f"Elasticity backend: {PUMA_ELASTICITY_METHOD}")

    try:
        t0 = time.time()
        for idx, direction in enumerate(directions):
            print(f"Computing {direction} direction...")

            if PUMA_ELASTICITY_METHOD == 'fe':
                res = puma_elast.compute_elasticity(
                    ws, puma_elast_map, direction=direction, method='fe',
                    solver_type=solver_type, maxiter=PUMA_ELASTICITY_MAXITER
                )
            else:
                res = puma_elast.compute_elasticity(
                    ws, puma_elast_map, direction=direction,
                    side_bc=PUMA_ELASTICITY_SIDE_BC,
                    solver_type=solver_type, maxiter=PUMA_ELASTICITY_MAXITER
                )

            if isinstance(res, tuple):
                ceff = res[0]
            else:
                ceff = res

            effective_matrix[:, idx] = np.asarray(ceff, dtype=float)

        total_time = time.time() - t0

        results = [
            float(effective_matrix[0, 0]),
            float(effective_matrix[1, 1]),
            float(effective_matrix[2, 2]),
            float(effective_matrix[3, 3]),
            float(effective_matrix[4, 4]),
            float(effective_matrix[5, 5]),
        ]

        print("PuMA elasticity computation completed")
        return results, total_time

    except Exception as e:
        print(f"PuMA encountered an error during elasticity computation: {e}")
        return [None] * 6, 0.0
