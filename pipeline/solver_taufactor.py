import time
import numpy as np

def run_taufactor(final_grid, prop_map):
    """
    Compute tortuosity factor (tau) and effective diffusivity (D_eff)
    using taufactor's PeriodicMultiPhaseSolver.
    """
    try:
        import taufactor as tau
    except ImportError:
        print("Warning: taufactor not installed. Skipping tau metrics.")
        return {}

    print("\n--- Running taufactor (PeriodicMultiPhaseSolver) ---")
    t0 = time.time()
    
    # Parse conductivities from prop_map (takes the first float value, safe for Mechanics "E nu" strings)
    cond = {}
    for k, v in prop_map.items():
        try:
            cond[int(k)] = float(str(v).split()[0])
        except (ValueError, TypeError):
            cond[int(k)] = 1e-4

    unique_ids = np.unique(final_grid)
    valid_cond = {k: cond[k] for k in unique_ids if k in cond}
    
    print(f"  -> Extracted conductivities for taufactor: {valid_cond}")

    metrics = {}
    
    # taufactor solves along axis 0 by default.
    # In NumPy (Z, Y, X), Z is axis 0, Y is axis 1, X is axis 2.
    # np.ascontiguousarray is used to ensure memory safety for PyTorch backend.
    directions = {
        'Z': final_grid,
        'Y': np.ascontiguousarray(np.swapaxes(final_grid, 0, 1)),
        'X': np.ascontiguousarray(np.swapaxes(final_grid, 0, 2))
    }
    
    for dir_name, img in directions.items():
        print(f"  -> Computing {dir_name}-direction...")
        try:
            solver = tau.PeriodicMultiPhaseSolver(img, cond=valid_cond, iter_limit=10000)
            solver.solve(verbose=False)
            metrics[f'tau_{dir_name}'] = getattr(solver, 'tau', "")
            metrics[f'D_eff_{dir_name}'] = getattr(solver, 'D_eff', "")
        except Exception as e:
            print(f"     ! taufactor error in {dir_name}: {e}")
            metrics[f'tau_{dir_name}'] = ""
            metrics[f'D_eff_{dir_name}'] = ""

    total_time = time.time() - t0
    print(f"  -> taufactor completed in {total_time:.2f}s")
    metrics['tau_Time_s'] = total_time
    
    return metrics
