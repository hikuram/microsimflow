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
    
    # Parse conductivities directly from prop_map without any modification or cutoff
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
    # Swapping axes to compute Z (axis 0), Y (axis 1), and X (axis 2).
    directions = {
        'Z': final_grid,
        'Y': np.ascontiguousarray(np.swapaxes(final_grid, 0, 1)),
        'X': np.ascontiguousarray(np.swapaxes(final_grid, 0, 2))
    }
    
    for dir_name, img in directions.items():
        print(f"  -> Computing {dir_name}-direction...")
        try:
            # Create solver instance with current latest API (diffusivities arg) 
            solver = tau.PeriodicMultiPhaseSolver(img, diffusivities=valid_cond)
            # Run solver with specified iteration limit 
            solver.solve(iter_limit=10000, verbose=False)
            
            # Retrieve results; defaults to empty string if attributes are missing
            tau_val = getattr(solver, 'tau', "")
            deff_val = getattr(solver, 'D_eff', "")
            
            # Convert NumPy array results to scalars if batch size is 1 
            t_scalar = tau_val.item() if isinstance(tau_val, np.ndarray) and tau_val.size == 1 else tau_val
            d_scalar = deff_val.item() if isinstance(deff_val, np.ndarray) and deff_val.size == 1 else deff_val
            
            # If no percolating path exists, taufactor may return inf or nan. 
            # These are converted to empty strings for stable CSV logging.
            if t_scalar is None or (isinstance(t_scalar, (float, int)) and (np.isinf(t_scalar) or np.isnan(t_scalar))):
                t_scalar = ""
            if d_scalar is None or (isinstance(d_scalar, (float, int)) and (np.isinf(d_scalar) or np.isnan(d_scalar))):
                d_scalar = ""
                
            metrics[f'tau_{dir_name}'] = t_scalar
            metrics[f'D_eff_{dir_name}'] = d_scalar
            
        except Exception as e:
            # Fallback to empty string for disconnected or failed paths
            print(f"     ! taufactor error in {dir_name} (likely no percolating path): {e}")
            metrics[f'tau_{dir_name}'] = ""
            metrics[f'D_eff_{dir_name}'] = ""

    total_time = time.time() - t0
    print(f"  -> taufactor completed in {total_time:.2f}s")
    metrics['tau_Time_s'] = total_time
    
    return metrics
    
