#!/usr/bin/env python3
"""
Microsimflow Utility: Real Image Importer
Converts 3D image stacks (TIFF/RAW) into 4-phase voxel models.
Uses purely mathematical Distance Transforms and Dual-Radii logic 
to eliminate "Volume Inflation" artifacts in tunneling interfaces.
Includes adjustable SNOW parameters to prevent fiber over-segmentation.
"""

import argparse
import numpy as np
import os
import json
from scipy.ndimage import (
    distance_transform_edt, maximum_filter, minimum_filter, generate_binary_structure
)

try:
    import porespy as ps
except ImportError:
    print("Error: 'porespy' is required for image import. Please install it.")
    exit(1)

def save_debug_slice(grid, filename, cmap='viridis'):
    """Helper to save the Z-mid slice of a 3D grid as a PNG for debugging."""
    try:
        import matplotlib.pyplot as plt
        z_mid = grid.shape[0] // 2
        plt.imsave(filename, grid[z_mid, :, :], cmap=cmap, origin='upper')
        print(f"     [Debug] Saved slice: {os.path.basename(filename)}")
    except ImportError:
        pass

def get_spherical_footprint(radius):
    """
    Generates a strict discrete sphere kernel using Euclidean distance.
    Replaces scipy's generate_binary_structure to prevent connectivity rank errors.
    """
    r = int(np.ceil(radius))
    z, y, x = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return (x**2 + y**2 + z**2) <= r**2

def apply_pbc_reflection(binary_grid):
    """
    Enforce Periodic Boundary Conditions (PBC) via 3D mirror reflection.
    Creates a virtual periodic structure with 8x the original volume.
    """
    print(f"  -> Enforcing PBC via 3D Mirror Reflection...")
    grid_x = np.concatenate([binary_grid, np.flip(binary_grid, axis=2)], axis=2)
    grid_y = np.concatenate([grid_x, np.flip(grid_x, axis=1)], axis=1)
    grid_z = np.concatenate([grid_y, np.flip(grid_y, axis=0)], axis=0)
    return grid_z

def extract_interfaces(binary_grid, pattern, tunnel_radius=2, tunnel_radius_thin=1, 
                       debug_dir=None, base_name="", snow_sigma=0.4, snow_r_max=4):
    """
    Partition filler network and extract Contact (ID:3) and Tunneling (ID:2) phases.
    Uses Dual-Radii logic to accurately isolate connective bottlenecks.
    """
    print(f"  -> Extracting interfaces using purely analytical distance fields...")
    
    # --- [Safety Constraint] Auto-correction to prevent mid-gap disconnection ---
    min_safe_thin = int(np.ceil(tunnel_radius / 2.0))
    if tunnel_radius_thin < min_safe_thin:
        print(f"     [Warning] tunnel_radius_thin ({tunnel_radius_thin}) risks mid-gap disconnection.")
        print(f"               Auto-adjusting to mathematically safe minimum: {min_safe_thin}")
        tunnel_radius_thin = min_safe_thin

    final_grid = np.zeros_like(binary_grid, dtype=np.uint8)
    
    # 1. SNOW algorithm to partition fillers
    print(f"     Running SNOW algorithm (sigma={snow_sigma}, r_max={snow_r_max})...")
    snow = ps.filters.snow_partitioning(binary_grid, sigma=snow_sigma, r_max=snow_r_max)
    regions = snow.regions 
    max_label = np.max(regions)
    
    if debug_dir: 
        save_debug_slice(regions, os.path.join(debug_dir, f"{base_name}_step1_regions.png"), cmap='prism')
    
    # Create an array where background (0) is invalidated for minimum filters
    regions_no_bg = np.where(regions == 0, max_label + 1, regions)
    
    # 2. Extract Primary Contacts (Internal & 1-voxel gaps)
    print("     Calculating Primary (Contact) interfaces...")
    # Use 26-neighborhood (3x3 cube) for direct contact detection
    struct_26n = generate_binary_structure(3, 3)
    
    max_c = maximum_filter(regions, footprint=struct_26n)
    min_c = minimum_filter(regions_no_bg, footprint=struct_26n)
    contact_junctions = (max_c != min_c) & (min_c <= max_label)
    
    internal_contact_mask = binary_grid & contact_junctions
    gap_contact_mask = (~binary_grid) & contact_junctions
    
    if debug_dir: 
        save_debug_slice(internal_contact_mask, os.path.join(debug_dir, f"{base_name}_step2a_internal_contacts.png"), cmap='gray')
        save_debug_slice(gap_contact_mask, os.path.join(debug_dir, f"{base_name}_step2b_gap_contacts.png"), cmap='gray')
    
    # 3. Extract Secondary Tunneling (Dual-Radii Logic)
    print(f"     Calculating Tunneling interfaces using Dual-Radii (thick={tunnel_radius}, thin={tunnel_radius_thin})...")
    dist_to_filler = distance_transform_edt(~binary_grid)
    
    ball_thick = get_spherical_footprint(tunnel_radius)
    ball_thin = get_spherical_footprint(tunnel_radius_thin)
    
    max_thick = maximum_filter(regions, footprint=ball_thick)
    min_thick = minimum_filter(regions_no_bg, footprint=ball_thick)
    
    # For the thin zone, we only need to know if ANY filler is present.
    # min_thin <= max_label implies there is at least one valid filler ID in the thin footprint.
    min_thin = minimum_filter(regions_no_bg, footprint=ball_thin)
    
    # --- Dual-Radii Logic Gates ---
    # Equivalent to RSA side: (n_thin >= 1) & (n_thick >= 2)
    has_thin = (min_thin <= max_label)
    has_multiple_thick = (max_thick != min_thick)
    
    tunnel_junctions = has_thin & has_multiple_thick
    
    # Apply distance threshold to maintain smooth spherical boundaries around the gap
    strict_tunnel_mask = (~binary_grid) & (dist_to_filler <= tunnel_radius) & tunnel_junctions
    
    # Exclude 1-voxel gaps that were already processed as Primary Contacts
    strict_tunnel_mask = strict_tunnel_mask & (~gap_contact_mask)
    
    if debug_dir: 
        save_debug_slice(strict_tunnel_mask, os.path.join(debug_dir, f"{base_name}_step3_strict_tunnel.png"), cmap='gray')

    # 4. Phase Assignment based on the requested pattern
    final_grid[binary_grid] = 4
    if pattern == 'erosion':
        final_grid[internal_contact_mask] = 3
        final_grid[strict_tunnel_mask] = 2
    elif pattern == 'dilation':
        final_grid[internal_contact_mask] = 3
        final_grid[gap_contact_mask] = 3
        final_grid[strict_tunnel_mask] = 2

    if debug_dir:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('viridis', 5)
            z_mid = final_grid.shape[0] // 2
            plt.imsave(os.path.join(debug_dir, f"{base_name}_step4_final_{pattern}.png"), 
                       final_grid[z_mid, :, :], cmap=cmap, vmin=-0.5, vmax=4.5, origin='upper')
        except ImportError: 
            pass

    return final_grid

def main():
    parser = argparse.ArgumentParser(description="Import real 3D images to Microsimflow format.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image (TIFF or RAW stack)")
    parser.add_argument("--format", type=str, choices=['tiff', 'raw'], required=True, help="Input file format")
    parser.add_argument("--raw_shape", type=int, nargs=3, help="Dimensions (Z Y X) required for RAW format")
    parser.add_argument("--voxel_size", type=float, required=True, help="Physical size of one voxel in meters")
    parser.add_argument("--threshold", type=float, default=128, help="Binarization threshold (Matrix < Threshold <= Filler)")
    parser.add_argument("--pattern", type=str, choices=['erosion', 'dilation'], default='dilation', help="Interface generation algorithm")
    parser.add_argument("--tunnel_radius", type=int, default=2, help="Radius for secondary tunneling interface in voxels")
    parser.add_argument("--tunnel_radius_thin", type=int, default=1, help="Thin radius for Dual-Radii logic. Limits the bulkiness of tunneling bridges.")
    parser.add_argument("--enforce_pbc", action="store_true", help="Apply 3D mirror reflection to enforce PBC")
    
    # Anti-over-segmentation knobs
    parser.add_argument("--snow_sigma", type=float, default=0.4, help="Gaussian blur sigma for SNOW. Increase to smooth surfaces.")
    parser.add_argument("--snow_r_max", type=int, default=4, help="Max radius to search for peaks in SNOW. Increase to merge close peaks.")
    
    parser.add_argument("--out_dir", type=str, default="imported_models", help="Output directory for .raw and .nf files")
    parser.add_argument("--save_debug_slices", action="store_true", help="Save mid-Z slices for debugging internal masks")
    
    args = parser.parse_args()
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    if args.enforce_pbc: 
        base_name += "_pbc"
        
    debug_dir = os.path.join(args.out_dir, "debug_slices") if args.save_debug_slices else None
    if debug_dir: 
        os.makedirs(debug_dir, exist_ok=True)
    
    # 1. Load 3D Image Data
    if args.format == 'raw':
        if not args.raw_shape:
            raise ValueError("--raw_shape is required for RAW format")
        print(f"Loading RAW image with shape {args.raw_shape}...")
        img = np.fromfile(args.input, dtype=np.uint8).reshape(args.raw_shape)
    else:
        import tifffile
        print(f"Loading TIFF stack from {args.input}...")
        img = tifffile.imread(args.input)
        
    # 2. Binarization (Matrix: False, Filler: True)
    print(f"Applying threshold: {args.threshold}")
    binary_grid = (img >= args.threshold).astype(bool)
    
    # 3. PBC Mirror Reflection
    if args.enforce_pbc: 
        binary_grid = apply_pbc_reflection(binary_grid)
        
    # 4. Extract Interfaces and Generate Multi-phase Grid
    final_grid = extract_interfaces(
        binary_grid, 
        pattern=args.pattern, 
        tunnel_radius=args.tunnel_radius, 
        tunnel_radius_thin=args.tunnel_radius_thin, 
        debug_dir=debug_dir, 
        base_name=base_name, 
        snow_sigma=args.snow_sigma, 
        snow_r_max=args.snow_r_max
    )

    # 5. Export results
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, f"{base_name}_final.raw")
    nf_path = os.path.join(args.out_dir, f"{base_name}_meta.nf")
    
    print(f"  -> Saving model binary to {raw_path}")
    final_grid.tofile(raw_path)
    
    meta_data = {
        "source_image": args.input, 
        "grid_size": list(final_grid.shape),
        "voxel_size_m": args.voxel_size, 
        "pbc_enforced": args.enforce_pbc,
        "interface_pattern": args.pattern, 
        "tunnel_radius": args.tunnel_radius,
        "tunnel_radius_thin": args.tunnel_radius_thin,
        "snow_sigma": args.snow_sigma, 
        "snow_r_max": args.snow_r_max
    }
    with open(nf_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
        
    print(f"Import process successfully completed.")

if __name__ == "__main__":
    main()
