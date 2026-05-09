#!/usr/bin/env python3
"""
Microsimflow Utility: Real Image Importer
Converts 3D TIFF stacks or RAW binaries into 4-phase voxel models (Matrix, Tunnel, Contact, Filler).
Supports PBC enforcement via 3D mirror reflection and dual-pattern interface extraction.
"""

import argparse
import numpy as np
import os
import json
from scipy.ndimage import (
    distance_transform_edt, maximum_filter, minimum_filter, 
    binary_dilation, generate_binary_structure
)

try:
    import porespy as ps
except ImportError:
    print("Error: 'porespy' is required for image import. Please install it.")
    exit(1)

def apply_pbc_reflection(binary_grid):
    """
    Enforce Periodic Boundary Conditions (PBC) by 3D mirror reflection.
    Creates a virtual periodic structure with 8x the original volume.
    """
    print(f"  -> Enforcing PBC via 3D Mirror Reflection...")
    # Reflect and concatenate along Z, Y, and X axes
    grid_x = np.concatenate([binary_grid, np.flip(binary_grid, axis=2)], axis=2)
    grid_y = np.concatenate([grid_x, np.flip(grid_x, axis=1)], axis=1)
    grid_z = np.concatenate([grid_y, np.flip(grid_y, axis=0)], axis=0)
    print(f"     Reflected shape: {grid_z.shape}")
    return grid_z

def extract_interfaces(binary_grid, pattern, tunnel_radius=2):
    """
    Partition filler network and extract Contact (ID:3) and Tunneling (ID:2) phases.
    Pattern 'erosion': Minimizes contact area to represent high contact resistance.
    Pattern 'dilation': Heals voxelization artifacts to ensure robust connectivity.
    """
    print(f"  -> Extracting interfaces using pattern: '{pattern}'")
    final_grid = np.zeros_like(binary_grid, dtype=np.uint8)
    
    # 1. Partition filler network into individual particles using SNOW algorithm
    print("     Running SNOW algorithm to partition fillers...")
    snow = ps.filters.snow_partitioning(binary_grid)
    regions = snow.regions  # 0: matrix, 1~N: individual fillers
    
    # --- High-speed boundary detection algorithm ---
    # Replace background 0 with a large value to avoid interference with minimum_filter
    max_label = np.max(regions)
    regions_no_bg = np.where(regions == 0, max_label + 1, regions)
    
    struct_contact = generate_binary_structure(3, 1) # 6-neighborhood
    max_f = maximum_filter(regions, footprint=struct_contact)
    min_f = minimum_filter(regions_no_bg, footprint=struct_contact)
    
    # Identify voxels where different filler IDs meet (Contact Interface)
    internal_contact_mask = binary_grid & (max_f != min_f)
    
    # 2. Identify Tunneling (Secondary) interface candidates
    print("     Calculating Tunneling (Secondary) interfaces...")
    dist_to_filler = distance_transform_edt(~binary_grid)
    
    # Secondary interface exists in matrix space within tunnel_radius of multiple fillers
    struct_tunnel = generate_binary_structure(3, tunnel_radius)
    max_t = maximum_filter(regions, footprint=struct_tunnel)
    min_t = minimum_filter(regions_no_bg, footprint=struct_tunnel)
    
    strict_tunnel_mask = (~binary_grid) & (dist_to_filler <= tunnel_radius) & (max_t != min_t)

    # 3. Phase assignment based on requested pattern
    if pattern == 'erosion':
        # Erosion: Convert contact points into ID 3 within the existing filler body
        print("     Applying 'erosion' (Bottlenecking contacts)...")
        final_grid[binary_grid] = 4
        final_grid[internal_contact_mask] = 3
        final_grid[strict_tunnel_mask] = 2

    elif pattern == 'dilation':
        # Dilation: Expand contact points by 1 voxel to bridge gaps between discrete fillers
        print("     Applying 'dilation' (Healing voxel defects)...")
        # scipy.ndimage.binary_dilation uses 'structure' instead of 'footprint'
        dilated_contact = binary_dilation(internal_contact_mask, structure=struct_contact)
        
        final_grid[binary_grid] = 4
        final_grid[dilated_contact] = 3
        # Ensure tunnel phase does not overwrite filler/contact phases
        final_grid[strict_tunnel_mask & (final_grid == 0)] = 2

    return final_grid

def main():
    parser = argparse.ArgumentParser(description="Import real 3D images (TIFF/RAW) to Microsimflow pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image (TIFF or RAW stack)")
    parser.add_argument("--format", type=str, choices=['tiff', 'raw'], required=True, help="Input file format")
    parser.add_argument("--raw_shape", type=int, nargs=3, help="Dimensions (Z Y X) required for RAW format")
    parser.add_argument("--voxel_size", type=float, required=True, help="Physical size of one voxel in meters")
    parser.add_argument("--threshold", type=float, default=128, help="Binarization threshold (Matrix < Threshold <= Filler)")
    parser.add_argument("--pattern", type=str, choices=['erosion', 'dilation'], default='dilation', help="Interface generation algorithm")
    parser.add_argument("--tunnel_radius", type=int, default=2, help="Radius for secondary tunneling interface in voxels")
    parser.add_argument("--enforce_pbc", action="store_true", help="Apply 3D mirror reflection to enforce PBC (Increases volume 8x)")
    parser.add_argument("--out_dir", type=str, default="imported_models", help="Output directory for .raw and .nf files")
    
    args = parser.parse_args()
    
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
    
    # 3. PBC Mirror Reflection (Must be applied before interface extraction)
    if args.enforce_pbc:
        binary_grid = apply_pbc_reflection(binary_grid)
        
    # 4. Extract Interfaces and Generate Multi-phase Grid
    final_grid = extract_interfaces(binary_grid, pattern=args.pattern, tunnel_radius=args.tunnel_radius)
    
    # 5. Export results
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    if args.enforce_pbc:
        base_name += "_pbc"
        
    raw_path = os.path.join(args.out_dir, f"{base_name}_final.raw")
    nf_path = os.path.join(args.out_dir, f"{base_name}_meta.nf")
    
    print(f"  -> Saving model binary to {raw_path}")
    final_grid.tofile(raw_path)
    
    # Metadata for run_imported.py
    meta_data = {
        "source_image": args.input,
        "grid_size": list(final_grid.shape),
        "voxel_size_m": args.voxel_size,
        "pbc_enforced": args.enforce_pbc,
        "interface_pattern": args.pattern,
        "tunnel_radius": args.tunnel_radius
    }
    with open(nf_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
        
    print(f"Import process successfully completed.")

if __name__ == "__main__":
    main()