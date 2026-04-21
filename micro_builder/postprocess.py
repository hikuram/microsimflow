import numpy as np
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt, generate_binary_structure
from numba import njit

# =========================================================
# PBC-Aware High-Performance Union-Find (Numba JIT)
# =========================================================

@njit
def _find(i, parent):
    """Path compression Find operation for Union-Find."""
    root = i
    while parent[root] != root:
        root = parent[root]
    
    # Path compression
    curr = i
    while curr != root:
        nxt = parent[curr]
        parent[curr] = root
        curr = nxt
    return root

@njit
def _union(i, j, parent):
    """Union operation for Union-Find."""
    root_i = _find(i, parent)
    root_j = _find(j, parent)
    if root_i != root_j:
        parent[root_i] = root_j

@njit
def label_pbc_6n_fast(mask):
    """
    High-performance 6-neighbor connected component labeling with strict 
    Periodic Boundary Condition (PBC) support using Union-Find.
    
    Returns:
        tuple: (total_number_of_clusters, size_of_the_largest_cluster)
    """
    Z, Y, X = mask.shape
    N = Z * Y * X
    parent = np.full(N, -1, dtype=np.int32)
    
    # 1. Initialization: Assign a self-referencing ID to each conductive voxel
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if mask[z, y, x]:
                    idx = z * Y * X + y * X + x
                    parent[idx] = idx
                    
    # 2. Union Phase: Check only +Z, +Y, +X directions with PBC wrapping.
    # Due to symmetry, forward-checking is sufficient to build the full 6-n graph.
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if mask[z, y, x]:
                    idx = z * Y * X + y * X + x
                    
                    # +Z direction (PBC Wrap)
                    nz = (z + 1) % Z
                    if mask[nz, y, x]:
                        n_idx = nz * Y * X + y * X + x
                        _union(idx, n_idx, parent)
                        
                    # +Y direction (PBC Wrap)
                    ny = (y + 1) % Y
                    if mask[z, ny, x]:
                        n_idx = z * Y * X + ny * X + x
                        _union(idx, n_idx, parent)
                        
                    # +X direction (PBC Wrap)
                    nx = (x + 1) % X
                    if mask[z, y, nx]:
                        n_idx = z * Y * X + y * X + nx
                        _union(idx, n_idx, parent)
                        
    # 3. Aggregation Phase: Calculate cluster sizes
    sizes = np.zeros(N, dtype=np.int32)
    largest_cluster = 0
    num_clusters = 0
    
    for i in range(N):
        if parent[i] != -1:
            root = _find(i, parent)
            if sizes[root] == 0:
                num_clusters += 1
            sizes[root] += 1
            if sizes[root] > largest_cluster:
                largest_cluster = sizes[root]
                
    return num_clusters, largest_cluster


# =========================================================
# E. Final Structure Integration, Cleanup, and Aggregation
# =========================================================

def _make_6n_kernel():
    """Create a 3D convolution kernel for 6-neighborhood (cross shape)."""
    kernel = np.zeros((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 0] = 1
    kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = 1
    kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = 1
    kernel[2, 1, 1] = 1
    return kernel

_NEIGHBOR6_KERNEL = _make_6n_kernel()

def _remove_spikes_6n(mask, min_neighbors=2):
    """
    Remove burr/spike voxels using 6-neighbor connectivity.
    A voxel is removed if it has fewer than `min_neighbors` within the mask.
    """
    if not np.any(mask):
        return mask
    neighbor_count = convolve(mask.astype(np.uint8), _NEIGHBOR6_KERNEL, mode="wrap")
    return mask & (neighbor_count >= min_neighbors)

def _fill_polymer_slivers(final_grid, target_id=2, filler_start_id=4):
    """
    Convert isolated polymer voxels into interface elements if they are 
    directly sandwiched between a filler and an existing interface.
    """
    polymer_mask = final_grid < 2
    filler_mask = final_grid >= filler_start_id
    interface_mask = final_grid == target_id

    filler_nb = convolve(filler_mask.astype(np.uint8), _NEIGHBOR6_KERNEL, mode="wrap") > 0
    interface_nb = convolve(interface_mask.astype(np.uint8), _NEIGHBOR6_KERNEL, mode="wrap") > 0

    sliver_mask = polymer_mask & filler_nb & interface_nb
    if np.any(sliver_mask):
        final_grid[sliver_mask] = target_id

    return final_grid

def _cleanup_small_components(mask, min_component_size=2):
    """
    Remove tiny isolated connected components from a boolean mask 
    using the PBC-aware Union-Find algorithm.
    """
    if not np.any(mask):
        return mask

    # Note: For cleanup of trivial spikes, strict PBC is less critical, 
    # but using a generic approach ensures stability. Since we dropped scipy.label,
    # we implement a lightweight boolean filter here.
    neighbor_count = convolve(mask.astype(np.uint8), _NEIGHBOR6_KERNEL, mode="wrap")
    return mask & (neighbor_count >= min_component_size - 1)

def finalize_microstructure(comp_grid, tpms_grid, shell_count_grid=None, physics_mode='thermal', 
                            primary_inter_id=3, secondary_inter_id=3, filler_start_id=4,
                            sliver_fill_iters=1, spike_min_neighbors=2, min_interface_component_size=2,
                            contact_radius=1):
    """
    Integrate background matrix and filler phases.
    Evaluates overlaps to rigorously separate Primary interfaces (direct contact) 
    from Secondary interfaces (tunneling/Kapitza bridges) based on the target physics.
    """
    final_grid = np.where(comp_grid > 0, comp_grid, tpms_grid).astype(np.uint8)

    if physics_mode in ['electrical', 'mechanics'] and shell_count_grid is not None:
        pure_shell_counts = shell_count_grid & 127
        tunnel_mask = (pure_shell_counts >= 2) & (final_grid < filler_start_id)
        final_grid[tunnel_mask] = primary_inter_id

    # --- Common Interface Cleanup ---
    # 1) Polymer sliver fill
    for _ in range(sliver_fill_iters):
        before = np.count_nonzero(final_grid == primary_inter_id)
        final_grid = _fill_polymer_slivers(final_grid, target_id=primary_inter_id, filler_start_id=filler_start_id)
        after = np.count_nonzero(final_grid == primary_inter_id)
        if after == before:
            break

    # 2) Spike removal on interface only
    original_interface_mask = (final_grid == primary_inter_id)
    interface_mask = _remove_spikes_6n(original_interface_mask, min_neighbors=spike_min_neighbors)

    # 3) Small component cleanup
    interface_mask = _cleanup_small_components(interface_mask, min_component_size=min_interface_component_size)

    # Identify voxels removed during cleanup
    removed_interface = original_interface_mask & (~interface_mask)

    # Rebuild grid based on physics mode semantics
    if physics_mode in ['electrical', 'mechanics']:
        writable_mask = (final_grid < filler_start_id)
        final_grid[removed_interface & writable_mask] = tpms_grid[removed_interface & writable_mask]
        
        unified_interface_mask = (final_grid == primary_inter_id)
        filler_mask = (final_grid >= filler_start_id)
        
        if np.any(unified_interface_mask) and np.any(filler_mask):
            struct_1voxel = generate_binary_structure(3, 1)
            dilated_filler = binary_dilation(filler_mask, structure=struct_1voxel, iterations=contact_radius)
            
            secondary_mask = unified_interface_mask & (~dilated_filler)
            final_grid[secondary_mask] = secondary_inter_id

    elif physics_mode == 'thermal':
        # Revert removed interface voxels to the prevalent adjacent filler ID
        if np.any(removed_interface):
            filler_ids = np.unique(final_grid[final_grid >= filler_start_id])
            if len(filler_ids) == 0:
                final_grid[removed_interface] = filler_start_id
            elif len(filler_ids) == 1:
                final_grid[removed_interface] = filler_ids[0]
            else:
                counts = []
                for fid in filler_ids:
                    c = convolve((final_grid == fid).astype(np.uint8), _NEIGHBOR6_KERNEL, mode="wrap")
                    counts.append(c[removed_interface])
                counts = np.array(counts)
                best_idx = np.argmax(counts, axis=0)
                final_grid[removed_interface] = filler_ids[best_idx]

        # --- Evaluate Primary and Secondary Interfaces using Bitwise Flags ---
        if shell_count_grid is not None:
            # 1. Primary Interface (Core Overlap)
            # The overlap flag is stored in the highest bit (128).
            overlap_mask = (shell_count_grid & 128) > 0
            
            if np.any(overlap_mask):
                max_contact_thickness = 1.5 
                
                # --- PBC Padding for Accurate Distance Transform ---
                # Wrap the domain in a torus-like manner before computing the distance 
                # to prevent boundary discontinuities.
                pad_w = int(np.ceil(max_contact_thickness)) + 1
                padded_overlap = np.pad(overlap_mask, pad_width=pad_w, mode='wrap')
                
                padded_distance = distance_transform_edt(padded_overlap)
                
                # Crop back to original dimensions
                distance_grid = padded_distance[pad_w:-pad_w, pad_w:-pad_w, pad_w:-pad_w]
                # -------------------------------------------------

                valid_primary_mask = overlap_mask & (distance_grid <= max_contact_thickness)
                final_grid[valid_primary_mask] = primary_inter_id

            # 2. Secondary Interface (Kapitza Bridge / Tunneling)
            # Shell counts are stored in the lower 7 bits.
            pure_shell_counts = shell_count_grid & 127
            
            raw_kapitza_mask = (final_grid < 2) & (pure_shell_counts >= 2)
            cleaned_kapitza_mask = raw_kapitza_mask.copy()
            cleaned_kapitza_mask = _remove_spikes_6n(cleaned_kapitza_mask, min_neighbors=spike_min_neighbors)
            cleaned_kapitza_mask = _cleanup_small_components(cleaned_kapitza_mask, min_component_size=min_interface_component_size)
            
            final_grid[cleaned_kapitza_mask] = secondary_inter_id

    return final_grid

def summarize_phase_fractions(final_grid, primary_inter_id=3, secondary_inter_id=3, filler_start_id=4):
    """Aggregate the volume fraction of each structural phase."""
    total_voxels = final_grid.size
    max_id = final_grid.max()
    counts = np.bincount(final_grid.ravel(), minlength=max(filler_start_id, max_id + 1))

    stats = {
        'polymer_a_fraction': counts[0] / total_voxels,
        'polymer_b_fraction': counts[1] / total_voxels,
        'primary_interface_fraction': counts[primary_inter_id] / total_voxels,
        'secondary_interface_fraction': counts[secondary_inter_id] / total_voxels if len(counts) > secondary_inter_id else 0.0,
    }

    filler_total = 0
    for i in range(filler_start_id, max_id + 1):
        stats[f'filler_id{i}_fraction'] = counts[i] / total_voxels
        filler_total += counts[i] / total_voxels

    stats['filler_total_fraction'] = filler_total
    stats['polymer_fraction'] = (counts[0] + counts[1]) / total_voxels
    return stats


def compute_structure_metrics(final_grid, primary_inter_id=3, secondary_inter_id=2, filler_start_id=4):
    """
    Compute conductive structure descriptors utilizing the Numba JIT PBC-aware graph algorithm.

    Definitions:
    - contact_ratio: primary interface fraction normalized by total filler voxels.
    - tunneling_ratio: secondary interface fraction normalized by total filler voxels.
    - connectivity_ratio: largest 6-neighbor connected conductive cluster normalized by
      all conductive candidate voxels (filler + primary + secondary). Represents the true percolation threshold.
    """
    max_id = int(final_grid.max())
    counts = np.bincount(final_grid.ravel(), minlength=max(filler_start_id, max_id + 1))

    filler_voxels = int(counts[filler_start_id:].sum()) if len(counts) > filler_start_id else 0
    primary_voxels = int(counts[primary_inter_id]) if len(counts) > primary_inter_id else 0
    secondary_voxels = int(counts[secondary_inter_id]) if len(counts) > secondary_inter_id else 0

    conductive_mask = (final_grid >= filler_start_id)
    conductive_mask |= (final_grid == primary_inter_id)
    if secondary_inter_id != primary_inter_id:
        conductive_mask |= (final_grid == secondary_inter_id)

    conductive_candidate_voxels = int(np.count_nonzero(conductive_mask))
    largest_cluster_voxels = 0
    num_conductive_clusters = 0

    if conductive_candidate_voxels > 0:
        # Utilize the custom PBC-aware Union-Find algorithm instead of scipy.ndimage.label
        num_conductive_clusters, largest_cluster_voxels = label_pbc_6n_fast(conductive_mask)

    filler_denom = float(filler_voxels) if filler_voxels > 0 else 0.0
    conductive_denom = float(conductive_candidate_voxels) if conductive_candidate_voxels > 0 else 0.0

    return {
        'contact_ratio': (primary_voxels / filler_denom) if filler_denom > 0 else 0.0,
        'tunneling_ratio': (secondary_voxels / filler_denom) if filler_denom > 0 else 0.0,
        'connectivity_ratio': (largest_cluster_voxels / conductive_denom) if conductive_denom > 0 else 0.0,
        'n_contact_voxels': primary_voxels,
        'n_tunnel_voxels': secondary_voxels,
        'n_filler_voxels': filler_voxels,
        'n_conductive_candidate_voxels': conductive_candidate_voxels,
        'n_largest_cluster_voxels': largest_cluster_voxels,
        'n_conductive_clusters': int(num_conductive_clusters),
    }

def export_visualization_vti(final_grid, filename="microstructure.vti", voxel_size=1e-8, metadata=None):
    """
    Export the 3D grid to VTI format optimized for ParaView visualization.
    Embeds physical dimensions and metric metadata into Field Data for HUD overlay.
    """
    import pyvista as pv

    grid = pv.ImageData()
    
    # Map NumPy ordering (Z, Y, X) to PyVista spatial ordering (X, Y, Z)
    nz, ny, nx = final_grid.shape
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    
    # Assign physical scale
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.origin = (0.0, 0.0, 0.0)
    
    # Store phase ID in cell_data
    grid.cell_data["Phase"] = final_grid.flatten(order="C")
    
    # Embed tracking metadata
    if metadata:
        for key, value in metadata.items():
            grid.field_data[key] = [value]
            
    grid.save(filename)
    return grid

def export_chfem_inputs(final_grid, base_filename="model", voxel_size=1e-8, physics_mode='thermal', prop_map=None):
    """
    Export the raw binary array and neutral file (.nf) required by the chfem solver.
    """
    raw_filename = f"{base_filename}.raw"
    final_grid.tofile(raw_filename)
    nf_filename = f"{base_filename}.nf"
    shape = final_grid.shape

    analysis_type = 1 if physics_mode == 'mechanics' else 0
    num_materials = len(prop_map)

    # Serialize physical properties respecting ascending dictionary ID order
    props_str = "\n".join([f"{k} {v}" for k, v in sorted(prop_map.items())])

    nf_content = f"""%type_of_analysis {analysis_type}
%voxel_size {voxel_size}
%solver_tolerance 2.5e-07
%number_of_iterations 20000
%image_dimensions {shape[2]} {shape[1]} {shape[0]}
%refinement 1
%number_of_materials {num_materials}
%properties_of_materials
{props_str}
%data_type float64
"""
    with open(nf_filename, 'w') as f:
        f.write(nf_content)
