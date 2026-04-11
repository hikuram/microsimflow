import math
import numpy as np
from scipy.ndimage import binary_dilation, convolve, label, generate_binary_structure, affine_transform
from tqdm.auto import tqdm
import pyvista as pv
from numba import njit

# --- Initialize global random number generator ---
rng = np.random.default_rng()

def set_random_seed(seed):
    """Reinitialize the generator by fixing the seed externally"""
    global rng
    rng = np.random.default_rng(seed)
    
def crop_mask_to_bbox(mask):
    """Trim the margins of the mask and crop it with a bounding box"""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return mask
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    return mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

def crop_and_standardize(mask, cm_global_abs):
    """
    Universal standardizer for rigid-body kinematics in voxel grids.
    1. Crops the empty margins of the 3D mask.
    2. Calculates the exact offset vector from the cropped array center to the true physical CM.
    This guarantees zero "jumping" during affine deformation and placement.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0: 
        return mask, np.zeros(3)
        
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    cropped_mask = mask[mins[0]:maxs[0]+1, mins[1]:maxs[1]+1, mins[2]:maxs[2]+1]
    cropped_center = np.array(cropped_mask.shape) // 2
    
    # Vector from the geometric center of the cropped array to the physical center of mass
    cm_in_cropped = cm_global_abs - mins
    offset_center_to_cm = cm_in_cropped - cropped_center
    
    return cropped_mask, offset_center_to_cm

# =========================================================
# A. Background Phase (Polymer Matrix) Generation Module
# =========================================================

def build_single_phase_grid(grid_size):
    """Single polymer phase (all Phase A: 0)"""
    tpms_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    return None, tpms_grid, 0.0, 1.0

def build_tpms_grid_with_target_ratio(grid_size, wavelength=10, target_phaseA_ratio=0.5):
    """Co-continuous Gyroid phase"""
    """Binarize the gyroid to match the specified Phase A volume ratio"""
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    gyroid = (np.sin(x / wavelength) * np.cos(y / wavelength) + 
              np.sin(y / wavelength) * np.cos(z / wavelength) + 
              np.sin(z / wavelength) * np.cos(x / wavelength))
    threshold = np.percentile(gyroid, target_phaseA_ratio * 100)
    tpms_grid = np.where(gyroid > threshold, 1, 0).astype(np.uint8)
    actual_phaseA_ratio = np.mean(tpms_grid == 0)
    return gyroid, tpms_grid, threshold, actual_phaseA_ratio

def build_lamellar_grid(grid_size, wavelength=10, target_phaseA_ratio=0.5):
    """Lamellar structure (spread in XY plane, stacked in Z direction)"""
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    
    # 1D periodic function in the Z direction
    field = np.cos(2 * np.pi * z / wavelength)
    
    threshold = np.percentile(field, target_phaseA_ratio * 100)
    grid = np.where(field > threshold, 1, 0).astype(np.uint8)
    actual_phaseA_ratio = np.mean(grid == 0)
    
    return field, grid, threshold, actual_phaseA_ratio

def build_cylinder_hex_grid(grid_size, wavelength=15, target_phaseA_ratio=0.7):
    """Cylinder structure (upright in Z-axis direction, hexagonal array in XY plane)"""
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    q = 2 * np.pi / wavelength
    
    # Approximation of 2D hexagonal lattice (superposition of waves in 3 directions)
    field = (np.cos(q * x) + 
             np.cos(q * (x + np.sqrt(3) * y) / 2.0) + 
             np.cos(q * (x - np.sqrt(3) * y) / 2.0))
    
    threshold = np.percentile(field, target_phaseA_ratio * 100)
    grid = np.where(field > threshold, 1, 0).astype(np.uint8)
    actual_phaseA_ratio = np.mean(grid == 0)
    
    return field, grid, threshold, actual_phaseA_ratio

def build_bcc_grid(grid_size, wavelength=15, target_phaseA_ratio=0.7):
    """Body-centered cubic structure (BCC: 3D regular array of spherical domains)"""
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    q = 2 * np.pi / wavelength
    
    # Approximation of BCC lattice
    field = (np.cos(q * x) * np.cos(q * y) + 
             np.cos(q * y) * np.cos(q * z) + 
             np.cos(q * z) * np.cos(q * x))
    
    threshold = np.percentile(field, target_phaseA_ratio * 100)
    grid = np.where(field > threshold, 1, 0).astype(np.uint8)
    actual_phaseA_ratio = np.mean(grid == 0)
    
    return field, grid, threshold, actual_phaseA_ratio

def build_sea_island_grid(grid_size, island_radius=8, target_phaseA_ratio=0.7):
    """Sea-island structure (Random sphere placement by Boolean model: Island is Phase B)"""
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    target_voxels = int((grid_size**3) * (1 - target_phaseA_ratio))
    placed_voxels = 0
    
    # Create island mask
    size = int(island_radius * 2 + 2)
    z, y, x = np.indices((size, size, size))
    cz, cy, cx = size//2, size//2, size//2
    sphere = (z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= island_radius**2
    coords = np.argwhere(sphere)
    center = np.array(sphere.shape) // 2
    offsets = coords - center
    
    attempts = 0
    # Random placement allowing overlap
    while placed_voxels < target_voxels and attempts < 100000:
        attempts += 1
        cz, cy, cx = rng.integers(0, grid_size, size=3)
        t_coords = offsets + np.array([cz, cy, cx])
        tz, ty, tx = t_coords[:, 0] % grid_size, t_coords[:, 1] % grid_size, t_coords[:, 2] % grid_size
        
        grid[tz, ty, tx] = 1
        placed_voxels = np.sum(grid == 1)
        
    actual_phaseA_ratio = 1.0 - (placed_voxels / (grid_size**3))
    return None, grid, 0.0, actual_phaseA_ratio

def build_island_sea_grid(grid_size, island_radius=8, target_phaseA_ratio=0.7):
    """Island-sea structure (Random sphere placement by Boolean model: Island is Phase A)"""
    target_phaseB_ratio = 1.0 - target_phaseA_ratio
    _, grid, _, actual_phaseB_ratio = build_sea_island_grid(grid_size, island_radius, target_phaseB_ratio)
    grid ^= 1
    actual_phaseA_ratio = 1.0 - actual_phaseB_ratio
    return None, grid, 0.0, actual_phaseA_ratio

# =========================================================
# B. Rigid Stamp Module (Sphere, Flake, Rigid Cylinder, etc.)
# =========================================================

def create_rotated_grid_with_normal(shape, angles):
    """Helper to create a 3D grid and return both the grid and its Z-axis normal vector"""
    z, y, x = np.indices(shape)
    cz, cy, cx = shape[0]//2, shape[1]//2, shape[2]//2
    Z = z - cz; Y = y - cy; X = x - cx
    az, ay, ax = angles
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    R = Rz @ Ry @ Rx
    coords = np.stack([Z.ravel(), Y.ravel(), X.ravel()])
    rotated_coords = R @ coords
    normal_vector = R @ np.array([1, 0, 0])
    return rotated_coords[2,:].reshape(shape), rotated_coords[1,:].reshape(shape), rotated_coords[0,:].reshape(shape), normal_vector

def get_sphere_mask(radius, physics_mode='thermal'):
    """Perfect sphere filler standardized for kinematics"""
    size = int(radius * 2 + 2)
    z, y, x = np.indices((size, size, size))
    cz, cy, cx = size//2, size//2, size//2
    mask = (z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= radius**2
    
    cm_global_abs = np.array([cz, cy, cx], dtype=float)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    
    geom_data = {
        'base_type': 'sphere', 'radius': radius,
        'R_orig': np.eye(3).tolist(), 'local_kinematics': [[0.0, 0.0, 0.0]],
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def get_flake_mask(radius, thickness, physics_mode='thermal'):
    """Flake-shaped filler with Polar Decomposition support"""
    size = int(radius * 2 + 4)
    angles = rng.random(3) * 2 * np.pi
    az, ay, ax = angles
    R_orig = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]]) @ \
             np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]]) @ \
             np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
             
    Z, Y, X = np.indices((size, size, size))
    Z = Z - size//2; Y = Y - size//2; X = X - size//2
    rot = R_orig @ np.stack([Z.ravel(), Y.ravel(), X.ravel()])
    Zr, Yr, Xr = rot[0,:].reshape((size, size, size)), rot[1,:].reshape((size, size, size)), rot[2,:].reshape((size, size, size))
    
    mask = (Xr**2 + Yr**2 <= radius**2) & (np.abs(Zr) <= thickness/2)
    cm_global_abs = np.array([size//2, size//2, size//2], dtype=float)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    
    geom_data = {
        'base_type': 'flake', 'radius': radius, 'thickness': thickness,
        'R_orig': R_orig.tolist(), 'local_kinematics': [[0.0, 0.0, 0.0]], 
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def create_staggered_flakes_mask(radius=15, layer_thickness=2, min_layers=1, max_layers=4, max_offset_pct=20):
    """Generate a compound stamp of staggered platelets, rigorously standardizing its centroid."""
    rng = np.random.default_rng()
    num_layers = rng.integers(min_layers, max_layers + 1)
    max_offset_px = radius * (max_offset_pct / 100.0)
    box_size = int(math.ceil((radius + num_layers * max_offset_px) * 2 + num_layers * layer_thickness)) + 4
    
    angles = rng.random(3) * 2 * np.pi
    az, ay, ax = angles
    R_orig = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]]) @ \
             np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]]) @ \
             np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
             
    Z, Y, X = np.indices((box_size, box_size, box_size))
    Z = Z - box_size//2; Y = Y - box_size//2; X = X - box_size//2
    rot = R_orig @ np.stack([Z.ravel(), Y.ravel(), X.ravel()])
    Zr, Yr, Xr = rot[0,:].reshape((box_size, box_size, box_size)), rot[1,:].reshape((box_size, box_size, box_size)), rot[2,:].reshape((box_size, box_size, box_size))

    local_centers = []
    cy, cx = 0.0, 0.0
    for i in range(num_layers):
        if i > 0:
            angle = rng.uniform(0, 2 * np.pi)
            cy += rng.uniform(0, max_offset_px) * np.sin(angle)
            cx += rng.uniform(0, max_offset_px) * np.cos(angle)
        z_c = - (num_layers * layer_thickness) / 2.0 + (i + 0.5) * layer_thickness
        local_centers.append([z_c, cy, cx])
        
    local_centers = np.array(local_centers)
    cm_local = np.mean(local_centers, axis=0)
    local_centers -= cm_local # Pure local kinematic points (CM = 0,0,0)
    
    cm_global_rel = R_orig.T @ cm_local
    cm_global_abs = np.array([box_size//2, box_size//2, box_size//2], dtype=float) + cm_global_rel

    stamp = np.zeros((box_size, box_size, box_size), dtype=bool)
    for z_c, y_c, x_c in (local_centers + cm_local):
        stamp |= ((Xr - x_c)**2 + (Yr - y_c)**2 <= radius**2) & (np.abs(Zr - z_c) <= layer_thickness / 2.0)

    cropped_mask, offset = crop_and_standardize(stamp, cm_global_abs)
    
    geom_data = {
        'base_type': 'staggered', 'radius': radius, 'layer_thickness': layer_thickness,
        'R_orig': R_orig.tolist(), 'local_kinematics': local_centers.tolist(), 
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def get_staggered_flakes_mask(radius=15, layer_thickness=2, min_layers=1, max_layers=4, max_offset_pct=30, physics_mode='thermal'):
    return create_staggered_flakes_mask(radius, layer_thickness, min_layers, max_layers, max_offset_pct)

def create_fiber_mask(length, radius, max_bend_deg=90, max_total_bends=10):
    """Generate a mask for a single flexible fiber and calculate true physical CM"""
    box_size = int(length * 2 + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    
    start_pos = np.array([box_size//2, box_size//2, box_size//2], dtype=float)
    backbone = [np.round(start_pos).astype(int)]
    current_pos = start_pos
    vec = rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    bends_made = 0

    for _ in range(int(length)):
        next_pos_f = current_pos + vec
        next_pos_i = np.round(next_pos_f).astype(int)
        if any(p < radius or p >= box_size - radius for p in next_pos_i): break
        if bends_made < max_total_bends and rng.random() < (max_total_bends / length):
            angle_rad = np.radians(rng.uniform(10, max_bend_deg))
            noise = rng.standard_normal(3)
            noise -= noise.dot(vec) * vec
            if np.linalg.norm(noise) > 0:
                noise /= np.linalg.norm(noise)
                new_vec = vec * np.cos(angle_rad) + noise * np.sin(angle_rad)
                vec = new_vec / np.linalg.norm(new_vec)
                bends_made += 1
        current_pos = current_pos + vec
        backbone.append(np.round(current_pos).astype(int))

    for (bz, by, bx) in backbone:
        mask[bz, by, bx] = True
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    mask = binary_dilation(mask, structure=brush)

    cm_global_abs = np.mean(backbone, axis=0)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    local_kinematics = np.array(backbone) - cm_global_abs
    
    geom_data = {
        'base_type': 'fiber', 'radius': radius,
        'R_orig': np.eye(3).tolist(), 'local_kinematics': local_kinematics.tolist(),
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def get_flexible_fiber_mask(length=90, radius=2, max_bend_deg=90, max_total_bends=10, physics_mode='thermal'):
    return create_fiber_mask(length, radius, max_bend_deg, max_total_bends)

def get_rigid_cylinder_mask(length, radius, physics_mode='thermal'):
    """Rigid short fiber extracted as pure kinematics"""
    box_size = int(length + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    vec = rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    start_pos = np.array([box_size//2, box_size//2, box_size//2], dtype=float) - vec * (length / 2)
    
    backbone = []
    current_pos = start_pos
    for _ in range(int(length)):
        iz, iy, ix = np.round(current_pos).astype(int)
        mask[iz, iy, ix] = True
        backbone.append([current_pos[0], current_pos[1], current_pos[2]])
        current_pos += vec
        
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    mask = binary_dilation(mask, structure=brush)
    
    cm_global_abs = np.mean(backbone, axis=0)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    local_kinematics = np.array(backbone) - cm_global_abs
    
    geom_data = {
        'base_type': 'fiber', 'radius': radius,
        'R_orig': np.eye(3).tolist(), 'local_kinematics': local_kinematics.tolist(),
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends):
    box_size = int(length + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    center_pos = (box_size // 2, box_size // 2, box_size // 2)
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    straight_steps = int(max(3, length * 0.05))
    half_len_a = length // 2
    half_len_b = length - half_len_a
    vec = rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    start_pos = center_pos
    
    def grow_path(steps, v_dir):
        path = []
        curr = np.array(start_pos, dtype=float)
        v = v_dir.copy()
        bends = 0
        bend_prob = (max_total_bends / 2) / max(1, steps - straight_steps)
        for i in range(steps):
            if i >= straight_steps and bends < (max_total_bends / 2):
                if rng.random() < bend_prob:
                    angle_rad = np.radians(rng.uniform(20, max_bend_deg))
                    noise = rng.standard_normal(3)
                    noise -= noise.dot(v) * v
                    if np.linalg.norm(noise) > 0:
                        noise /= np.linalg.norm(noise)
                        v = v * np.cos(angle_rad) + noise * np.sin(angle_rad)
                        v /= np.linalg.norm(v)
                        bends += 1
            curr += v
            z, y, x = np.round(curr).astype(int)
            z = max(radius, min(box_size - radius - 1, z))
            y = max(radius, min(box_size - radius - 1, y))
            x = max(radius, min(box_size - radius - 1, x))
            path.append((z, y, x))
        return path

    backbone_a = grow_path(half_len_a, vec)
    backbone_b = grow_path(half_len_b, -vec)
    backbone = backbone_b[::-1] + [tuple(start_pos)] + backbone_a
    for (z, y, x) in backbone:
        mask[z, y, x] = True
    mask = binary_dilation(mask, structure=brush)
    return mask, backbone

def create_agglomerate_mask(num_fibers, length, radius, max_bend_deg=90, max_total_bends=10, physics_mode='thermal', filler_id=4, inter_id=3):
    box_size = int(length + radius * 2 + 5)
    combined_mask = np.zeros((box_size, box_size, box_size), dtype=np.uint8)
    start_radius = radius * 2
    all_backbones = []
    
    for _ in range(num_fibers):
        fiber_mask, fiber_bb = _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends)
        offset = np.round(rng.standard_normal(3) * start_radius).astype(int)
        shifted_bb = np.array(fiber_bb) + offset
        all_backbones.append(shifted_bb)
        shift_z, shift_y, shift_x = offset
        
        z_start, z_end = max(0, shift_z), min(box_size, box_size + shift_z)
        y_start, y_end = max(0, shift_y), min(box_size, box_size + shift_y)
        x_start, x_end = max(0, shift_x), min(box_size, box_size + shift_x)
        
        fz_start = max(0, -shift_z); fz_end = fz_start + (z_end - z_start)
        fy_start = max(0, -shift_y); fy_end = fy_start + (y_end - y_start)
        fx_start = max(0, -shift_x); fx_end = fx_start + (x_end - x_start)
        
        target_view = combined_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        source_view = fiber_mask[fz_start:fz_end, fy_start:fy_end, fx_start:fx_end]
        
        if physics_mode == 'thermal':
            target_view[(target_view >= 2) & source_view] = inter_id  
            target_view[(target_view == 0) & source_view] = filler_id 
        else:
            target_view[source_view] = filler_id

    all_pts = np.vstack(all_backbones)
    cm_global_abs = np.mean(all_pts, axis=0)
    cropped_mask, offset = crop_and_standardize(combined_mask > 0, cm_global_abs)
    
    local_kinematics = [(bb - cm_global_abs).tolist() for bb in all_backbones]
    
    geom_data = {
        'base_type': 'agglomerate', 'radius': radius,
        'R_orig': np.eye(3).tolist(), 'local_kinematics': local_kinematics,
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def calculate_protrusion_limit(filler_voxels, total_voxels, half_protrusion_vol_ratio=0.0025):
    """Calculation of adaptive protrusion tolerance based on half-value volume ratio model"""
    if half_protrusion_vol_ratio <= 0:
        return 0.0
    x = filler_voxels / total_voxels
    C = half_protrusion_vol_ratio
    return ((1 + C) * x) / (x + C)

# =========================================================
# C. Topology-Adaptive Growth Module (Straight penetration + 180-deg U-turn & void generation)
# =========================================================

def grow_adaptive_fiber(tpms_grid, comp_grid, start_pos, length, overlap_mode,
                        max_bend_deg=45, max_total_bends=5, max_retries_per_step=10, 
                        min_backbone_ratio=0.9, protrusion_coef=0.0025, radius=2):
    
    shape = tpms_grid.shape
    total_voxels = tpms_grid.size
    backbone = [start_pos]
    current_pos = np.array(start_pos, dtype=float)
    
    vec = rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    
    bends_made = 0
    
    # Application of asymptotic model for protrusion limits
    filler_voxels = length * np.pi * radius**2
    limit = calculate_protrusion_limit(filler_voxels, total_voxels, protrusion_coef)
    max_bridge_steps = max(1, int(length * limit))

    for _ in range(int(length)):
        next_pos_f = current_pos + vec
        next_pos_i = np.round(next_pos_f).astype(int)
        
        zg, yg, xg = next_pos_i[0] % shape[0], next_pos_i[1] % shape[1], next_pos_i[2] % shape[2]

        phase_b = (tpms_grid[zg, yg, xg] == 1)
        filler = (not overlap_mode) and (comp_grid[zg, yg, xg] > 0)
        
        if not phase_b and not filler:
            current_pos = next_pos_f
            backbone.append((zg, yg, xg))
            continue

        # --- Obstacle / Interface Judgment ---
        need_to_bend = False

        if filler:
            need_to_bend = True
        elif phase_b:
            bridged = False
            hit_filler_in_bridge = False
            bridge_path = [(zg, yg, xg)]
            test_pos_f = next_pos_f

            for step in range(1, max_bridge_steps):
                test_pos_f = test_pos_f + vec
                test_pos_i = np.round(test_pos_f).astype(int)
                tz, ty, tx = test_pos_i[0] % shape[0], test_pos_i[1] % shape[1], test_pos_i[2] % shape[2]
                bridge_path.append((tz, ty, tx))

                if (not overlap_mode) and (comp_grid[tz, ty, tx] > 0):
                    hit_filler_in_bridge = True
                    break
                elif tpms_grid[tz, ty, tx] == 0:
                    bridged = True
                    break
            
            if bridged:
                backbone.extend(bridge_path)
                current_pos = test_pos_f
                continue
            elif hit_filler_in_bridge:
                need_to_bend = True
            else:
                break # Stopped because the wall is too thick to penetrate

        # --- Bending / U-turn Processing ---
        if need_to_bend:
            step_success = False
            if bends_made < max_total_bends:
                for _ in range(max_retries_per_step):
                    angle_deg = rng.uniform(10, max_bend_deg)
                    
                    # Snap to "180-degree U-turn + side step" if 90 degrees or more
                    if angle_deg >= 90.0:
                        noise = rng.standard_normal(3)
                        u_ortho = noise - noise.dot(vec) * vec
                        if np.linalg.norm(u_ortho) == 0: continue
                        u_ortho /= np.linalg.norm(u_ortho)
                        
                        # Calculate side step distance: Center-to-center distance of 1.6 * R
                        shift_dist = max(2, int(radius * 1.6))
                        
                        sidestep_success = True
                        temp_pos = current_pos.copy()
                        temp_path = []
                        
                        # Crab-walk to the side
                        for _ in range(shift_dist):
                            temp_pos += u_ortho
                            temp_i = np.round(temp_pos).astype(int)
                            tz, ty, tx = temp_i[0] % shape[0], temp_i[1] % shape[1], temp_i[2] % shape[2]
                            
                            # Check for collisions with other fillers or walls during side step
                            if (tpms_grid[tz, ty, tx] == 1) or ((not overlap_mode) and (comp_grid[tz, ty, tx] > 0)):
                                sidestep_success = False
                                break
                            temp_path.append((tz, ty, tx))
                            
                        if sidestep_success:
                            current_pos = temp_pos
                            backbone.extend(temp_path)
                            vec = -vec  # Reverse traveling direction (180-degree U-turn)
                            bends_made += 1
                            step_success = True
                            break
                            
                    else:
                        # Normal bend of less than 90 degrees
                        angle_rad = np.radians(angle_deg)
                        noise = rng.standard_normal(3)
                        noise -= noise.dot(vec) * vec
                        if np.linalg.norm(noise) > 0:
                            noise /= np.linalg.norm(noise)
                            
                        new_vec = vec * np.cos(angle_rad) + noise * np.sin(angle_rad)
                        new_vec /= np.linalg.norm(new_vec)
                        
                        next_pos_f_try = current_pos + new_vec
                        next_pos_i_try = np.round(next_pos_f_try).astype(int)
                        zg_try, yg_try, xg_try = next_pos_i_try[0] % shape[0], next_pos_i_try[1] % shape[1], next_pos_i_try[2] % shape[2]
                        
                        if tpms_grid[zg_try, yg_try, xg_try] == 0 and not ((not overlap_mode) and (comp_grid[zg_try, yg_try, xg_try] > 0)):
                            current_pos = next_pos_f_try
                            vec = new_vec
                            backbone.append((zg_try, yg_try, xg_try))
                            bends_made += 1
                            step_success = True
                            break
                        
            if not step_success:
                break
                
    if len(backbone) < length * min_backbone_ratio:
        return None
        
    return backbone
    
def apply_brush_and_write(comp_grid, backbone, radius, physics_mode='thermal', shell_count_grid=None, filler_id=4, inter_id=3):
    shape = comp_grid.shape
    size = int(radius * 2 + 2)
    z, y, x = np.indices((size, size, size))
    cz, cy, cx = size//2, size//2, size//2
    brush_mask = (z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= radius**2
    local_z, local_y, local_x = np.where(brush_mask)

    fiber_voxels = set()
    for (z, y, x) in backbone:
        glob_z = (local_z - cz + z) % shape[0]
        glob_y = (local_y - cy + y) % shape[1]
        glob_x = (local_x - cx + x) % shape[2]
        for i in range(len(glob_z)):
            fiber_voxels.add((glob_z[i], glob_y[i], glob_x[i]))

    if not fiber_voxels:
        return 0

    fv_array = np.array(list(fiber_voxels))
    gz, gy, gx = fv_array[:, 0], fv_array[:, 1], fv_array[:, 2]

    if physics_mode == 'thermal':
        # Thermal conduction mode: Areas overlapping with existing fillers are set to penalty
        contact_mask = (comp_grid[gz, gy, gx] >= 2)
        comp_grid[gz[contact_mask], gy[contact_mask], gx[contact_mask]] = inter_id
        body_mask = ~contact_mask
        comp_grid[gz[body_mask], gy[body_mask], gx[body_mask]] = filler_id
    else:
        # Electrical/Mechanics mode: The main body is unconditionally maintained
        comp_grid[gz, gy, gx] = filler_id
        
        # Count the shell dilated by (tunnel_radius) voxels
        if shell_count_grid is not None:
            size_sh = int((radius + tunnel_radius) * 2 + 2)
            z_sh, y_sh, x_sh = np.indices((size_sh, size_sh, size_sh))
            cz_sh, cy_sh, cx_sh = size_sh//2, size_sh//2, size_sh//2
            shell_brush_mask = (z_sh - cz_sh)**2 + (y_sh - cy_sh)**2 + (x_sh - cx_sh)**2 <= (radius + tunnel_radius)**2
            local_z_sh, local_y_sh, local_x_sh = np.where(shell_brush_mask)
            
            shell_voxels = set()
            for (z, y, x) in backbone:
                glob_z = (local_z_sh - cz_sh + z) % shape[0]
                glob_y = (local_y_sh - cy_sh + y) % shape[1]
                glob_x = (local_x_sh - cx_sh + x) % shape[2]
                for i in range(len(glob_z)):
                    shell_voxels.add((glob_z[i], glob_y[i], glob_x[i]))
                    
            if shell_voxels:
                sv_array = np.array(list(shell_voxels))
                sz, sy, sx = sv_array[:, 0], sv_array[:, 1], sv_array[:, 2]
                shell_count_grid[sz, sy, sx] += 1

    return len(gz)

def place_adaptive_fibers(comp_grid, tpms_grid, target_vol_frac, length, radius,
                          max_bend_deg=45, max_total_bends=10, max_retries_per_step=10, max_protrusion_ratio=0.1,
                          min_backbone_ratio=0.9, max_attempts=1000000, desc="", log_file=None,
                          physics_mode='thermal', shell_count_grid=None, filler_id=4, inter_id=3):
    shape = comp_grid.shape
    target_voxels = int(np.prod(shape) * target_vol_frac)
    valid_z, valid_y, valid_x = np.where(tpms_grid == 0)
    num_valid_coords = len(valid_z)

    placed_voxels = np.sum(comp_grid >= 2)
    initial_placed = placed_voxels
    attempts = 0
    consecutive_fails = 0
    overlap_mode = False

    with tqdm(total=target_voxels, desc=desc, unit="voxel") as pbar:
        while placed_voxels < initial_placed + target_voxels and attempts < max_attempts:
            attempts += 1
            
            # Allow intersections between fillers (soft-core mode) if consecutive failures exceed 500
            if not overlap_mode and consecutive_fails > 500:
                overlap_mode = True

            idx = rng.integers(0, num_valid_coords)
            start_pos = (valid_z[idx], valid_y[idx], valid_x[idx])

            backbone = grow_adaptive_fiber(
                tpms_grid, comp_grid, start_pos, length, overlap_mode,
                max_bend_deg=max_bend_deg, 
                max_total_bends=max_total_bends, 
                max_retries_per_step=max_retries_per_step,
                min_backbone_ratio=min_backbone_ratio,
                protrusion_coef=max_protrusion_ratio,
                radius=radius
            )

            if backbone is None:
                consecutive_fails += 1
                continue

            consecutive_fails = 0
            apply_brush_and_write(comp_grid, backbone, radius, physics_mode, shell_count_grid, filler_id, inter_id)

            current_total = np.sum(comp_grid >= 2)
            added_voxels = current_total - placed_voxels
            placed_voxels = current_total

            if added_voxels > 0:
                pbar.update(added_voxels)

    log_text = (
        f"\n[{desc or 'Adaptive Fibers'}] placement summary\n"
        f"  attempts: {attempts}\n"
        f"  overlap_mode_switches: {1 if overlap_mode else 0}\n"
        f"  final_added_voxels: {placed_voxels - initial_placed}\n"
    )
    print(log_text)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_text)

    return placed_voxels

# =========================================================
# D. RSA Placement Logic (Ultra-fast index version)
# =========================================================

# ==========================================
# Optimized Core Routine for RSA Placement
# ==========================================

@njit
def _check_and_place_fast(comp_grid, tpms_grid, cz, cy, cx, 
                          stamp_offsets, stamp_vals, current_protrusion_limit, 
                          overlap_mode, is_thermal, filler_id, inter_id):
    """
    Numba-optimized JIT compiled function.
    Performs boundary checks, collision detection, and voxel writing without Python loop overhead.
    """
    shape_z, shape_y, shape_x = comp_grid.shape
    num_coords = stamp_offsets.shape[0]
    
    # 1. Protrusion Check (Pre-calculate number of voxels extending into Phase B)
    protrusion_count = 0
    for i in range(num_coords):
        tz = (stamp_offsets[i, 0] + cz) % shape_z
        ty = (stamp_offsets[i, 1] + cy) % shape_y
        tx = (stamp_offsets[i, 2] + cx) % shape_x
        
        if tpms_grid[tz, ty, tx] == 1:
            protrusion_count += 1
            
    # Abort if protrusion exceeds the allowable limit
    if protrusion_count > num_coords * current_protrusion_limit:
        return False
        
    # 2. Overlap Check (Only if overlap_mode is False)
    if not overlap_mode:
        for i in range(num_coords):
            tz = (stamp_offsets[i, 0] + cz) % shape_z
            ty = (stamp_offsets[i, 1] + cy) % shape_y
            tx = (stamp_offsets[i, 2] + cx) % shape_x
            if comp_grid[tz, ty, tx] >= 2:
                return False # Immediate failure upon collision
                
    # 3. Write operation (Guaranteed to succeed at this point)
    for i in range(num_coords):
        tz = (stamp_offsets[i, 0] + cz) % shape_z
        ty = (stamp_offsets[i, 1] + cy) % shape_y
        tx = (stamp_offsets[i, 2] + cx) % shape_x
        
        current_val = comp_grid[tz, ty, tx]
        new_val = stamp_vals[i]
        
        if is_thermal:
            # Thermal mode: Overlaps become contact resistance phase (inter_id)
            if current_val >= 2:
                comp_grid[tz, ty, tx] = inter_id
            else:
                comp_grid[tz, ty, tx] = new_val
        else:
            # Electrical/Mechanics mode: Write main body as a good conductor
            if new_val > 0:
                comp_grid[tz, ty, tx] = filler_id
                
    return True

def place_fillers_hybrid(comp_grid, tpms_grid, filler_func, kwargs, target_vol_frac,
                         max_attempts=1000000, fallback_func=None, desc="",
                         protrusion_coef=0.0025, log_file=None,
                         physics_mode='thermal', shell_count_grid=None,
                         filler_id=4, inter_id=3, tunnel_radius=2, placement_registry=None):
    # Initialize RNG inside the function for safety if not using global
    rng = np.random.default_rng()
    
    shape = comp_grid.shape
    total_voxels = comp_grid.size
    target_voxels = int(total_voxels * target_vol_frac)
    
    valid_z, valid_y, valid_x = np.where(tpms_grid == 0)
    num_valid_coords = len(valid_z)

    placed_voxels = np.sum(comp_grid >= 2)
    initial_placed = placed_voxels
    attempts = 0
    consecutive_fails = 0
    overlap_mode = False

    stamp_offsets = None
    stamp_vals = None
    shell_offsets = None
    cache_reuse_count = 0
    MAX_CACHE_REUSE = 50
    current_protrusion_limit = 0.0 

    if 'physics_mode' not in kwargs:
        kwargs['physics_mode'] = physics_mode
        
    is_thermal = (physics_mode == 'thermal')

    with tqdm(total=target_voxels, desc=desc, unit="voxel") as pbar:
        while placed_voxels < initial_placed + target_voxels and attempts < max_attempts:
            attempts += 1
            
            # Switch to soft-core (overlap allowed) if failed continuously
            if not overlap_mode and consecutive_fails > 500:
                overlap_mode = True

            # Update cache for rigid filler geometries
            if stamp_offsets is None or cache_reuse_count >= MAX_CACHE_REUSE:
                result = filler_func(**kwargs)
                # Handle the new tuple return type (mask, geom_data)
                if isinstance(result, tuple):
                    raw_stamp, current_geom_data = result
                else:
                    raw_stamp = result
                    current_geom_data = {'base_type': 'unknown'}
                
                # For placement judgment: Extract only the occupied coordinates of space
                coords = np.argwhere(raw_stamp > 0)
                if len(coords) == 0:
                    stamp_offsets = None
                    continue
                
                center = np.array(raw_stamp.shape) // 2
                stamp_offsets = coords - center
                
                # Pre-extract values for writing
                if raw_stamp.dtype == bool:
                    stamp_vals = np.full(len(coords), filler_id, dtype=np.uint8)
                else:
                    stamp_vals = raw_stamp[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.uint8)

                # Shell extraction for electrical mode
                if not is_thermal and shell_count_grid is not None:
                    rz, ry, rx = np.ogrid[-tunnel_radius:tunnel_radius+1, 
                                          -tunnel_radius:tunnel_radius+1, 
                                          -tunnel_radius:tunnel_radius+1]
                    shell_brush = rx**2 + ry**2 + rz**2 <= tunnel_radius**2
                    dilated = binary_dilation(raw_stamp > 0, structure=shell_brush)
                    shell_coords = np.argwhere(dilated)
                    shell_offsets = shell_coords - center
                else:
                    shell_offsets = None

                filler_voxels = len(coords)
                # Note: Assuming calculate_protrusion_limit is defined elsewhere in the file
                current_protrusion_limit = calculate_protrusion_limit(filler_voxels, total_voxels, protrusion_coef)
                cache_reuse_count = 0

            # Pick a random valid coordinate
            idx = rng.integers(0, num_valid_coords)
            cz, cy, cx = valid_z[idx], valid_y[idx], valid_x[idx]

            # Execute Numba-optimized placement routine
            success = _check_and_place_fast(
                comp_grid, tpms_grid, cz, cy, cx, 
                stamp_offsets, stamp_vals, current_protrusion_limit, 
                overlap_mode, is_thermal, filler_id, inter_id
            )

            if not success:
                cache_reuse_count += 1
                consecutive_fails += 1
                continue

            # Update shell count grid outside Numba using fast vectorized Numpy operations
            if not is_thermal and shell_count_grid is not None and shell_offsets is not None:
                stz = (shell_offsets[:, 0] + cz) % shape[0]
                sty = (shell_offsets[:, 1] + cy) % shape[1]
                stx = (shell_offsets[:, 2] + cx) % shape[2]
                shell_count_grid[stz, sty, stx] += 1
            
            # Reset cache and fails only on success
            stamp_offsets = None 
            consecutive_fails = 0
            
            # Record successful placement geometry
            if placement_registry is not None:
                placement_registry.append({
                    'geom': current_geom_data,
                    'center': (cz, cy, cx),
                    'filler_id': filler_id,
                    'inter_id': inter_id
                })

            # Update progress
            current_total = np.sum(comp_grid >= 2)
            added_voxels = current_total - placed_voxels
            placed_voxels = current_total

            if added_voxels > 0:
                pbar.update(added_voxels)

    log_text = (
        f"\n[{desc or filler_func.__name__}] placement summary\n"
        f"  attempts: {attempts}\n"
        f"  overlap_mode_switches: {1 if overlap_mode else 0}\n"
        f"  final_added_voxels: {placed_voxels - initial_placed}\n"
    )
    print(log_text)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_text)

    return placed_voxels

# =========================================================
# E. Final Structure Integration, Cleanup, and Aggregation
# =========================================================

def _make_6n_kernel():
    """Create a 3D convolution kernel for 6-neighborhood (cross shape)"""
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
    A voxel is removed if it has fewer than min_neighbors neighbors within the same mask.
    """
    if not np.any(mask):
        return mask
    neighbor_count = convolve(mask.astype(np.uint8), _NEIGHBOR6_KERNEL, mode="wrap")
    return mask & (neighbor_count >= min_neighbors)

def _fill_polymer_slivers(final_grid, target_id=2, filler_start_id=4):
    """
    Convert polymer voxels to interface when they are directly sandwiched
    between filler and interface in 6-neighborhood.
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
    Remove tiny isolated connected components from a boolean mask.
    Uses 6-neighbor connectivity.
    """
    if not np.any(mask):
        return mask

    labeled, num = label(mask, structure=_NEIGHBOR6_KERNEL)
    if num == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_component_size
    keep[0] = False

    return keep[labeled]

def finalize_microstructure(comp_grid, tpms_grid, shell_count_grid=None, physics_mode='thermal', 
                            primary_inter_id=3, secondary_inter_id=3, filler_start_id=4,
                            sliver_fill_iters=1, spike_min_neighbors=2, min_interface_component_size=2,
                            contact_radius=1):
    """
    Combine background and filler phases.
    For electrical/mechanics modes, cleans up the unified interface, then splits it 
    into Primary (distance 1) and Secondary (distance 2) based on filler proximity.
    """
    final_grid = np.where(comp_grid > 0, comp_grid, tpms_grid).astype(np.uint8)

    if physics_mode in ['electrical', 'mechanics'] and shell_count_grid is not None:
        # Base interface extraction using primary_inter_id temporarily for the unified interface
        tunnel_mask = (shell_count_grid >= 2) & (final_grid < filler_start_id)
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

    # 3) Small component cleanup on interface only
    interface_mask = _cleanup_small_components(
        interface_mask,
        min_component_size=min_interface_component_size
    )

    # Identify voxels removed during cleanup
    removed_interface = original_interface_mask & (~interface_mask)

    # Rebuild grid based on physics mode semantics
    if physics_mode in ['electrical', 'mechanics']:
        # Revert removed unified interface back to polymer
        writable_mask = (final_grid < filler_start_id)
        final_grid[removed_interface & writable_mask] = tpms_grid[removed_interface & writable_mask]
        
        # --- Split remaining interface into Primary and Secondary based on distance ---
        unified_interface_mask = (final_grid == primary_inter_id)
        filler_mask = (final_grid >= filler_start_id)
        
        if np.any(unified_interface_mask) and np.any(filler_mask):
            # Dilate filler (contact_radius) times by exactly 1 voxel in 6-neighborhood
            struct_1voxel = generate_binary_structure(3, 1)
            dilated_filler = binary_dilation(filler_mask, structure=struct_1voxel, iterations=contact_radius)
            
            # Secondary interface is the part of the unified interface NOT touched by the dilation
            secondary_mask = unified_interface_mask & (~dilated_filler)
            
            # Update the grid
            final_grid[secondary_mask] = secondary_inter_id
            # (Primary interface voxels remain as primary_inter_id)

    elif physics_mode == 'thermal':
        # Revert removed interface (contact penalty) to the most prevalent adjacent filler ID
        if np.any(removed_interface):
            filler_ids = np.unique(final_grid[final_grid >= filler_start_id])
            
            if len(filler_ids) == 0:
                # Fallback if no fillers are somehow present
                final_grid[removed_interface] = filler_start_id
            elif len(filler_ids) == 1:
                # Fast path for single filler type
                final_grid[removed_interface] = filler_ids[0]
            else:
                # Multiple filler types: count 6-neighbors for each filler ID
                counts = []
                for fid in filler_ids:
                    # Convolve returns how many 'fid' voxels touch each grid point
                    c = convolve((final_grid == fid).astype(np.uint8), _NEIGHBOR6_KERNEL, mode="wrap")
                    counts.append(c[removed_interface])
                
                # counts array shape: (num_filler_ids, num_removed_voxels)
                counts = np.array(counts)
                best_idx = np.argmax(counts, axis=0)
                final_grid[removed_interface] = filler_ids[best_idx]

    return final_grid

def summarize_phase_fractions(final_grid, primary_inter_id=3, secondary_inter_id=3, filler_start_id=4):
    """Aggregate the volume fraction of each phase"""
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

def export_visualization_vti(final_grid, filename="microstructure.vti", voxel_size=1e-8, metadata=None):
    """
    Output the 3D grid in the optimal VTI format for ParaView.
    Embed physical dimensions (voxel_size) and metadata for HUD (metadata) in Field Data.
    """
    import pyvista as pv

    grid = pv.ImageData()
    
    # NumPy (Z, Y, X) -> PyVista (X, Y, Z)
    nz, ny, nx = final_grid.shape
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    
    # Set physical scale
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.origin = (0.0, 0.0, 0.0)
    
    # Store in cell_data
    grid.cell_data["Phase"] = final_grid.flatten(order="C")
    
    # Embed metadata (for HUD)
    if metadata:
        for key, value in metadata.items():
            grid.field_data[key] = [value]
            
    grid.save(filename)
    return grid

def export_chfem_inputs(final_grid, base_filename="model", voxel_size=1e-8, physics_mode='thermal', prop_map=None):
    """Export raw binary array and neutral file (.nf) for chfem solver"""
    raw_filename = f"{base_filename}.raw"
    final_grid.tofile(raw_filename)
    nf_filename = f"{base_filename}.nf"
    shape = final_grid.shape

    analysis_type = 1 if physics_mode == 'mechanics' else 0
    num_materials = len(prop_map)

    # Write out physical properties in dictionary ID order (0, 1, 2...)
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

# =========================================================
# F. Affine Deformation & Rendering Module
# =========================================================

def apply_background_deformation(grid, stretch_ratio=1.0, poisson_ratio=0.4):
    """Applies affine deformation ONLY to the continuous polymer background."""
    from scipy.ndimage import affine_transform
    if stretch_ratio == 1.0:
        return grid.copy()

    lam = stretch_ratio
    lam_nu = lam ** (-poisson_ratio)
    nz, ny, nx = grid.shape
    new_shape = (
        max(1, int(round(nz * lam_nu))),
        max(1, int(round(ny * lam_nu))),
        max(1, int(round(nx * lam)))
    )
    matrix = np.array([
        [1.0 / lam_nu, 0.0, 0.0],
        [0.0, 1.0 / lam_nu, 0.0],
        [0.0, 0.0, 1.0 / lam]
    ])
    return affine_transform(grid, matrix=matrix, output_shape=new_shape, order=0, mode='wrap')

def _transform_fiber_kinematics(local_bb, lam, lam_nu):
    """Calculates rigid-body transformation of an inextensible fiber backbone"""
    diffs = np.diff(local_bb, axis=0)
    diffs_def = np.zeros_like(diffs, dtype=float)
    diffs_def[:, 0] = diffs[:, 0] * lam_nu
    diffs_def[:, 1] = diffs[:, 1] * lam_nu
    diffs_def[:, 2] = diffs[:, 2] * lam

    orig_lens = np.linalg.norm(diffs, axis=1)
    def_lens = np.linalg.norm(diffs_def, axis=1)
    valid = def_lens > 1e-8
    diffs_def[valid] = (diffs_def[valid] / def_lens[valid, None]) * orig_lens[valid, None]

    new_bb = np.zeros((len(local_bb), 3), dtype=float)
    new_bb[1:] = np.cumsum(diffs_def, axis=0)
    start_anchor = local_bb[0] * np.array([lam_nu, lam_nu, lam])
    new_bb += (start_anchor - new_bb[0])
    return new_bb

def _paste_mask_to_grid(comp_grid, shell_count_grid, cz, cy, cx, mask, filler_id, inter_id, is_thermal, tunnel_radius):
    """Helper to paste a generic boolean mask (flakes/spheres) into the grid"""
    shape = comp_grid.shape
    coords = np.argwhere(mask > 0)
    if len(coords) == 0: return
    
    center = np.array(mask.shape) // 2
    offsets = coords - center
    
    gz = (offsets[:, 0] + cz) % shape[0]
    gy = (offsets[:, 1] + cy) % shape[1]
    gx = (offsets[:, 2] + cx) % shape[2]
    
    if is_thermal:
        contact = (comp_grid[gz, gy, gx] >= 2)
        comp_grid[gz[contact], gy[contact], gx[contact]] = inter_id
        comp_grid[gz[~contact], gy[~contact], gx[~contact]] = filler_id
    else:
        comp_grid[gz, gy, gx] = filler_id
        if shell_count_grid is not None:
            struct = generate_binary_structure(3, 1)
            dilated_mask = binary_dilation(mask, structure=struct, iterations=tunnel_radius)
            shell_coords = np.argwhere(dilated_mask)
            sh_offsets = shell_coords - center
            sz = (sh_offsets[:, 0] + cz) % shape[0]
            sy = (sh_offsets[:, 1] + cy) % shape[1]
            sx = (sh_offsets[:, 2] + cx) % shape[2]
            shell_count_grid[sz, sy, sx] += 1

def _render_and_paste_kinematics(comp_grid, shell_count_grid, P_CM_new, F_mat, geom, item, is_thermal, tunnel_radius):
    """
    Dynamically renders transformed local kinematics into a tight bounding box,
    drastically reducing memory overhead and preserving exact rigid-body volume.
    """
    base_type = geom['base_type']
    R_orig = np.array(geom['R_orig'])
    radius = geom['radius']
    local_kinematics = geom['local_kinematics']

    from scipy.linalg import polar

    if base_type in ['flake', 'staggered']:
        # Extract pure rigid-body rotation (Polar Decomposition)
        R_local_to_global = R_orig.T
        R_pure_local_to_global, _ = polar(F_mat @ R_local_to_global)

        loc_pts = np.array(local_kinematics)
        new_rel_global_centers = (R_pure_local_to_global @ loc_pts.T).T

        # Dynamic bounding box based on transformed geometry
        max_radius = radius + 2
        min_b = np.floor(new_rel_global_centers.min(axis=0)).astype(int) - int(max_radius)
        max_b = np.ceil(new_rel_global_centers.max(axis=0)).astype(int) + int(max_radius)
        box_shape = tuple(max_b - min_b + 1)

        Z, Y, X = np.mgrid[min_b[0]:max_b[0]+1, min_b[1]:max_b[1]+1, min_b[2]:max_b[2]+1]
        coords_global_shifted = np.stack([Z.ravel(), Y.ravel(), X.ravel()])

        R_global_to_local = R_pure_local_to_global.T
        coords_local = R_global_to_local @ coords_global_shifted
        Z_loc = coords_local[0,:].reshape(box_shape)
        Y_loc = coords_local[1,:].reshape(box_shape)
        X_loc = coords_local[2,:].reshape(box_shape)

        mask = np.zeros(box_shape, dtype=bool)

        if base_type == 'staggered':
            layer_thickness = geom['layer_thickness']
            for z_c, y_c, x_c in local_kinematics:
                mask |= ((X_loc - x_c)**2 + (Y_loc - y_c)**2 <= radius**2) & (np.abs(Z_loc - z_c) <= layer_thickness / 2.0)
        elif base_type == 'flake':
            thickness = geom['thickness']
            z_c, y_c, x_c = local_kinematics[0]
            mask |= ((X_loc - x_c)**2 + (Y_loc - y_c)**2 <= radius**2) & (np.abs(Z_loc - z_c) <= thickness / 2.0)

        coords_nz = np.argwhere(mask > 0)
        if len(coords_nz) == 0: return

        c_mins = coords_nz.min(axis=0)
        c_maxs = coords_nz.max(axis=0)
        cropped = mask[c_mins[0]:c_maxs[0]+1, c_mins[1]:c_maxs[1]+1, c_mins[2]:c_maxs[2]+1]

        # Mathematically reverse-calculate exact paste target coordinates from the dynamic CM
        cm_in_cropped = -min_b - c_mins
        new_offset = cm_in_cropped - (np.array(cropped.shape) // 2)

    elif base_type in ['fiber', 'agglomerate']:
        lam = F_mat[2, 2]
        lam_nu = F_mat[0, 0]

        if base_type == 'fiber':
            new_rel_bb = _transform_fiber_kinematics(np.array(local_kinematics), lam, lam_nu)
            new_rel_bb -= np.mean(new_rel_bb, axis=0) # Strictly enforce rigid rotation around CM
            bbs_list = [new_rel_bb]
        else: # agglomerate
            bbs_list = []
            for bb in local_kinematics:
                new_rel_bb = _transform_fiber_kinematics(np.array(bb), lam, lam_nu)
                bbs_list.append(new_rel_bb)
            all_pts = np.vstack(bbs_list)
            cm_shift = np.mean(all_pts, axis=0)
            bbs_list = [bb - cm_shift for bb in bbs_list]

        all_pts = np.vstack(bbs_list)
        max_radius = radius + 2
        min_b = np.floor(all_pts.min(axis=0)).astype(int) - int(max_radius)
        max_b = np.ceil(all_pts.max(axis=0)).astype(int) + int(max_radius)
        box_shape = tuple(max_b - min_b + 1)

        mask = np.zeros(box_shape, dtype=bool)
        rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        brush = rx**2 + ry**2 + rz**2 <= radius**2
        bz, by, bx = np.where(brush)

        for bb in bbs_list:
            shifted_bb = bb - min_b
            for pt in shifted_bb:
                gz, gy, gx = bz + int(round(pt[0])), by + int(round(pt[1])), bx + int(round(pt[2]))
                valid = (gz>=0)&(gz<box_shape[0])&(gy>=0)&(gy<box_shape[1])&(gx>=0)&(gx<box_shape[2])
                mask[gz[valid], gy[valid], gx[valid]] = True

        coords_nz = np.argwhere(mask > 0)
        if len(coords_nz) == 0: return

        c_mins = coords_nz.min(axis=0)
        c_maxs = coords_nz.max(axis=0)
        cropped = mask[c_mins[0]:c_maxs[0]+1, c_mins[1]:c_maxs[1]+1, c_mins[2]:c_maxs[2]+1]

        cm_in_cropped = -min_b - c_mins
        new_offset = cm_in_cropped - (np.array(cropped.shape) // 2)

    target_center_global = P_CM_new - new_offset
    new_shape = comp_grid.shape
    new_cz = int(round(target_center_global[0])) % new_shape[0]
    new_cy = int(round(target_center_global[1])) % new_shape[1]
    new_cx = int(round(target_center_global[2])) % new_shape[2]

    _paste_mask_to_grid(comp_grid, shell_count_grid, new_cz, new_cy, new_cx, cropped, item['filler_id'], item['inter_id'], is_thermal, tunnel_radius)

def render_deformed_fillers(placement_registry, base_shape, stretch_ratio, poisson_ratio, is_thermal, comp_grid, shell_count_grid, tunnel_radius=2):
    """Renders rigid fillers into the deformed configuration."""
    if stretch_ratio == 1.0:
        return # Skip unnecessary deformation/rendering operations for the initial state

    lam = stretch_ratio
    lam_nu = stretch_ratio ** (-poisson_ratio)
    F_mat = np.diag([lam_nu, lam_nu, lam])
    new_shape = comp_grid.shape

    for item in tqdm(placement_registry, desc=f"Rendering Fillers (Stretch: {stretch_ratio})"):
        geom = item['geom']
        cz, cy, cx = item['center']
        offset = np.array(geom.get('offset_center_to_cm', [0, 0, 0]))

        # Track the absolute True Physical CM in the global grid at Stretch = 1.0
        P_CM_global = np.array([cz, cy, cx], dtype=float) + offset

        # Affine translation of the physical CM
        P_CM_new = F_mat @ P_CM_global

        if geom['base_type'] == 'sphere':
            # Spheres do not rotate, they only translate
            target = P_CM_new - offset
            new_cz = int(round(target[0])) % new_shape[0]
            new_cy = int(round(target[1])) % new_shape[1]
            new_cx = int(round(target[2])) % new_shape[2]
            mask, _ = get_sphere_mask(geom['radius'])
            _paste_mask_to_grid(comp_grid, shell_count_grid, new_cz, new_cy, new_cx, mask, item['filler_id'], item['inter_id'], is_thermal, tunnel_radius)
        else:
            # Route all complex kinematics to the unified dynamic bounding box renderer
            _render_and_paste_kinematics(comp_grid, shell_count_grid, P_CM_new, F_mat, geom, item, is_thermal, tunnel_radius)
