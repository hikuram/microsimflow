import math

import numpy as np
from numba import njit
from scipy.linalg import polar
from scipy.ndimage import affine_transform, binary_dilation, generate_binary_structure
from tqdm.auto import tqdm

from . import builder

def get_spherical_brush(radius):
    """
    Generate an exact spherical footprint.
    Replaces scipy's generate_binary_structure to prevent connectivity rank errors.
    """
    size = int(radius * 2 + 1)
    z, y, x = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    return z**2 + y**2 + x**2 <= radius**2

def _add_shell_safely(shell_grid, z, y, x, add_thick=0, add_thin=0):
    """
    Safely increment the Thick/Thin counters within the lower 7 bits
    without altering the highest bit (128) used for primary contact flags.
    """
    if len(z) == 0:
        return
        
    vals = shell_grid[z, y, x]
    flags = vals & 128
    counts = vals & 127
    
    n_thick = counts % 16
    n_thin = counts // 16
    
    # Prevent overflow: max 15 for thick (4 bits), max 7 for thin (3 bits)
    n_thick = np.minimum(n_thick + add_thick, 15)
    n_thin = np.minimum(n_thin + add_thin, 7)
    
    # Recombine the flags and new counts
    shell_grid[z, y, x] = flags | (n_thin * 16 + n_thick).astype(np.uint8)

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
# Mathematical Utilities for Kinematics & Orientation
# =========================================================

def get_oriented_rotation_matrix(mean_dir=(0.0, 0.0, 1.0), kappa=0.0):
    """
    Generate a 3x3 rotation matrix.
    If kappa == 0: Isotropic random rotation.
    If kappa >  0: Polar alignment along mean_dir.
    If kappa <  0: Girdle (equatorial) alignment perpendicular to mean_dir.
    
    Parameters:
      mean_dir (tuple/list): The target mean direction vector.
      kappa (float): Concentration parameter (0 = random, >0 = aligned to mean_dir).
    """
    # 1. Sample the Z-component (W)
    if abs(kappa) <= 1e-5:
        W = builder.rng.uniform(-1.0, 1.0)
    elif kappa > 0:
        # Polar alignment (vMF style inverse transform sampling)
        U = builder.rng.uniform(0.0, 1.0)
        W = 1.0 + (1.0 / kappa) * np.log(U + (1.0 - U) * np.exp(-2.0 * kappa))
    else:
        # Girdle alignment (Pseudo-Watson truncated normal sampling)
        sigma = 1.0 / np.sqrt(2.0 * abs(kappa))
        while True:
            W = builder.rng.normal(0.0, sigma)
            if abs(W) <= 1.0:
                break
                
    # 2. Sample the azimuthal angle (Theta) uniformly
    theta = builder.rng.uniform(0.0, 2.0 * np.pi)
    
    # 3. Generate the local vector aligned around the Z-axis
    sin_W = np.sqrt(max(0.0, 1.0 - W**2))
    v_z = np.array([sin_W * np.cos(theta), sin_W * np.sin(theta), W])
    
    # 4. Compute the rotation matrix to align the Z-axis with the specified mean_dir
    m_dir = np.array(mean_dir, dtype=float)
    m_norm = np.linalg.norm(m_dir)
    if m_norm > 0:
        m_dir /= m_norm
    else:
        m_dir = np.array([0.0, 0.0, 1.0])
        
    base_z = np.array([0.0, 0.0, 1.0])
    v = np.cross(base_z, m_dir)
    s = np.linalg.norm(v)
    c = np.dot(base_z, m_dir)
    
    if s < 1e-8:
        R_align = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vX = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + vX + (vX @ vX) * ((1.0 - c) / (s**2))
        
    actual_z = R_align @ v_z
    
    # 5. Build the final rotation from base_z to actual_z, adding a random roll twist
    v2 = np.cross(base_z, actual_z)
    s2 = np.linalg.norm(v2)
    c2 = np.dot(base_z, actual_z)
    
    if s2 < 1e-8:
        R_target = np.eye(3) if c2 > 0 else np.diag([1, -1, -1])
    else:
        vX2 = np.array([[0, -v2[2], v2[1]], [v2[2], 0, -v2[0]], [-v2[1], v2[0], 0]])
        R_target = np.eye(3) + vX2 + (vX2 @ vX2) * ((1.0 - c2) / (s2**2))
        
    roll = builder.rng.uniform(0.0, 2.0 * np.pi)
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    return R_target @ R_roll

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

def create_fiber_mask(length, radius, max_bend_deg=90, max_total_bends=10):
    """Generate a mask for a single flexible fiber with fast brush pasting"""
    box_size = int(length * 2 + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    
    start_pos = np.array([box_size//2, box_size//2, box_size//2], dtype=float)
    backbone = [np.round(start_pos).astype(int)]
    current_pos = start_pos
    vec = builder.rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    bends_made = 0

    for _ in range(int(length)):
        next_pos_f = current_pos + vec
        next_pos_i = np.round(next_pos_f).astype(int)
        if any(p < radius or p >= box_size - radius for p in next_pos_i): break
        if bends_made < max_total_bends and builder.rng.random() < (max_total_bends / length):
            angle_rad = np.radians(builder.rng.uniform(10, max_bend_deg))
            noise = builder.rng.standard_normal(3)
            noise -= noise.dot(vec) * vec
            if np.linalg.norm(noise) > 0:
                noise /= np.linalg.norm(noise)
                new_vec = vec * np.cos(angle_rad) + noise * np.sin(angle_rad)
                vec = new_vec / np.linalg.norm(new_vec)
                bends_made += 1
        current_pos = current_pos + vec
        backbone.append(np.round(current_pos).astype(int))

    # Fast vectorized pasting
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    bz, by, bx = np.where(brush)
    
    for (z, y, x) in backbone:
        gz, gy, gx = bz + z - radius, by + y - radius, bx + x - radius
        mask[gz, gy, gx] = True

    cm_global_abs = np.mean(backbone, axis=0)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    local_kinematics = np.array(backbone) - cm_global_abs
    
    geom_data = {
        'base_type': 'fiber', 'radius': radius,
        'R_orig': np.eye(3).tolist(), 'local_kinematics': local_kinematics.tolist(),
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def create_agglomerate_mask(num_fibers, length, radius, max_bend_deg=90, max_total_bends=10, physics_mode='thermal', filler_id=4, inter_id=3):
    box_size = int(length + radius * 2 + 5)
    combined_mask = np.zeros((box_size, box_size, box_size), dtype=np.uint8)
    start_radius = radius * 2
    all_backbones = []
    
    for _ in range(num_fibers):
        fiber_mask, fiber_bb = _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends)
        offset = np.round(builder.rng.standard_normal(3) * start_radius).astype(int)
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

def create_staggered_flakes_mask(radius=15, layer_thickness=2, min_layers=1, max_layers=4, max_offset_pct=20, mean_dir=(0.0, 0.0, 1.0), kappa=0.0):
    """Generate a compound stamp of staggered platelets, rigorously standardizing its centroid."""
    num_layers = builder.rng.integers(min_layers, max_layers + 1)
    max_offset_px = radius * (max_offset_pct / 100.0)
    box_size = int(math.ceil((radius + num_layers * max_offset_px) * 2 + num_layers * layer_thickness)) + 4
    
    # Random Euler angles removed. Replaced with the unified directional rotation matrix.
    R_orig = get_oriented_rotation_matrix(mean_dir=mean_dir, kappa=kappa)
             
    Z, Y, X = np.indices((box_size, box_size, box_size))
    Z = Z - box_size//2; Y = Y - box_size//2; X = X - box_size//2
    
    # Apply inverse rotation (transpose) for global-to-local mapping
    rot = R_orig.T @ np.stack([Z.ravel(), Y.ravel(), X.ravel()])
    
    # Assign rot[2] to Zr, as index 2 is the primary direction axis in local space
    Xr = rot[0,:].reshape((box_size, box_size, box_size))
    Yr = rot[1,:].reshape((box_size, box_size, box_size))
    Zr = rot[2,:].reshape((box_size, box_size, box_size))

    local_centers = []
    cy, cx = 0.0, 0.0
    for i in range(num_layers):
        if i > 0:
            angle = builder.rng.uniform(0, 2 * np.pi)
            cy += builder.rng.uniform(0, max_offset_px) * np.sin(angle)
            cx += builder.rng.uniform(0, max_offset_px) * np.cos(angle)
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

def _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends):
    box_size = int(length + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    center_pos = (box_size // 2, box_size // 2, box_size // 2)
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    bz, by, bx = np.where(brush)
    straight_steps = int(max(3, length * 0.05))
    half_len_a = length // 2
    half_len_b = length - half_len_a
    vec = builder.rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    start_pos = center_pos
    
    def grow_path(steps, v_dir):
        # ... (Keep existing grow_path logic exactly as is) ...
        path = []
        curr = np.array(start_pos, dtype=float)
        v = v_dir.copy()
        bends = 0
        bend_prob = (max_total_bends / 2) / max(1, steps - straight_steps)
        for i in range(steps):
            if i >= straight_steps and bends < (max_total_bends / 2):
                if builder.rng.random() < bend_prob:
                    angle_rad = np.radians(builder.rng.uniform(20, max_bend_deg))
                    noise = builder.rng.standard_normal(3)
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
    
    # Fast vectorized pasting
    for (z, y, x) in backbone:
        gz, gy, gx = bz + z - radius, by + y - radius, bx + x - radius
        mask[gz, gy, gx] = True
        
    return mask, backbone

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

def get_flake_mask(radius, thickness, mean_dir=(0.0, 0.0, 1.0), kappa=0.0, physics_mode='thermal'):
    """
    Generate a flake-shaped filler (platelet) mask with Polar Decomposition support.
    
    Parameters:
      radius (float): Radius of the flake.
      thickness (float): Thickness of the flake.
      mean_dir (tuple/list): Target orientation vector (normal to the flake surface).
      kappa (float): vMF concentration parameter (0 = random, higher = strictly aligned).
    """
    size = int(radius * 2 + 4)
    
    # Apply vMF distribution for orientation control instead of isotropic random Euler angles
    R_orig = get_oriented_rotation_matrix(mean_dir=mean_dir, kappa=kappa)
             
    Z, Y, X = np.indices((size, size, size))
    Z = Z - size//2; Y = Y - size//2; X = X - size//2
    
    # Apply inverse rotation (transpose) for global-to-local mapping
    rot = R_orig.T @ np.stack([Z.ravel(), Y.ravel(), X.ravel()])
    
    # Assign rot[2] to Zr, as index 2 is the primary direction axis in vMF/local space
    Xr = rot[0,:].reshape((size, size, size))
    Yr = rot[1,:].reshape((size, size, size))
    Zr = rot[2,:].reshape((size, size, size))
    
    # Mask generation based on rotated coordinates
    mask = (Xr**2 + Yr**2 <= radius**2) & (np.abs(Zr) <= thickness/2)
    cm_global_abs = np.array([size//2, size//2, size//2], dtype=float)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    
    geom_data = {
        'base_type': 'flake', 'radius': radius, 'thickness': thickness,
        'R_orig': R_orig.tolist(), 'local_kinematics': [[0.0, 0.0, 0.0]], 
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def get_staggered_flakes_mask(radius=15, layer_thickness=2, min_layers=1, max_layers=4, max_offset_pct=30, mean_dir=(0.0, 0.0, 1.0), kappa=0.0, physics_mode='thermal'):
    return create_staggered_flakes_mask(radius, layer_thickness, min_layers, max_layers, max_offset_pct, mean_dir, kappa)

def get_flexible_fiber_mask(length=90, radius=2, max_bend_deg=90, max_total_bends=10, physics_mode='thermal'):
    return create_fiber_mask(length, radius, max_bend_deg, max_total_bends)

def get_irregular_fiber_mask(length, shape_type='ellipse', radius_max=5.0, ratio=0.5, 
                             mean_dir=(0.0, 0.0, 1.0), kappa=0.0, physics_mode='thermal'):
    """
    Generate a rigid straight fiber with an irregular cross-section (Ellipse, Bean, or C-shape).
    
    Parameters:
      shape_type (str): 'ellipse', 'bean', or 'c-shape'
      radius_max (float): Major radius for Ellipse, or Maximum outer radius for Bean/C-shape.
      ratio (float): A scaling factor (0.0 < ratio < 1.0) where radius_max is the denominator.
      mean_dir (tuple/list): Target orientation vector for the fiber axis.
      kappa (float): vMF concentration parameter (0 = random, higher = strictly aligned).
    """
    # Create a sufficient bounding box to accommodate arbitrary 3D rotation
    box_size = int(length + radius_max * 2 + 5)
    
    # Apply vMF distribution for orientation control instead of isotropic random Euler angles
    R_orig = get_oriented_rotation_matrix(mean_dir=mean_dir, kappa=kappa)
             
    Z, Y, X = np.indices((box_size, box_size, box_size))
    Z = Z - box_size//2; Y = Y - box_size//2; X = X - box_size//2
    
    # Apply inverse rotation (transpose) for global-to-local mapping
    rot = R_orig.T @ np.stack([Z.ravel(), Y.ravel(), X.ravel()])
    
    # Assign rot[2] to Zr, as index 2 is the primary direction axis in vMF/local space
    Xr = rot[0,:].reshape((box_size, box_size, box_size))
    Yr = rot[1,:].reshape((box_size, box_size, box_size))
    Zr = rot[2,:].reshape((box_size, box_size, box_size))
    
    # Common longitudinal mask along the local Z-axis
    z_mask = np.abs(Zr) <= length / 2.0
    
    if shape_type == 'ellipse':
        r_min = radius_max * ratio
        xy_mask = (Xr**2 / radius_max**2 + Yr**2 / r_min**2 <= 1)
        
    elif shape_type == 'bean':
        r_min = radius_max * ratio
        # Bend the X direction proportionally to the square of Y (Mathematical model for Kidney/Bean shape)
        X_w = Xr - 0.5 * (Yr**2 / radius_max)
        xy_mask = (X_w**2 / r_min**2 + Yr**2 / radius_max**2 <= 1)
        
    elif shape_type == 'c-shape':
        r_in = radius_max * ratio
        r2 = Xr**2 + Yr**2
        angle = np.arctan2(Yr, Xr)
        # Literal 'C' shape to allow polymer/filler packing inside.
        # 120-degree opening (cut out +-60 degrees), resulting in a typical 240-degree wrap.
        xy_mask = (r2 <= radius_max**2) & (r2 >= r_in**2) & (np.abs(angle) > np.radians(60))
        
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")
        
    mask = xy_mask & z_mask
    
    cm_global_abs = np.array([box_size//2, box_size//2, box_size//2], dtype=float)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    
    geom_data = {
        'base_type': 'irregular_fiber', 
        'shape_type': shape_type,
        'length': length,
        'radius_max': radius_max, 
        'ratio': ratio,
        'radius': radius_max, # Required for dynamic bounding box calculations
        'R_orig': R_orig.tolist(), 
        'local_kinematics': [[0.0, 0.0, 0.0]], 
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def get_rigid_cylinder_mask(length, radius, mean_dir=(0.0, 0.0, 1.0), kappa=0.0, physics_mode='thermal'):
    """
    Generate a rigid short fiber matching the exact original geometry but accelerated 
    without binary_dilation.
    
    Parameters:
      length (float): Length of the fiber backbone.
      radius (float): Cross-sectional radius of the fiber.
      mean_dir (tuple/list): Target orientation vector for the fiber axis.
      kappa (float): vMF concentration parameter (0 = random, higher = strictly aligned).
    """
    box_size = int(length + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    
    # Extract the Z-axis column from the vMF rotation matrix as the growth direction vector
    R_vmf = get_oriented_rotation_matrix(mean_dir=mean_dir, kappa=kappa)
    vec = R_vmf[:, 2] 
    
    start_pos = np.array([box_size//2, box_size//2, box_size//2], dtype=float) - vec * (length / 2)
    
    kinematic_backbone = []
    backbone_idx = []
    current_pos = start_pos
    for _ in range(int(length)):
        kinematic_backbone.append([current_pos[0], current_pos[1], current_pos[2]])
        backbone_idx.append(np.round(current_pos).astype(int))
        current_pos += vec
        
    # Create the spherical brush
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    bz, by, bx = np.where(brush)
    
    # Fast vectorized pasting (Replaces single-threaded binary_dilation)
    for (z, y, x) in backbone_idx:
        gz, gy, gx = bz + z - radius, by + y - radius, bx + x - radius
        mask[gz, gy, gx] = True
    
    cm_global_abs = np.mean(kinematic_backbone, axis=0)
    cropped_mask, offset = crop_and_standardize(mask, cm_global_abs)
    local_kinematics = np.array(kinematic_backbone) - cm_global_abs
    
    geom_data = {
        'base_type': 'fiber', 'radius': radius,
        # Rigid fibers encode orientation entirely within local_kinematics points, 
        # so R_orig is kept as identity to prevent double-rotation during stretch rendering.
        'R_orig': np.eye(3).tolist(), 'local_kinematics': local_kinematics.tolist(),
        'offset_center_to_cm': offset.tolist()
    }
    return cropped_mask, geom_data

def get_agglomerate_mask(num_fibers=5, length=90, radius=2, max_bend_deg=90, max_total_bends=10, physics_mode='thermal'):
    return create_agglomerate_mask(num_fibers, length, radius, max_bend_deg, max_total_bends, physics_mode)

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
    
    vec = builder.rng.standard_normal(3)
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
                    angle_deg = builder.rng.uniform(10, max_bend_deg)
                    
                    # Snap to "180-degree U-turn + side step" if 90 degrees or more
                    if angle_deg >= 90.0:
                        noise = builder.rng.standard_normal(3)
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
                        noise = builder.rng.standard_normal(3)
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

def apply_brush_and_write(comp_grid, backbone, radius, physics_mode='thermal', shell_count_grid=None, filler_id=4, tunnel_radius=2):
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

    # Thermal mode: Identify overlap areas, but do NOT overwrite existing fillers to maintain VF.
    contact_mask = (comp_grid[gz, gy, gx] >= 2)
    
    if shell_count_grid is not None:
        # Use bitwise OR to set the highest bit (128) as the overlap flag in the uint8 array
        shell_count_grid[gz[contact_mask], gy[contact_mask], gx[contact_mask]] |= 128
        
    # Write only non-overlapping bodies to protect existing VF
    body_mask = ~contact_mask
    comp_grid[gz[body_mask], gy[body_mask], gx[body_mask]] = filler_id

    # Compute dual-radii shell for all modes unconditionally
    if shell_count_grid is not None:
        r_thick = tunnel_radius
        r_thin = max(1, int(np.ceil(tunnel_radius / 2.0)))
        
        # 1. Thick shell
        size_thk = int((radius + r_thick) * 2 + 2)
        z_thk, y_thk, x_thk = np.indices((size_thk, size_thk, size_thk))
        cz_thk, cy_thk, cx_thk = size_thk//2, size_thk//2, size_thk//2
        brush_thick = (z_thk - cz_thk)**2 + (y_thk - cy_thk)**2 + (x_thk - cx_thk)**2 <= (radius + r_thick)**2
        local_z_thk, local_y_thk, local_x_thk = np.where(brush_thick)
        
        shell_thick_voxels = set()
        for (z, y, x) in backbone:
            glob_z = (local_z_thk - cz_thk + z) % shape[0]
            glob_y = (local_y_thk - cy_thk + y) % shape[1]
            glob_x = (local_x_thk - cx_thk + x) % shape[2]
            for i in range(len(glob_z)):
                shell_thick_voxels.add((glob_z[i], glob_y[i], glob_x[i]))
                
        if shell_thick_voxels:
            sv_array = np.array(list(shell_thick_voxels))
            _add_shell_safely(shell_count_grid, sv_array[:, 0], sv_array[:, 1], sv_array[:, 2], add_thick=1)

        # 2. Thin shell
        size_thn = int((radius + r_thin) * 2 + 2)
        z_thn, y_thn, x_thn = np.indices((size_thn, size_thn, size_thn))
        cz_thn, cy_thn, cx_thn = size_thn//2, size_thn//2, size_thn//2
        brush_thin = (z_thn - cz_thn)**2 + (y_thn - cy_thn)**2 + (x_thn - cx_thn)**2 <= (radius + r_thin)**2
        local_z_thn, local_y_thn, local_x_thn = np.where(brush_thin)
        
        shell_thin_voxels = set()
        for (z, y, x) in backbone:
            glob_z = (local_z_thn - cz_thn + z) % shape[0]
            glob_y = (local_y_thn - cy_thn + y) % shape[1]
            glob_x = (local_x_thn - cx_thn + x) % shape[2]
            for i in range(len(glob_z)):
                shell_thin_voxels.add((glob_z[i], glob_y[i], glob_x[i]))
                
        if shell_thin_voxels:
            sv_array = np.array(list(shell_thin_voxels))
            _add_shell_safely(shell_count_grid, sv_array[:, 0], sv_array[:, 1], sv_array[:, 2], add_thin=1)

    return len(gz)

def place_adaptive_fibers(comp_grid, tpms_grid, target_vol_frac, length, radius,
                          max_bend_deg=45, max_total_bends=10, max_retries_per_step=10, max_protrusion_ratio=0.1,
                          min_backbone_ratio=0.9, max_attempts=1000000, desc="", log_file=None,
                          physics_mode='thermal', shell_count_grid=None, filler_id=4, tunnel_radius=2):
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

            idx = builder.rng.integers(0, num_valid_coords)
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
            apply_brush_and_write(comp_grid, backbone, radius, physics_mode, shell_count_grid, filler_id, tunnel_radius)

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
                          filler_id):
    """
    Numba-optimized JIT compiled function. (Strictly for Hardcore mode)
    Performs boundary checks, strict collision detection (early exit), and voxel writing.
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
        
    # 2. Overlap Check (Strict Early Exit for Hardcore)
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
        
        # Only write filler_id where space is empty (to protect existing VF).
        if current_val < 2 and new_val > 0:
            comp_grid[tz, ty, tx] = new_val

    return True


@njit
def _evaluate_overlap_softcore_fast(comp_grid, tpms_grid, cz, cy, cx, 
                                    stamp_offsets, current_protrusion_limit):
    """
    Numba-optimized JIT compiled function for Softcore evaluation (Best-of-N).
    Returns -1 if protrusion limit is exceeded, otherwise returns the number of overlapping voxels.
    """
    shape_z, shape_y, shape_x = comp_grid.shape
    num_coords = stamp_offsets.shape[0]
    
    # 1. Protrusion Check
    protrusion_count = 0
    for i in range(num_coords):
        tz = (stamp_offsets[i, 0] + cz) % shape_z
        ty = (stamp_offsets[i, 1] + cy) % shape_y
        tx = (stamp_offsets[i, 2] + cx) % shape_x
        
        if tpms_grid[tz, ty, tx] == 1:
            protrusion_count += 1
            
    if protrusion_count > num_coords * current_protrusion_limit:
        return -1 # Failed protrusion check
        
    # 2. Count Overlaps
    overlap_count = 0
    for i in range(num_coords):
        tz = (stamp_offsets[i, 0] + cz) % shape_z
        ty = (stamp_offsets[i, 1] + cy) % shape_y
        tx = (stamp_offsets[i, 2] + cx) % shape_x
        if comp_grid[tz, ty, tx] >= 2:
            overlap_count += 1
            
    return overlap_count


@njit
def _write_candidate_fast(comp_grid, shell_count_grid, cz, cy, cx, stamp_offsets, stamp_vals):
    """
    Numba-optimized JIT compiled function. 
    Writes the selected best candidate to the grid and tracks primary contacts.
    """
    shape_z, shape_y, shape_x = comp_grid.shape
    num_coords = stamp_offsets.shape[0]
    
    for i in range(num_coords):
        tz = (stamp_offsets[i, 0] + cz) % shape_z
        ty = (stamp_offsets[i, 1] + cy) % shape_y
        tx = (stamp_offsets[i, 2] + cx) % shape_x
        
        current_val = comp_grid[tz, ty, tx]
        new_val = stamp_vals[i]
        
        if current_val >= 2:
            # Overlap! Set the primary contact flag (highest bit: 128)
            shell_count_grid[tz, ty, tx] |= 128
        elif new_val > 0:
            # Only write filler_id where space is empty
            comp_grid[tz, ty, tx] = new_val


def place_fillers_hybrid(comp_grid, tpms_grid, filler_func, kwargs, target_vol_frac,
                         max_attempts=1000000, fallback_func=None, desc="",
                         protrusion_coef=0.0025, log_file=None,
                         physics_mode='thermal', shell_count_grid=None,
                         filler_id=4, tunnel_radius=2, placement_registry=None):
    
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
    shell_thick_offsets = None
    shell_thin_offsets = None
    cache_reuse_count = 0
    MAX_CACHE_REUSE = 50
    current_protrusion_limit = 0.0 

    if 'physics_mode' not in kwargs:
        kwargs['physics_mode'] = physics_mode

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

                # Extract dual-radii shells for ALL modes if shell_count_grid is provided
                if shell_count_grid is not None:
                    # Automatically define dual radii from the single tunnel_radius parameter
                    r_thick = tunnel_radius
                    r_thin = max(1, int(np.ceil(tunnel_radius / 2.0)))
                    
                    # Pad the stamp to prevent dilation from clipping at the bounding box edges
                    padded_stamp = np.pad(raw_stamp > 0, pad_width=r_thick, mode='constant', constant_values=False)
                    padded_center = center + r_thick
                    
                    brush_thick = get_spherical_brush(r_thick)
                    dilated_thick = binary_dilation(padded_stamp, structure=brush_thick)
                    shell_thick_offsets = np.argwhere(dilated_thick) - padded_center

                    brush_thin = get_spherical_brush(r_thin)
                    dilated_thin = binary_dilation(padded_stamp, structure=brush_thin)
                    shell_thin_offsets = np.argwhere(dilated_thin) - padded_center
                else:
                    shell_thick_offsets = None
                    shell_thin_offsets = None

                filler_voxels = len(coords)
                current_protrusion_limit = calculate_protrusion_limit(filler_voxels, total_voxels, protrusion_coef)
                cache_reuse_count = 0

            # Execute placement logic based on the current mode
            if not overlap_mode:
                # --- Hardcore Mode: Ultra-fast single random pick and strict placement ---
                idx = builder.rng.integers(0, num_valid_coords)
                cz, cy, cx = valid_z[idx], valid_y[idx], valid_x[idx]

                success = _check_and_place_fast(
                    comp_grid, tpms_grid, cz, cy, cx, 
                    stamp_offsets, stamp_vals, current_protrusion_limit, 
                    filler_id
                )
                
            else:
                # --- Softcore Mode: Best-of-N (N=3) approach to minimize overlaps ---
                best_score = float('inf')
                best_coord = None
                
                # Sample 3 candidates
                for _ in range(3):
                    idx = builder.rng.integers(0, num_valid_coords)
                    cand_z, cand_y, cand_x = valid_z[idx], valid_y[idx], valid_x[idx]
                    
                    score = _evaluate_overlap_softcore_fast(
                        comp_grid, tpms_grid, cand_z, cand_y, cand_x, 
                        stamp_offsets, current_protrusion_limit
                    )
                    
                    # -1 means protrusion failed. Valid score implies acceptable boundary conditions
                    if score != -1 and score < best_score:
                        best_score = score
                        best_coord = (cand_z, cand_y, cand_x)
                        # Perfect spot found, no need to evaluate further
                        if score == 0:
                            break
                            
                if best_coord is not None:
                    cz, cy, cx = best_coord
                    _write_candidate_fast(comp_grid, shell_count_grid, cz, cy, cx, stamp_offsets, stamp_vals)
                    success = True
                else:
                    success = False

            if not success:
                cache_reuse_count += 1
                consecutive_fails += 1
                continue

            # Increment Thick shell counter
            if shell_thick_offsets is not None:
                stz = (shell_thick_offsets[:, 0] + cz) % shape[0]
                sty = (shell_thick_offsets[:, 1] + cy) % shape[1]
                stx = (shell_thick_offsets[:, 2] + cx) % shape[2]
                _add_shell_safely(shell_count_grid, stz, sty, stx, add_thick=1)

            # Increment Thin shell counter
            if shell_thin_offsets is not None:
                stz_th = (shell_thin_offsets[:, 0] + cz) % shape[0]
                sty_th = (shell_thin_offsets[:, 1] + cy) % shape[1]
                stx_th = (shell_thin_offsets[:, 2] + cx) % shape[2]
                _add_shell_safely(shell_count_grid, stz_th, sty_th, stx_th, add_thin=1)
            
            # Update progress
            current_total = np.sum(comp_grid >= 2)
            added_voxels = current_total - placed_voxels
            placed_voxels = current_total

            # Record successful placement geometry
            if placement_registry is not None:
                # Store the original stamp for coarse deformation mode.
                raw_offsets = stamp_offsets.astype(np.int16, copy=True)
                raw_vals = stamp_vals.astype(np.uint8, copy=True)
                if shell_thick_offsets is None:
                    raw_shell_thick_offsets = None
                    raw_shell_thin_offsets = None
                else:
                    raw_shell_thick_offsets = shell_thick_offsets.astype(np.int16, copy=True)
                    raw_shell_thin_offsets = shell_thin_offsets.astype(np.int16, copy=True)

                placement_registry.append({
                    'geom': current_geom_data,
                    'center': (cz, cy, cx),
                    'filler_id': filler_id,
                    'raw_offsets': raw_offsets,
                    'raw_vals': raw_vals,
                    'raw_shell_offsets': raw_shell_thick_offsets,
                    'raw_shell_thin_offsets': raw_shell_thin_offsets,
                    'raw_voxel_count': int(len(raw_offsets)),
                    'added_voxel_count': int(added_voxels)
                })
                
            # Reset cache and fails only on success (Keep cache if it's a sphere)
            if current_geom_data.get('base_type') != 'sphere':
                stamp_offsets = None 
            consecutive_fails = 0

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
# F. Affine Deformation & Rendering Module
# =========================================================

def apply_background_deformation(grid, stretch_ratio=1.0, poisson_ratio=0.4, stretch_axis='X'):
    """Applies affine deformation ONLY to the continuous polymer background."""
    if stretch_ratio == 1.0:
        return grid.copy()

    lam = stretch_ratio
    lam_nu = lam ** (-poisson_ratio)
    nz, ny, nx = grid.shape
    
    if stretch_axis == 'X':
        F_diag = [lam_nu, lam_nu, lam]
        new_shape = (
            max(1, int(round(nz * lam_nu))),
            max(1, int(round(ny * lam_nu))),
            max(1, int(round(nx * lam)))
        )
    elif stretch_axis == 'Y':
        F_diag = [lam_nu, lam, lam_nu]
        new_shape = (
            max(1, int(round(nz * lam_nu))),
            max(1, int(round(ny * lam))),
            max(1, int(round(nx * lam_nu)))
        )
    elif stretch_axis == 'Z':
        F_diag = [lam, lam_nu, lam_nu]
        new_shape = (
            max(1, int(round(nz * lam))),
            max(1, int(round(ny * lam_nu))),
            max(1, int(round(nx * lam_nu)))
        )
    else:
        raise ValueError(f"Invalid stretch_axis: {stretch_axis}. Must be X, Y, or Z.")

    matrix = np.diag([1.0 / F_diag[0], 1.0 / F_diag[1], 1.0 / F_diag[2]])
    return affine_transform(grid, matrix=matrix, output_shape=new_shape, order=0, mode='wrap')

def _transform_fiber_kinematics(local_bb, F_diag):
    """Calculates rigid-body transformation of an inextensible fiber backbone"""
    diffs = np.diff(local_bb, axis=0)
    diffs_def = np.zeros_like(diffs, dtype=float)
    diffs_def[:, 0] = diffs[:, 0] * F_diag[0]
    diffs_def[:, 1] = diffs[:, 1] * F_diag[1]
    diffs_def[:, 2] = diffs[:, 2] * F_diag[2]

    orig_lens = np.linalg.norm(diffs, axis=1)
    def_lens = np.linalg.norm(diffs_def, axis=1)
    valid = def_lens > 1e-8
    diffs_def[valid] = (diffs_def[valid] / def_lens[valid, None]) * orig_lens[valid, None]

    new_bb = np.zeros((len(local_bb), 3), dtype=float)
    new_bb[1:] = np.cumsum(diffs_def, axis=0)
    start_anchor = local_bb[0] * np.array(F_diag)
    new_bb += (start_anchor - new_bb[0])
    return new_bb

def _paste_mask_to_grid(comp_grid, shell_count_grid, cz, cy, cx, mask, filler_id, tunnel_radius):
    """Helper to paste a generic boolean mask (flakes/spheres) into the grid"""
    shape = comp_grid.shape
    coords = np.argwhere(mask > 0)
    if len(coords) == 0: return
    
    center = np.array(mask.shape) // 2
    offsets = coords - center
    
    gz = (offsets[:, 0] + cz) % shape[0]
    gy = (offsets[:, 1] + cy) % shape[1]
    gx = (offsets[:, 2] + cx) % shape[2]

    contact = (comp_grid[gz, gy, gx] >= 2)
    
    if shell_count_grid is not None:
        # Set overlap flag using bitwise OR (128)
        shell_count_grid[gz[contact], gy[contact], gx[contact]] |= 128
        
    # Protect existing VF
    comp_grid[gz[~contact], gy[~contact], gx[~contact]] = filler_id

    # Compute dual-radii shell for all modes unconditionally
    if shell_count_grid is not None:
        r_thick = tunnel_radius
        r_thin = max(1, int(np.ceil(tunnel_radius / 2.0)))
        
        # Pad the dynamically generated mask to prevent clipping
        padded_mask = np.pad(mask, pad_width=r_thick, mode='constant', constant_values=False)
        padded_center = center + r_thick

        # Thick shell
        brush_thick = get_spherical_brush(r_thick)
        dilated_thick = binary_dilation(padded_mask, structure=brush_thick)
        shell_coords_thick = np.argwhere(dilated_thick)
        sh_offsets_thick = shell_coords_thick - padded_center
        sz = (sh_offsets_thick[:, 0] + cz) % shape[0]
        sy = (sh_offsets_thick[:, 1] + cy) % shape[1]
        sx = (sh_offsets_thick[:, 2] + cx) % shape[2]
        _add_shell_safely(shell_count_grid, sz, sy, sx, add_thick=1)

        # Thin shell
        brush_thin = get_spherical_brush(r_thin)
        dilated_thin = binary_dilation(padded_mask, structure=brush_thin)
        shell_coords_thin = np.argwhere(dilated_thin)
        sh_offsets_thin = shell_coords_thin - padded_center
        sz_th = (sh_offsets_thin[:, 0] + cz) % shape[0]
        sy_th = (sh_offsets_thin[:, 1] + cy) % shape[1]
        sx_th = (sh_offsets_thin[:, 2] + cx) % shape[2]
        _add_shell_safely(shell_count_grid, sz_th, sy_th, sx_th, add_thin=1)

def _paste_offsets_to_grid(comp_grid, shell_count_grid, cz, cy, cx, offsets, vals,
                           filler_id, tunnel_radius, shell_offsets=None, shell_thin_offsets=None):
    """Paste stored voxel offsets into the grid without re-voxelizing the filler."""
    shape = comp_grid.shape
    if offsets is None or len(offsets) == 0:
        return 0

    offsets = np.asarray(offsets)
    gz = (offsets[:, 0] + cz) % shape[0]
    gy = (offsets[:, 1] + cy) % shape[1]
    gx = (offsets[:, 2] + cx) % shape[2]

    contact = (comp_grid[gz, gy, gx] >= 2)

    if shell_count_grid is not None:
        # Set overlap flag using bitwise OR (128)
        shell_count_grid[gz[contact], gy[contact], gx[contact]] |= 128

    if vals is None:
        paste_vals = np.full(len(offsets), filler_id, dtype=np.uint8)
    else:
        paste_vals = np.asarray(vals, dtype=np.uint8)

    written = int(np.count_nonzero(~contact))
    comp_grid[gz[~contact], gy[~contact], gx[~contact]] = paste_vals[~contact]

    # Compute shell for all modes unconditionally
    if shell_count_grid is not None:
        if shell_offsets is not None:
            shell_offsets = np.asarray(shell_offsets)
            sz = (shell_offsets[:, 0] + cz) % shape[0]
            sy = (shell_offsets[:, 1] + cy) % shape[1]
            sx = (shell_offsets[:, 2] + cx) % shape[2]
            _add_shell_safely(shell_count_grid, sz, sy, sx, add_thick=1)
            
        if shell_thin_offsets is not None:
            shell_thin_offsets = np.asarray(shell_thin_offsets)
            sz_th = (shell_thin_offsets[:, 0] + cz) % shape[0]
            sy_th = (shell_thin_offsets[:, 1] + cy) % shape[1]
            sx_th = (shell_thin_offsets[:, 2] + cx) % shape[2]
            _add_shell_safely(shell_count_grid, sz_th, sy_th, sx_th, add_thin=1)
            
    return written


def _count_new_voxels_for_mask(comp_grid, cz, cy, cx, mask):
    """Count voxels that would be newly occupied by a candidate mask."""
    shape = comp_grid.shape
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0

    center = np.array(mask.shape) // 2
    offsets = coords - center
    gz = (offsets[:, 0] + cz) % shape[0]
    gy = (offsets[:, 1] + cy) % shape[1]
    gx = (offsets[:, 2] + cx) % shape[2]
    return int(np.count_nonzero(comp_grid[gz, gy, gx] < 2))


def _rotation_matrix_from_axis_angle(axis, angle_rad):
    """Create a 3D rotation matrix from an axis and an angle."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle_rad) < 1e-15:
        return np.eye(3)

    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _orthonormal_tilt_axes(main_axis):
    """Return two unit axes perpendicular to the given main axis."""
    main_axis = np.asarray(main_axis, dtype=float)
    norm = np.linalg.norm(main_axis)
    if norm < 1e-12:
        main_axis = np.array([0.0, 0.0, 1.0])
    else:
        main_axis = main_axis / norm

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(main_axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u_axis = np.cross(main_axis, ref)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(main_axis, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return u_axis, v_axis


def _quick_tilt_candidates(main_axis, max_tilt_deg):
    """Build small two-axis tilt candidates for fine deformation mode."""
    u_axis, v_axis = _orthonormal_tilt_axes(main_axis)
    candidates = [(np.eye(3), 0.0)]

    base_levels = [0.10, 0.25, 0.50]
    levels = [deg for deg in base_levels if deg <= max_tilt_deg + 1e-12]
    if max_tilt_deg > 0 and not levels:
        levels = [max_tilt_deg]
    elif max_tilt_deg > 0 and levels and abs(levels[-1] - max_tilt_deg) > 1e-12 and max_tilt_deg < base_levels[-1]:
        levels.append(max_tilt_deg)

    for deg in levels:
        for sign in (-1.0, 1.0):
            angle_rad = math.radians(sign * deg)
            candidates.append((_rotation_matrix_from_axis_angle(u_axis, angle_rad), deg))
            candidates.append((_rotation_matrix_from_axis_angle(v_axis, angle_rad), deg))

    return candidates


def _pca_main_axis(points):
    """Estimate the main axis of a point cloud."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.array([0.0, 0.0, 1.0])

    pts = pts - np.mean(pts, axis=0)
    if np.max(np.linalg.norm(pts, axis=1)) < 1e-12:
        return np.array([0.0, 0.0, 1.0])

    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    return vh[0]


def _get_deformed_main_axis(geom, F_mat):
    """Estimate the primary direction used by the two-axis tilt search."""
    base_type = geom['base_type']
    R_orig = np.array(geom.get('R_orig', np.eye(3)))
    local_kinematics = geom.get('local_kinematics', [[0.0, 0.0, 0.0]])

    if base_type in ['flake', 'staggered', 'irregular_fiber']:
        R_local_to_global = np.array(R_orig)
        R_pure_local_to_global, _ = polar(F_mat @ R_local_to_global)
        return R_pure_local_to_global @ np.array([0.0, 0.0, 1.0])

    if base_type == 'fiber':
        F_diag = [F_mat[0, 0], F_mat[1, 1], F_mat[2, 2]]
        new_rel_bb = _transform_fiber_kinematics(np.array(local_kinematics), F_diag)
        if len(new_rel_bb) >= 2:
            axis = new_rel_bb[-1] - new_rel_bb[0]
            if np.linalg.norm(axis) > 1e-12:
                return axis
        return _pca_main_axis(new_rel_bb)

    if base_type == 'agglomerate':
        F_diag = [F_mat[0, 0], F_mat[1, 1], F_mat[2, 2]]
        all_pts = []
        for bb in local_kinematics:
            new_rel_bb = _transform_fiber_kinematics(np.array(bb), F_diag)
            all_pts.append(new_rel_bb)
        return _pca_main_axis(np.vstack(all_pts))

    return np.array([0.0, 0.0, 1.0])


def _build_kinematic_mask(F_mat, geom, tilt_rotation=None):
    """
    Dynamically renders transformed local kinematics into a tight bounding box,
    drastically reducing memory overhead and preserving exact rigid-body volume.
    """
    base_type = geom['base_type']
    R_orig = np.array(geom.get('R_orig', np.eye(3)))
    radius = geom.get('radius', 0)
    local_kinematics = geom.get('local_kinematics', [[0.0, 0.0, 0.0]])
    if tilt_rotation is None:
        tilt_rotation = np.eye(3)

    if base_type in ['flake', 'staggered', 'irregular_fiber']:
        R_local_to_global = np.array(R_orig)
        R_pure_local_to_global, _ = polar(F_mat @ R_local_to_global)
        R_pure_local_to_global = tilt_rotation @ R_pure_local_to_global

        loc_pts = np.array(local_kinematics)
        new_rel_global_centers = (R_pure_local_to_global @ loc_pts.T).T

        if base_type == 'irregular_fiber':
            effective_radius = geom['length'] / 2.0 + radius
        else:
            effective_radius = radius

        max_radius = effective_radius + 2
        min_b = np.floor(new_rel_global_centers.min(axis=0)).astype(int) - int(max_radius)
        max_b = np.ceil(new_rel_global_centers.max(axis=0)).astype(int) + int(max_radius)
        box_shape = tuple(max_b - min_b + 1)

        Z, Y, X = np.mgrid[min_b[0]:max_b[0]+1, min_b[1]:max_b[1]+1, min_b[2]:max_b[2]+1]
        coords_global_shifted = np.stack([Z.ravel(), Y.ravel(), X.ravel()])

        R_global_to_local = R_pure_local_to_global.T
        coords_local = R_global_to_local @ coords_global_shifted

        X_loc = coords_local[0,:].reshape(box_shape)
        Y_loc = coords_local[1,:].reshape(box_shape)
        Z_loc = coords_local[2,:].reshape(box_shape)

        mask = np.zeros(box_shape, dtype=bool)

        if base_type == 'staggered':
            layer_thickness = geom['layer_thickness']
            for z_c, y_c, x_c in local_kinematics:
                mask |= ((X_loc - x_c)**2 + (Y_loc - y_c)**2 <= radius**2) & (np.abs(Z_loc - z_c) <= layer_thickness / 2.0)
        elif base_type == 'flake':
            thickness = geom['thickness']
            z_c, y_c, x_c = local_kinematics[0]
            mask |= ((X_loc - x_c)**2 + (Y_loc - y_c)**2 <= radius**2) & (np.abs(Z_loc - z_c) <= thickness / 2.0)

        elif base_type == 'irregular_fiber':
            length = geom['length']
            shape_type = geom['shape_type']
            r_max = geom['radius_max']
            ratio = geom['ratio']
            z_c, y_c, x_c = local_kinematics[0]

            Z_rel = Z_loc - z_c
            Y_rel = Y_loc - y_c
            X_rel = X_loc - x_c

            z_mask = np.abs(Z_rel) <= length / 2.0

            if shape_type == 'ellipse':
                r_min = r_max * ratio
                xy_mask = (X_rel**2 / r_max**2 + Y_rel**2 / r_min**2 <= 1)

            elif shape_type == 'bean':
                r_min = r_max * ratio
                X_w = X_rel - 0.5 * (Y_rel**2 / r_max)
                xy_mask = (X_w**2 / r_min**2 + Y_rel**2 / r_max**2 <= 1)

            elif shape_type == 'c-shape':
                r_in = r_max * ratio
                r2 = X_rel**2 + Y_rel**2
                angle = np.arctan2(Y_rel, X_rel)
                xy_mask = (r2 <= r_max**2) & (r2 >= r_in**2) & (np.abs(angle) > np.radians(60))

            mask |= (xy_mask & z_mask)

        coords_nz = np.argwhere(mask > 0)
        if len(coords_nz) == 0:
            return None, None

        c_mins = coords_nz.min(axis=0)
        c_maxs = coords_nz.max(axis=0)
        cropped = mask[c_mins[0]:c_maxs[0]+1, c_mins[1]:c_maxs[1]+1, c_mins[2]:c_maxs[2]+1]

        cm_in_cropped = -min_b - c_mins
        new_offset = cm_in_cropped - (np.array(cropped.shape) // 2)
        return cropped, new_offset

    elif base_type in ['fiber', 'agglomerate']:
        F_diag = [F_mat[0, 0], F_mat[1, 1], F_mat[2, 2]]

        if base_type == 'fiber':
            new_rel_bb = _transform_fiber_kinematics(np.array(local_kinematics), F_diag)
            new_rel_bb -= np.mean(new_rel_bb, axis=0)
            new_rel_bb = (tilt_rotation @ new_rel_bb.T).T
            bbs_list = [new_rel_bb]
        else: 
            bbs_list = []
            for bb in local_kinematics:
                new_rel_bb = _transform_fiber_kinematics(np.array(bb), F_diag)
                bbs_list.append(new_rel_bb)
            all_pts = np.vstack(bbs_list)
            cm_shift = np.mean(all_pts, axis=0)
            bbs_list = [((tilt_rotation @ (bb - cm_shift).T).T) for bb in bbs_list]

        all_pts = np.vstack(bbs_list)
        max_radius = radius + 2
        min_b = np.floor(all_pts.min(axis=0)).astype(int) - int(max_radius)
        max_b = np.ceil(all_pts.max(axis=0)).astype(int) + int(max_radius)
        box_shape = tuple(max_b - min_b + 1)

        mask = np.zeros(box_shape, dtype=bool)
        brush_radius = int(round(radius))

        rz, ry, rx = np.ogrid[
            -brush_radius:brush_radius+1,
            -brush_radius:brush_radius+1,
            -brush_radius:brush_radius+1
        ]
        brush = rx**2 + ry**2 + rz**2 <= radius**2
        bz, by, bx = np.where(brush)

        for bb in bbs_list:
            shifted_bb = bb - min_b
            for pt in shifted_bb:
                gz = bz + int(round(pt[0])) - brush_radius
                gy = by + int(round(pt[1])) - brush_radius
                gx = bx + int(round(pt[2])) - brush_radius

                valid = (
                    (gz >= 0) & (gz < box_shape[0]) &
                    (gy >= 0) & (gy < box_shape[1]) &
                    (gx >= 0) & (gx < box_shape[2])
                )
                mask[gz[valid], gy[valid], gx[valid]] = True

        coords_nz = np.argwhere(mask > 0)
        if len(coords_nz) == 0:
            return None, None

        c_mins = coords_nz.min(axis=0)
        c_maxs = coords_nz.max(axis=0)
        cropped = mask[c_mins[0]:c_maxs[0]+1, c_mins[1]:c_maxs[1]+1, c_mins[2]:c_maxs[2]+1]

        cm_in_cropped = -min_b - c_mins
        new_offset = cm_in_cropped - (np.array(cropped.shape) // 2)
        return cropped, new_offset

    return None, None


def _candidate_center_from_offset(P_CM_new, new_offset, grid_shape):
    """Convert a candidate CM offset to a periodic paste center."""
    target_center_global = P_CM_new - new_offset
    return (
        int(round(target_center_global[0])) % grid_shape[0],
        int(round(target_center_global[1])) % grid_shape[1],
        int(round(target_center_global[2])) % grid_shape[2],
    )


def _render_and_paste_kinematics(comp_grid, shell_count_grid, P_CM_new, F_mat, geom, item, tunnel_radius):
    """
    Dynamically renders transformed local kinematics into a tight bounding box,
    drastically reducing memory overhead and preserving exact rigid-body volume.
    """
    cropped, new_offset = _build_kinematic_mask(F_mat, geom)
    if cropped is None:
        return 0

    new_cz, new_cy, new_cx = _candidate_center_from_offset(P_CM_new, new_offset, comp_grid.shape)
    before = _count_new_voxels_for_mask(comp_grid, new_cz, new_cy, new_cx, cropped)
    _paste_mask_to_grid(comp_grid, shell_count_grid, new_cz, new_cy, new_cx, cropped, item['filler_id'], tunnel_radius)
    return before

def _render_coarse_item(comp_grid, shell_count_grid, P_CM_new, geom, item, tunnel_radius):
    """Render one item in coarse mode using the saved raw stamp offsets."""
    raw_offsets = item.get('raw_offsets')
    if raw_offsets is None:
        return 0

    offset = np.array(geom.get('offset_center_to_cm', [0, 0, 0]))
    new_cz, new_cy, new_cx = _candidate_center_from_offset(P_CM_new, offset, comp_grid.shape)
    
    return _paste_offsets_to_grid(
        comp_grid, shell_count_grid, new_cz, new_cy, new_cx,
        raw_offsets, item.get('raw_vals'), item['filler_id'], tunnel_radius,
        shell_offsets=item.get('raw_shell_offsets'),
        shell_thin_offsets=item.get('raw_shell_thin_offsets')
    )

def _render_fine_item(comp_grid, shell_count_grid, P_CM_new, F_mat, geom, item,
                      tunnel_radius, volume_error_ledger, fine_volume_tol,
                      fine_max_tilt_deg, fine_ledger_cap):
    """Render one item in fine mode using small two-axis tilt candidates."""
    target_added = int(item.get('added_voxel_count', item.get('raw_voxel_count', 0)))

    if geom['base_type'] == 'sphere':
        # Spheres do not rotate, they only translate
        offset = np.array(geom.get('offset_center_to_cm', [0, 0, 0]))
        target = P_CM_new - offset
        new_cz = int(round(target[0])) % comp_grid.shape[0]
        new_cy = int(round(target[1])) % comp_grid.shape[1]
        new_cx = int(round(target[2])) % comp_grid.shape[2]
        mask, _ = get_sphere_mask(geom['radius'])
        actual_added = _count_new_voxels_for_mask(comp_grid, new_cz, new_cy, new_cx, mask)
        _paste_mask_to_grid(comp_grid, shell_count_grid, new_cz, new_cy, new_cx, mask, item['filler_id'], tunnel_radius)
        return actual_added, 0.0, target_added

    main_axis = _get_deformed_main_axis(geom, F_mat)
    tilt_candidates = _quick_tilt_candidates(main_axis, fine_max_tilt_deg)

    # The ledger is used gradually so that a single particle does not absorb all prior error.
    local_cap = int(max(1, round(fine_ledger_cap * max(1, target_added))))
    desired_added = target_added + int(np.clip(volume_error_ledger, -local_cap, local_cap))

    best = None
    for tilt_rotation, tilt_deg in tilt_candidates:
        cropped, new_offset = _build_kinematic_mask(F_mat, geom, tilt_rotation=tilt_rotation)
        if cropped is None:
            continue

        new_cz, new_cy, new_cx = _candidate_center_from_offset(P_CM_new, new_offset, comp_grid.shape)
        candidate_added = _count_new_voxels_for_mask(comp_grid, new_cz, new_cy, new_cx, cropped)
        score = (
            abs(candidate_added - desired_added),
            abs(tilt_deg),
            abs(candidate_added - target_added),
        )
        if best is None or score < best['score']:
            best = {
                'score': score,
                'mask': cropped,
                'center': (new_cz, new_cy, new_cx),
                'added': candidate_added,
                'tilt_deg': tilt_deg,
            }

    if best is None:
        # Fall back to coarse raw offsets if the fine renderer cannot build a mask.
        actual_added = _render_coarse_item(comp_grid, shell_count_grid, P_CM_new, geom, item, tunnel_radius)
        return actual_added, 0.0, target_added

    new_cz, new_cy, new_cx = best['center']
    _paste_mask_to_grid(comp_grid, shell_count_grid, new_cz, new_cy, new_cx,
                        best['mask'], item['filler_id'], tunnel_radius)
    return int(best['added']), float(best['tilt_deg']), target_added


def render_deformed_fillers(placement_registry, base_shape, stretch_ratio, poisson_ratio,
                            comp_grid, shell_count_grid, stretch_axis='X', tunnel_radius=2,
                            deformation_mode='fine', fine_volume_tol=0.01,
                            fine_max_tilt_deg=0.5, fine_ledger_cap=0.05):
    """Renders rigid fillers into the deformed configuration."""
    lam = stretch_ratio
    lam_nu = stretch_ratio ** (-poisson_ratio)
    
    if stretch_axis == 'X':
        F_diag = [lam_nu, lam_nu, lam]
    elif stretch_axis == 'Y':
        F_diag = [lam_nu, lam, lam_nu]
    elif stretch_axis == 'Z':
        F_diag = [lam, lam_nu, lam_nu]
    else:
        raise ValueError(f"Invalid stretch_axis: {stretch_axis}. Must be X, Y, or Z.")
        
    F_mat = np.diag(F_diag)
    deformation_mode = str(deformation_mode).lower()

    if deformation_mode not in ('coarse', 'fine'):
        raise ValueError("deformation_mode must be 'coarse' or 'fine'")

    volume_error_ledger = 0
    total_target_added = 0
    total_actual_added = 0
    max_abs_ledger = 0
    out_of_tol_count = 0

    for item in tqdm(placement_registry, desc=f"Rendering Fillers (Stretch: {stretch_ratio}, Mode: {deformation_mode})"):
        geom = item['geom']
        cz, cy, cx = item['center']
        offset = np.array(geom.get('offset_center_to_cm', [0, 0, 0]))

        # Track the absolute True Physical CM in the global grid at Stretch = 1.0
        P_CM_global = np.array([cz, cy, cx], dtype=float) + offset

        # Affine translation of the physical CM
        P_CM_new = F_mat @ P_CM_global

        if deformation_mode == 'coarse':
            _render_coarse_item(comp_grid, shell_count_grid, P_CM_new, geom, item, tunnel_radius)
            continue

        actual_added, tilt_deg, target_added = _render_fine_item(
            comp_grid, shell_count_grid, P_CM_new, F_mat, geom, item,
            tunnel_radius, volume_error_ledger, fine_volume_tol,
            fine_max_tilt_deg, fine_ledger_cap
        )
        total_target_added += target_added
        total_actual_added += actual_added
        volume_error_ledger += target_added - actual_added
        max_abs_ledger = max(max_abs_ledger, abs(volume_error_ledger))

        tol_voxels = max(1, int(math.ceil(fine_volume_tol * max(1, target_added))))
        if abs(actual_added - target_added) > tol_voxels:
            out_of_tol_count += 1

    if deformation_mode == 'fine':
        print(
            f"Fine deformation volume summary: "
            f"target_added={total_target_added}, actual_added={total_actual_added}, "
            f"ledger={volume_error_ledger}, max_abs_ledger={max_abs_ledger}, "
            f"out_of_tol_particles={out_of_tol_count}"
        )
