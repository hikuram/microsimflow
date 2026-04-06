import math
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm.auto import tqdm
import pyvista as pv

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

def create_fiber_mask(length, radius, max_bend_deg=90, max_total_bends=10):
    """Generate a mask for a single flexible fiber"""
    # A space large enough to definitely contain the fiber
    box_size = int(length * 2 + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    
    start_pos = (box_size//2, box_size//2, box_size//2)
    backbone = [start_pos]
    current_pos = np.array(start_pos, dtype=float)
    
    vec = rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    bends_made = 0

    for _ in range(int(length)):
        next_pos_f = current_pos + vec
        next_pos_i = np.round(next_pos_f).astype(int)
        
        # Safety check for box boundaries
        if any(p < radius or p >= box_size - radius for p in next_pos_i):
            break

        # Naturally bend with a certain probability according to the specified total number of bends and length
        if bends_made < max_total_bends and rng.random() < (max_total_bends / length):
            angle_rad = np.radians(rng.uniform(10, max_bend_deg))
            noise = rng.standard_normal(3)
            noise -= noise.dot(vec) * vec
            if np.linalg.norm(noise) > 0:
                noise /= np.linalg.norm(noise)
                new_vec = vec * np.cos(angle_rad) + noise * np.sin(angle_rad)
                new_vec /= np.linalg.norm(new_vec)
                vec = new_vec
                bends_made += 1
        
        current_pos = current_pos + vec
        backbone.append(np.round(current_pos).astype(int))

    # Flesh out with a brush
    # ==========================================
    # Dramatic speed-up point (Batch fleshing out by Dilation)
    # ==========================================
    # 1. Set only the backbone (core) pixels to True
    for (bz, by, bx) in backbone:
        if 0 <= bz < box_size and 0 <= by < box_size and 0 <= bx < box_size:
            mask[bz, by, bx] = True
            
    # 2. Create a spherical structuring element (brush)
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    
    # 3. Flesh out all at once using dilation processing
    mask = binary_dilation(mask, structure=brush)

    return mask

def _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends):
    """Single fiber mask for agglomerates that grows to both sides from the center and random walks midway"""
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
        # Adjust bending probability according to remaining steps
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
    # Combine both sides across the start position
    backbone = backbone_b[::-1] + [tuple(start_pos)] + backbone_a
    
    for (z, y, x) in backbone:
        mask[z, y, x] = True
        
    mask = binary_dilation(mask, structure=brush)
    return mask

def create_agglomerate_mask(num_fibers, length, radius, max_bend_deg=90, max_total_bends=10, physics_mode='thermal', filler_id=3, inter_id=2):
    """Generate an agglomerate mask of multiple entangled fibers, crop, and return"""
    # The fiber grows a max of length/2 from the center, so box size is scaled down based on length
    box_size = int(length + radius * 2 + 5)
    combined_mask = np.zeros((box_size, box_size, box_size), dtype=np.uint8)
    
    center = box_size // 2
    start_radius = radius * 2
    
    for _ in range(num_fibers):
        # Generate bilateral growth fibers specialized for agglomerates
        fiber_mask = _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends)
        
        # Shift the center position slightly and composite
        offset = np.round(rng.standard_normal(3) * start_radius).astype(int)
        
        # Slice combination processing exactly identical to the existing code
        shift_z, shift_y, shift_x = offset
        
        z_start = max(0, shift_z)
        z_end = min(box_size, box_size + shift_z)
        y_start = max(0, shift_y)
        y_end = min(box_size, box_size + shift_y)
        x_start = max(0, shift_x)
        x_end = min(box_size, box_size + shift_x)
        
        fz_start = max(0, -shift_z)
        fz_end = fz_start + (z_end - z_start)
        fy_start = max(0, -shift_y)
        fy_end = fy_start + (y_end - y_start)
        fx_start = max(0, -shift_x)
        fx_end = fx_start + (x_end - x_start)
        
        target_view = combined_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        source_view = fiber_mask[fz_start:fz_end, fy_start:fy_end, fx_start:fx_end]
        
        if physics_mode == 'thermal':
            # Thermal mode: Overlap between existing filler (>=2) and new fiber is 2 (Contact/Penalty)
            target_view[(target_view >= 2) & source_view] = inter_id  
            # For the new fiber, empty parts (==0) become filler
            target_view[(target_view == 0) & source_view] = filler_id 
        else:
            # Electrical/Mechanics mode: Maintain all as filler even upon contact or intersection
            target_view[source_view] = filler_id
            
    return crop_mask_to_bbox(combined_mask)

def get_flexible_fiber_mask(length=90, radius=2, max_bend_deg=90, max_total_bends=10, physics_mode='thermal'):
    return crop_mask_to_bbox(create_fiber_mask(length, radius, max_bend_deg, max_total_bends))

def get_agglomerate_mask(num_fibers=5, length=90, radius=2, max_bend_deg=90, max_total_bends=10, physics_mode='thermal'):
    return create_agglomerate_mask(num_fibers, length, radius, max_bend_deg, max_total_bends, physics_mode)

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
# B. Rigid Stamp Module (Sphere, Flake, Rigid Cylinder)
# =========================================================

def create_rotated_grid(shape, angles):
    """Helper to create a 3D grid and rotate it by specified Euler angles"""
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
    return rotated_coords[2,:].reshape(shape), rotated_coords[1,:].reshape(shape), rotated_coords[0,:].reshape(shape)

def get_sphere_mask(radius, physics_mode='thermal'):
    """Perfect sphere filler"""
    size = int(radius * 2 + 2)
    z, y, x = np.indices((size, size, size))
    cz, cy, cx = size//2, size//2, size//2
    return (z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= radius**2

def get_flake_mask(radius, thickness, physics_mode='thermal'):
    """Flake-shaped filler"""
    size = int(radius * 2 + 4)
    Xr, Yr, Zr = create_rotated_grid((size, size, size), rng.random(3) * 2 * np.pi)
    return (Xr**2 + Yr**2) / radius**2 + (Zr**2) / (thickness/2)**2 <= 1.0

def get_rigid_cylinder_mask(length, radius, physics_mode='thermal'):
    """Rigid short fiber"""
    size = int(length + radius * 2 + 4)
    Xr, Yr, Zr = create_rotated_grid((size, size, size), rng.random(3) * 2 * np.pi)
    return (Xr**2 + Yr**2 <= radius**2) & (np.abs(Zr) <= length/2)

def calculate_protrusion_limit(filler_voxels, total_voxels, half_protrusion_vol_ratio=0.0025):
    """Calculation of adaptive protrusion tolerance based on half-value volume ratio model"""
    if half_protrusion_vol_ratio <= 0:
        return 0.0
    x = filler_voxels / total_voxels
    C = half_protrusion_vol_ratio
    return ((1 + C) * x) / (x + C)

# =========================================================
# C. Topology-Adaptive Growth Module (Straight penetration + 180-deg U-turn & void generation integrated version)
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
    
    # Application of asymptotic model
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
                        
                        # Calculate side step distance
                        # Fiber radius (R) + Void (2R) + Radius after turn (R) = Center-to-center distance of 4 * R
                        shift_dist = max(2, int(radius * 4))
                        
                        sidestep_success = True
                        temp_pos = current_pos.copy()
                        temp_path = []
                        
                        # Crab-walk to the side
                        for _ in range(shift_dist):
                            temp_pos += u_ortho
                            temp_i = np.round(temp_pos).astype(int)
                            tz, ty, tx = temp_i[0] % shape[0], temp_i[1] % shape[1], temp_i[2] % shape[2]
                            
                            # Check for collisions with other fillers or walls during side step
                            # (Self-intersections are not yet reflected in comp_grid, so gaps are maintained as natural "transparent walls")
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
    
def apply_brush_and_write(comp_grid, backbone, radius, physics_mode='thermal', shell_count_grid=None, filler_id=3, inter_id=2):
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
        # Thermal conduction mode: Areas overlapping with existing fillers are set to penalty (2)
        contact_mask = (comp_grid[gz, gy, gx] >= 2)
        comp_grid[gz[contact_mask], gy[contact_mask], gx[contact_mask]] = inter_id
        body_mask = ~contact_mask
        comp_grid[gz[body_mask], gy[body_mask], gx[body_mask]] = filler_id
    else:
        # Electrical conduction mode: The main body is unconditionally maintained as a good conductor
        comp_grid[gz, gy, gx] = filler_id
        
        # Count the shell dilated by 2 voxels
        if shell_count_grid is not None:
            size_sh = int((radius + 2) * 2 + 2)
            z_sh, y_sh, x_sh = np.indices((size_sh, size_sh, size_sh))
            cz_sh, cy_sh, cx_sh = size_sh//2, size_sh//2, size_sh//2
            shell_brush_mask = (z_sh - cz_sh)**2 + (y_sh - cy_sh)**2 + (x_sh - cx_sh)**2 <= (radius + 2)**2
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
                          physics_mode='thermal', shell_count_grid=None, filler_id=3, inter_id=2):
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
# D. RSA Placement Logic (Ultra-fast index version + Electrical/Thermal mode support)
# =========================================================

def place_fillers_hybrid(comp_grid, tpms_grid, filler_func, kwargs, target_vol_frac,
                         max_attempts=1000000, fallback_func=None, desc="",
                         protrusion_coef=0.0025, log_file=None,
                         physics_mode='thermal', shell_count_grid=None, filler_id=3, inter_id=2):
    
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

    with tqdm(total=target_voxels, desc=desc, unit="voxel") as pbar:
        while placed_voxels < initial_placed + target_voxels and attempts < max_attempts:
            attempts += 1
            
            if not overlap_mode and consecutive_fails > 500:
                overlap_mode = True

            # Update cache
            if stamp_offsets is None or cache_reuse_count >= MAX_CACHE_REUSE:
                raw_stamp = filler_func(**kwargs)
                
                # For placement judgment: Extract only the occupied coordinates of space (determined by > 0 regardless of type)
                coords = np.argwhere(raw_stamp > 0)
                if len(coords) == 0:
                    stamp_offsets = None
                    continue
                
                center = np.array(raw_stamp.shape) // 2
                stamp_offsets = coords - center
                
                # Extract values for writing (if bool, assign 2 in batch; if uint8, extract that value)
                if raw_stamp.dtype == bool:
                    stamp_vals = np.full(len(coords), filler_id, dtype=np.uint8)
                else:
                    stamp_vals = raw_stamp[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.uint8)

                # For electrical mode, extract shell coordinates dilated by 2 voxels
                if physics_mode == 'electrical':
                    rz, ry, rx = np.ogrid[-2:3, -2:3, -2:3]
                    shell_brush = rx**2 + ry**2 + rz**2 <= 2**2
                    dilated = binary_dilation(raw_stamp > 0, structure=shell_brush)
                    shell_coords = np.argwhere(dilated)
                    shell_offsets = shell_coords - center
                else:
                    shell_offsets = None

                filler_voxels = len(coords)
                current_protrusion_limit = calculate_protrusion_limit(filler_voxels, total_voxels, protrusion_coef)
                cache_reuse_count = 0

            idx = rng.integers(0, num_valid_coords)
            cz, cy, cx = valid_z[idx], valid_y[idx], valid_x[idx]

            # Calculate global coordinates for judgment
            target_coords = stamp_offsets + np.array([cz, cy, cx])
            tz, ty, tx = target_coords[:, 0], target_coords[:, 1], target_coords[:, 2]
            tz, ty, tx = tz % shape[0], ty % shape[1], tx % shape[2]

            in_phase_b = (tpms_grid[tz, ty, tx] == 1)
            protrusion_count = np.sum(in_phase_b)
            
            success = False
            if protrusion_count <= len(target_coords) * current_protrusion_limit:
                # Collision check with existing fillers
                overlap_mask = comp_grid[tz, ty, tx] >= 2
                
                if overlap_mode or not np.any(overlap_mask):
                    current_vals = comp_grid[tz, ty, tx]
                    new_vals = stamp_vals.copy()
                    
                    if physics_mode == 'thermal':
                        # Areas overlapping with already placed fillers (>=2) are converted to contact resistance phase (2)
                        contact_idx = (current_vals >= 2)
                        new_vals[contact_idx] = inter_id
                        comp_grid[tz, ty, tx] = new_vals
                    else:
                        # Electrical mode: Main body is overwritten entirely as a good conductor
                        comp_grid[tz, ty, tx] = np.where(new_vals > 0, filler_id, current_vals)
                        # Update shell counter
                        if shell_count_grid is not None and shell_offsets is not None:
                            stz = (shell_offsets[:, 0] + cz) % shape[0]
                            sty = (shell_offsets[:, 1] + cy) % shape[1]
                            stx = (shell_offsets[:, 2] + cx) % shape[2]
                            shell_count_grid[stz, sty, stx] += 1

                    success = True

            if not success:
                cache_reuse_count += 1
                consecutive_fails += 1
                continue

            # Reset cache only on success
            stamp_offsets = None 
            consecutive_fails = 0
            
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
# E. Final Structure Integration, Output, and Aggregation
# =========================================================

def finalize_microstructure(comp_grid, tpms_grid, shell_count_grid=None, physics_mode='thermal', inter_id=2):
    """Combine background and filler phases, and extract the shell phase for electrical/mechanics modes"""
    final_grid = np.where(comp_grid > 0, comp_grid, tpms_grid).astype(np.uint8)
    
    if physics_mode in ['electrical', 'mechanics'] and shell_count_grid is not None:
        # Locations where IDs are less than 3 (i.e., polymers 0, 1 and extracted intermediate phase 2) and shells overlap become tunnel phase
        tunnel_mask = (shell_count_grid >= 2) & (final_grid < 3)
        final_grid[tunnel_mask] = inter_id
    return final_grid

def summarize_phase_fractions(final_grid, inter_id=2):
    """Aggregate the volume fraction of each phase"""
    total_voxels = final_grid.size
    max_id = final_grid.max() # Get the maximum ID present in the grid
    counts = np.bincount(final_grid.ravel(), minlength=max(4, max_id + 1))
    
    stats = {
        'polymer_a_fraction': counts[0] / total_voxels,
        'polymer_b_fraction': counts[1] / total_voxels,
        'interface_fraction': counts[inter_id] / total_voxels,
    }
    
    filler_total = 0
    # Filler IDs are from 3 to maximum value
    for i in range(3, max_id + 1):
        stats[f'filler_id{i}_fraction'] = counts[i] / total_voxels
        filler_total += counts[i] / total_voxels
        
    stats['filler_total_fraction'] = filler_total
    stats['polymer_fraction'] = (counts[0] + counts[1]) / total_voxels
    return stats

def export_visualization_vti(final_grid, filename="microstructure.vti", voxel_size=1e-8, metadata=None):
    """
    Output the 3D grid in the optimal VTI format for ParaView.
    Store as Point Data to enable volume rendering similar to TIFF stacks.
    Embed physical dimensions (voxel_size) and metadata for HUD (metadata) in Field Data.
    """
    import pyvista as pv
    grid = pv.ImageData()
    
    nz, ny, nx = final_grid.shape
    # Define dimensions as points (vertices) instead of cells (boxes) (same behavior as TIFF)
    grid.dimensions = (nx, ny, nz)
    
    # Set physical scale
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.origin = (0.0, 0.0, 0.0)
    
    # Store in point_data instead of cell_data
    grid.point_data["Phase"] = final_grid.flatten(order="C")
    
    # Embed metadata (for HUD)
    if metadata:
        for key, value in metadata.items():
            grid.field_data[key] = [value]
            
    grid.save(filename)
    return grid

def export_chfem_inputs(final_grid, base_filename="model", voxel_size=1e-8, physics_mode='thermal', prop_map=None):
    raw_filename = f"{base_filename}.raw"
    final_grid.tofile(raw_filename)
    nf_filename = f"{base_filename}.nf"
    shape = final_grid.shape

    analysis_type = 1 if physics_mode == 'mechanics' else 0
    num_materials = len(prop_map)

    """
    Example default settings
    if physics_mode == 'thermal':
        sigma_polymer = 0.30     # Phase A & B (Low thermal conductivity resin)
        sigma_filler = 300.0     # High thermal conductivity filler
        sigma_contact = 30.0     # Contact resistance phase (Penalty)
        analysis_type = 0
    elif physics_mode == 'electrical':
        sigma_polymer = 1e-6    # Phase A & B (Insulator, minute value for convergence)
        sigma_filler = 1e4      # Conductive filler
        sigma_contact = 1e-2    # Tunnel phase (Bonus)
        analysis_type = 0
    elif physics_mode == 'mechanics':
        sigma_polymer = "3.0 1.0"    # Phase A & B (Resin)
        sigma_filler = "100.0 50.0"  # Reinforcing filler
        sigma_contact = "15.0 5.0"   # Constrained phase (Bonus)
        analysis_type = 1
    """
    
    # Write out physical properties in dictionary ID order (0, 1, 2...)
    props_str = "\n".join([f"{k} {v}" for k, v in sorted(prop_map.items())])

    nf_content = f"""%type_of_analysis {analysis_type}
%voxel_size {voxel_size}
%solver_tolerance 1e-08
%number_of_iterations 10000
%image_dimensions {shape[2]} {shape[1]} {shape[0]}
%refinement 1
%number_of_materials {num_materials}
%properties_of_materials
{props_str}
%data_type float64
"""
    with open(nf_filename, 'w') as f:
        f.write(nf_content)import math
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm.auto import tqdm
import pyvista as pv

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

def create_fiber_mask(length, radius, max_bend_deg=90, max_total_bends=10):
    """Generate a mask for a single flexible fiber"""
    # A space large enough to definitely contain the fiber
    box_size = int(length * 2 + radius * 2 + 5)
    mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    
    start_pos = (box_size//2, box_size//2, box_size//2)
    backbone = [start_pos]
    current_pos = np.array(start_pos, dtype=float)
    
    vec = rng.standard_normal(3)
    vec /= np.linalg.norm(vec)
    bends_made = 0

    for _ in range(int(length)):
        next_pos_f = current_pos + vec
        next_pos_i = np.round(next_pos_f).astype(int)
        
        # Safety check for box boundaries
        if any(p < radius or p >= box_size - radius for p in next_pos_i):
            break

        # Naturally bend with a certain probability according to the specified total number of bends and length
        if bends_made < max_total_bends and rng.random() < (max_total_bends / length):
            angle_rad = np.radians(rng.uniform(10, max_bend_deg))
            noise = rng.standard_normal(3)
            noise -= noise.dot(vec) * vec
            if np.linalg.norm(noise) > 0:
                noise /= np.linalg.norm(noise)
                new_vec = vec * np.cos(angle_rad) + noise * np.sin(angle_rad)
                new_vec /= np.linalg.norm(new_vec)
                vec = new_vec
                bends_made += 1
        
        current_pos = current_pos + vec
        backbone.append(np.round(current_pos).astype(int))

    # Flesh out with a brush
    # ==========================================
    # Dramatic speed-up point (Batch fleshing out by Dilation)
    # ==========================================
    # 1. Set only the backbone (core) pixels to True
    for (bz, by, bx) in backbone:
        if 0 <= bz < box_size and 0 <= by < box_size and 0 <= bx < box_size:
            mask[bz, by, bx] = True
            
    # 2. Create a spherical structuring element (brush)
    rz, ry, rx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    brush = rx**2 + ry**2 + rz**2 <= radius**2
    
    # 3. Flesh out all at once using dilation processing
    mask = binary_dilation(mask, structure=brush)

    return mask

def _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends):
    """Single fiber mask for agglomerates that grows to both sides from the center and random walks midway"""
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
        # Adjust bending probability according to remaining steps
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
    # Combine both sides across the start position
    backbone = backbone_b[::-1] + [tuple(start_pos)] + backbone_a
    
    for (z, y, x) in backbone:
        mask[z, y, x] = True
        
    mask = binary_dilation(mask, structure=brush)
    return mask

def create_agglomerate_mask(num_fibers, length, radius, max_bend_deg=90, max_total_bends=10, physics_mode='thermal', filler_id=3, inter_id=2):
    """Generate an agglomerate mask of multiple entangled fibers, crop, and return"""
    # The fiber grows a max of length/2 from the center, so box size is scaled down based on length
    box_size = int(length + radius * 2 + 5)
    combined_mask = np.zeros((box_size, box_size, box_size), dtype=np.uint8)
    
    center = box_size // 2
    start_radius = radius * 2
    
    for _ in range(num_fibers):
        # Generate bilateral growth fibers specialized for agglomerates
        fiber_mask = _create_agglom_single_fiber_mask(length, radius, max_bend_deg, max_total_bends)
        
        # Shift the center position slightly and composite
        offset = np.round(rng.standard_normal(3) * start_radius).astype(int)
        
        # Slice combination processing exactly identical to the existing code
        shift_z, shift_y, shift_x = offset
        
        z_start = max(0, shift_z)
        z_end = min(box_size, box_size + shift_z)
        y_start = max(0, shift_y)
        y_end = min(box_size, box_size + shift_y)
        x_start = max(0, shift_x)
        x_end = min(box_size, box_size + shift_x)
        
        fz_start = max(0, -shift_z)
        fz_end = fz_start + (z_end - z_start)
        fy_start = max(0, -shift_y)
        fy_end = fy_start + (y_end - y_start)
        fx_start = max(0, -shift_x)
        fx_end = fx_start + (x_end - x_start)
        
        target_view = combined_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        source_view = fiber_mask[fz_start:fz_end, fy_start:fy_end, fx_start:fx_end]
        
        if physics_mode == 'thermal':
            # Thermal mode: Overlap between existing filler (>=2) and new fiber is 2 (Contact/Penalty)
            target_view[(target_view >= 2) & source_view] = inter_id  
            # For the new fiber, empty parts (==0) become filler
            target_view[(target_view == 0) & source_view] = filler_id 
        else:
            # Electrical/Mechanics mode: Maintain all as filler even upon contact or intersection
            target_view[source_view] = filler_id
            
    return crop_mask_to_bbox(combined_mask)

def get_flexible_fiber_mask(length=90, radius=2, max_bend_deg=90, max_total_bends=10, physics_mode='thermal'):
    return crop_mask_to_bbox(create_fiber_mask(length, radius, max_bend_deg, max_total_bends))

def get_agglomerate_mask(num_fibers=5, length=90, radius=2, max_bend_deg=90, max_total_bends=10, physics_mode='thermal'):
    return create_agglomerate_mask(num_fibers, length, radius, max_bend_deg, max_total_bends, physics_mode)

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
# B. Rigid Stamp Module (Sphere, Flake, Rigid Cylinder)
# =========================================================

def create_rotated_grid(shape, angles):
    """Helper to create a 3D grid and rotate it by specified Euler angles"""
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
    return rotated_coords[2,:].reshape(shape), rotated_coords[1,:].reshape(shape), rotated_coords[0,:].reshape(shape)

def get_sphere_mask(radius, physics_mode='thermal'):
    """Perfect sphere filler"""
    size = int(radius * 2 + 2)
    z, y, x = np.indices((size, size, size))
    cz, cy, cx = size//2, size//2, size//2
    return (z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= radius**2

def get_flake_mask(radius, thickness, physics_mode='thermal'):
    """Flake-shaped filler"""
    size = int(radius * 2 + 4)
    Xr, Yr, Zr = create_rotated_grid((size, size, size), rng.random(3) * 2 * np.pi)
    return (Xr**2 + Yr**2) / radius**2 + (Zr**2) / (thickness/2)**2 <= 1.0

def get_rigid_cylinder_mask(length, radius, physics_mode='thermal'):
    """Rigid short fiber"""
    size = int(length + radius * 2 + 4)
    Xr, Yr, Zr = create_rotated_grid((size, size, size), rng.random(3) * 2 * np.pi)
    return (Xr**2 + Yr**2 <= radius**2) & (np.abs(Zr) <= length/2)

def calculate_protrusion_limit(filler_voxels, total_voxels, half_protrusion_vol_ratio=0.0025):
    """Calculation of adaptive protrusion tolerance based on half-value volume ratio model"""
    if half_protrusion_vol_ratio <= 0:
        return 0.0
    x = filler_voxels / total_voxels
    C = half_protrusion_vol_ratio
    return ((1 + C) * x) / (x + C)

# =========================================================
# C. Topology-Adaptive Growth Module (Straight penetration + 180-deg U-turn & void generation integrated version)
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
    
    # Application of asymptotic model
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
                        
                        # Calculate side step distance
                        # Fiber radius (R) + Void (2R) + Radius after turn (R) = Center-to-center distance of 4 * R
                        shift_dist = max(2, int(radius * 4))
                        
                        sidestep_success = True
                        temp_pos = current_pos.copy()
                        temp_path = []
                        
                        # Crab-walk to the side
                        for _ in range(shift_dist):
                            temp_pos += u_ortho
                            temp_i = np.round(temp_pos).astype(int)
                            tz, ty, tx = temp_i[0] % shape[0], temp_i[1] % shape[1], temp_i[2] % shape[2]
                            
                            # Check for collisions with other fillers or walls during side step
                            # (Self-intersections are not yet reflected in comp_grid, so gaps are maintained as natural "transparent walls")
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
    
def apply_brush_and_write(comp_grid, backbone, radius, physics_mode='thermal', shell_count_grid=None, filler_id=3, inter_id=2):
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
        # Thermal conduction mode: Areas overlapping with existing fillers are set to penalty (2)
        contact_mask = (comp_grid[gz, gy, gx] >= 2)
        comp_grid[gz[contact_mask], gy[contact_mask], gx[contact_mask]] = inter_id
        body_mask = ~contact_mask
        comp_grid[gz[body_mask], gy[body_mask], gx[body_mask]] = filler_id
    else:
        # Electrical conduction mode: The main body is unconditionally maintained as a good conductor
        comp_grid[gz, gy, gx] = filler_id
        
        # Count the shell dilated by 2 voxels
        if shell_count_grid is not None:
            size_sh = int((radius + 2) * 2 + 2)
            z_sh, y_sh, x_sh = np.indices((size_sh, size_sh, size_sh))
            cz_sh, cy_sh, cx_sh = size_sh//2, size_sh//2, size_sh//2
            shell_brush_mask = (z_sh - cz_sh)**2 + (y_sh - cy_sh)**2 + (x_sh - cx_sh)**2 <= (radius + 2)**2
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
                          physics_mode='thermal', shell_count_grid=None, filler_id=3, inter_id=2):
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
# D. RSA Placement Logic (Ultra-fast index version + Electrical/Thermal mode support)
# =========================================================

def place_fillers_hybrid(comp_grid, tpms_grid, filler_func, kwargs, target_vol_frac,
                         max_attempts=1000000, fallback_func=None, desc="",
                         protrusion_coef=0.0025, log_file=None,
                         physics_mode='thermal', shell_count_grid=None, filler_id=3, inter_id=2):
    
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

    with tqdm(total=target_voxels, desc=desc, unit="voxel") as pbar:
        while placed_voxels < initial_placed + target_voxels and attempts < max_attempts:
            attempts += 1
            
            if not overlap_mode and consecutive_fails > 500:
                overlap_mode = True

            # Update cache
            if stamp_offsets is None or cache_reuse_count >= MAX_CACHE_REUSE:
                raw_stamp = filler_func(**kwargs)
                
                # For placement judgment: Extract only the occupied coordinates of space (determined by > 0 regardless of type)
                coords = np.argwhere(raw_stamp > 0)
                if len(coords) == 0:
                    stamp_offsets = None
                    continue
                
                center = np.array(raw_stamp.shape) // 2
                stamp_offsets = coords - center
                
                # Extract values for writing (if bool, assign 2 in batch; if uint8, extract that value)
                if raw_stamp.dtype == bool:
                    stamp_vals = np.full(len(coords), filler_id, dtype=np.uint8)
                else:
                    stamp_vals = raw_stamp[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.uint8)

                # For electrical mode, extract shell coordinates dilated by 2 voxels
                if physics_mode == 'electrical':
                    rz, ry, rx = np.ogrid[-2:3, -2:3, -2:3]
                    shell_brush = rx**2 + ry**2 + rz**2 <= 2**2
                    dilated = binary_dilation(raw_stamp > 0, structure=shell_brush)
                    shell_coords = np.argwhere(dilated)
                    shell_offsets = shell_coords - center
                else:
                    shell_offsets = None

                filler_voxels = len(coords)
                current_protrusion_limit = calculate_protrusion_limit(filler_voxels, total_voxels, protrusion_coef)
                cache_reuse_count = 0

            idx = rng.integers(0, num_valid_coords)
            cz, cy, cx = valid_z[idx], valid_y[idx], valid_x[idx]

            # Calculate global coordinates for judgment
            target_coords = stamp_offsets + np.array([cz, cy, cx])
            tz, ty, tx = target_coords[:, 0], target_coords[:, 1], target_coords[:, 2]
            tz, ty, tx = tz % shape[0], ty % shape[1], tx % shape[2]

            in_phase_b = (tpms_grid[tz, ty, tx] == 1)
            protrusion_count = np.sum(in_phase_b)
            
            success = False
            if protrusion_count <= len(target_coords) * current_protrusion_limit:
                # Collision check with existing fillers
                overlap_mask = comp_grid[tz, ty, tx] >= 2
                
                if overlap_mode or not np.any(overlap_mask):
                    current_vals = comp_grid[tz, ty, tx]
                    new_vals = stamp_vals.copy()
                    
                    if physics_mode == 'thermal':
                        # Areas overlapping with already placed fillers (>=2) are converted to contact resistance phase (2)
                        contact_idx = (current_vals >= 2)
                        new_vals[contact_idx] = inter_id
                        comp_grid[tz, ty, tx] = new_vals
                    else:
                        # Electrical mode: Main body is overwritten entirely as a good conductor
                        comp_grid[tz, ty, tx] = np.where(new_vals > 0, filler_id, current_vals)
                        # Update shell counter
                        if shell_count_grid is not None and shell_offsets is not None:
                            stz = (shell_offsets[:, 0] + cz) % shape[0]
                            sty = (shell_offsets[:, 1] + cy) % shape[1]
                            stx = (shell_offsets[:, 2] + cx) % shape[2]
                            shell_count_grid[stz, sty, stx] += 1

                    success = True

            if not success:
                cache_reuse_count += 1
                consecutive_fails += 1
                continue

            # Reset cache only on success
            stamp_offsets = None 
            consecutive_fails = 0
            
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
# E. Final Structure Integration, Output, and Aggregation
# =========================================================

def finalize_microstructure(comp_grid, tpms_grid, shell_count_grid=None, physics_mode='thermal', inter_id=2):
    """Combine background and filler phases, and extract the shell phase for electrical/mechanics modes"""
    final_grid = np.where(comp_grid > 0, comp_grid, tpms_grid).astype(np.uint8)
    
    if physics_mode in ['electrical', 'mechanics'] and shell_count_grid is not None:
        # Locations where IDs are less than 3 (i.e., polymers 0, 1 and extracted intermediate phase 2) and shells overlap become tunnel phase
        tunnel_mask = (shell_count_grid >= 2) & (final_grid < 3)
        final_grid[tunnel_mask] = inter_id
    return final_grid

def summarize_phase_fractions(final_grid, inter_id=2):
    """Aggregate the volume fraction of each phase"""
    total_voxels = final_grid.size
    max_id = final_grid.max() # Get the maximum ID present in the grid
    counts = np.bincount(final_grid.ravel(), minlength=max(4, max_id + 1))
    
    stats = {
        'polymer_a_fraction': counts[0] / total_voxels,
        'polymer_b_fraction': counts[1] / total_voxels,
        'interface_fraction': counts[inter_id] / total_voxels,
    }
    
    filler_total = 0
    # Filler IDs are from 3 to maximum value
    for i in range(3, max_id + 1):
        stats[f'filler_id{i}_fraction'] = counts[i] / total_voxels
        filler_total += counts[i] / total_voxels
        
    stats['filler_total_fraction'] = filler_total
    stats['polymer_fraction'] = (counts[0] + counts[1]) / total_voxels
    return stats

def export_visualization_vti(final_grid, filename="microstructure.vti", voxel_size=1e-8, metadata=None):
    """
    Output the 3D grid in the optimal VTI format for ParaView.
    Store as Point Data to enable volume rendering similar to TIFF stacks.
    Embed physical dimensions (voxel_size) and metadata for HUD (metadata) in Field Data.
    """
    import pyvista as pv
    grid = pv.ImageData()
    
    nz, ny, nx = final_grid.shape
    # Define dimensions as points (vertices) instead of cells (boxes) (same behavior as TIFF)
    grid.dimensions = (nx, ny, nz)
    
    # Set physical scale
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.origin = (0.0, 0.0, 0.0)
    
    # Store in point_data instead of cell_data
    grid.point_data["Phase"] = final_grid.flatten(order="C")
    
    # Embed metadata (for HUD)
    if metadata:
        for key, value in metadata.items():
            grid.field_data[key] = [value]
            
    grid.save(filename)
    return grid

def export_chfem_inputs(final_grid, base_filename="model", voxel_size=1e-8, physics_mode='thermal', prop_map=None):
    raw_filename = f"{base_filename}.raw"
    final_grid.tofile(raw_filename)
    nf_filename = f"{base_filename}.nf"
    shape = final_grid.shape

    analysis_type = 1 if physics_mode == 'mechanics' else 0
    num_materials = len(prop_map)

    """
    Example default settings
    if physics_mode == 'thermal':
        sigma_polymer = 0.30     # Phase A & B (Low thermal conductivity resin)
        sigma_filler = 300.0     # High thermal conductivity filler
        sigma_contact = 30.0     # Contact resistance phase (Penalty)
        analysis_type = 0
    elif physics_mode == 'electrical':
        sigma_polymer = 1e-4    # Phase A & B (Insulator, minute value for convergence)
        sigma_filler = 1e4      # Conductive filler
        sigma_contact = 1e0     # Tunnel phase (Bonus)
        analysis_type = 0
    elif physics_mode == 'mechanics':
        sigma_polymer = "3.0 1.0"    # Phase A & B (Resin)
        sigma_filler = "100.0 50.0"  # Reinforcing filler
        sigma_contact = "15.0 5.0"   # Constrained phase (Bonus)
        analysis_type = 1
    """
    
    # Write out physical properties in dictionary ID order (0, 1, 2...)
    props_str = "\n".join([f"{k} {v}" for k, v in sorted(prop_map.items())])

    nf_content = f"""%type_of_analysis {analysis_type}
%voxel_size {voxel_size}
%solver_tolerance 1e-08
%number_of_iterations 10000
%image_dimensions {shape[2]} {shape[1]} {shape[0]}
%refinement 1
%number_of_materials {num_materials}
%properties_of_materials
{props_str}
%data_type float64
"""
    with open(nf_filename, 'w') as f:
        f.write(nf_content)
