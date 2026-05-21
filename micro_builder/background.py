import numpy as np

from . import builder

# =========================================================
# A. Background Phase (Polymer Matrix) Generation Module
# =========================================================

def probabilistic_binarize_fast(field, target_phaseA_ratio, diffusion_factor=0.0):
    """
    Fast probabilistic binarization module that permits isolated voxels (islands)
    and enforces the use of the common random generator (builder.rng).
    This introduces a mutual diffusion layer (interphase) between Phase A and B.
    """
    if diffusion_factor <= 1e-5:
        # No diffusion: Sharp split using exact percentile (Legacy mode)
        threshold = np.percentile(field, target_phaseA_ratio * 100)
        grid = np.where(field > threshold, 1, 0).astype(np.uint8)
        return grid, threshold, target_phaseA_ratio
        
    # --- Mutual Diffusion (Stochastic Binarization) Process ---
    
    # 1. Normalize field (Shift percentile to 0)
    base_threshold = np.percentile(field, target_phaseA_ratio * 100)
    shifted = field - base_threshold
    
    # 2. Calculate probability field (Sigmoid function)
    # Higher diffusion_factor yields a gentler slope, widening the diffusion zone
    beta = 1.0 / diffusion_factor
    prob_B = 1.0 / (1.0 + np.exp(-np.clip(beta * shifted, -50, 50)))
    
    # 3. Sampling using the shared random table [CRITICAL]
    # Always use builder.rng to protect the continuity of the seed for RSA
    random_field = builder.rng.uniform(0.0, 1.0, size=field.shape)
    
    # 4. Binarization
    grid = np.where(random_field < prob_B, 1, 0).astype(np.uint8)
    
    # 5. Post-evaluation
    # Since this is independent probability sampling, the expected value asymptotically
    # approaches the target due to the Law of Large Numbers, but minor statistical
    # fluctuations occur, so we return the actual measured volume fraction.
    actual_phaseA_ratio = np.mean(grid == 0)
    
    return grid, base_threshold, actual_phaseA_ratio


def build_single_phase_grid(grid_size, feature_size=10, diffusion_factor=0.0):
    """Single polymer phase (all Phase A: 0)"""
    tpms_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    return None, tpms_grid, 0.0, 1.0

def build_tpms_grid_with_target_ratio(grid_size, feature_size=10, target_phaseA_ratio=0.5, diffusion_factor=0.0):
    """Co-continuous Gyroid phase"""
    # Binarize the gyroid to match the specified Phase A volume ratio
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    gyroid = (np.sin(x / feature_size) * np.cos(y / feature_size) + 
              np.sin(y / feature_size) * np.cos(z / feature_size) + 
              np.sin(z / feature_size) * np.cos(x / feature_size))
              
    tpms_grid, threshold, actual_phaseA_ratio = probabilistic_binarize_fast(
        gyroid, target_phaseA_ratio, diffusion_factor
    )
    
    return gyroid, tpms_grid, threshold, actual_phaseA_ratio

def build_lamellar_grid(grid_size, feature_size=10, target_phaseA_ratio=0.5, diffusion_factor=0.0):
    """Lamellar structure (spread in XY plane, stacked in Z direction)"""
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    
    # 1D periodic function in the Z direction
    field = np.cos(2 * np.pi * z / feature_size)
    
    grid, threshold, actual_phaseA_ratio = probabilistic_binarize_fast(
        field, target_phaseA_ratio, diffusion_factor
    )
    
    return field, grid, threshold, actual_phaseA_ratio

def build_cylinder_hex_grid(grid_size, feature_size=15, target_phaseA_ratio=0.7, diffusion_factor=0.0):
    """Cylinder structure (upright in Z-axis direction, hexagonal array in XY plane)"""
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    q = 2 * np.pi / feature_size
    
    # Approximation of 2D hexagonal lattice (superposition of waves in 3 directions)
    field = (np.cos(q * x) + 
             np.cos(q * (x + np.sqrt(3) * y) / 2.0) + 
             np.cos(q * (x - np.sqrt(3) * y) / 2.0))
    
    grid, threshold, actual_phaseA_ratio = probabilistic_binarize_fast(
        field, target_phaseA_ratio, diffusion_factor
    )
    
    return field, grid, threshold, actual_phaseA_ratio

def build_bcc_grid(grid_size, feature_size=15, target_phaseA_ratio=0.7, diffusion_factor=0.0):
    """Body-centered cubic structure (BCC: 3D regular array of spherical domains)"""
    z, y, x = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    q = 2 * np.pi / feature_size
    
    # Approximation of BCC lattice
    field = (np.cos(q * x) * np.cos(q * y) + 
             np.cos(q * y) * np.cos(q * z) + 
             np.cos(q * z) * np.cos(q * x))
    
    grid, threshold, actual_phaseA_ratio = probabilistic_binarize_fast(
        field, target_phaseA_ratio, diffusion_factor
    )
    
    return field, grid, threshold, actual_phaseA_ratio

def build_sea_island_grid(grid_size, feature_size=8, target_phaseA_ratio=0.7, diffusion_factor=0.0):
    """Sea-island structure (Random sphere placement by Boolean model: Island is Phase B)"""
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    target_voxels = int((grid_size**3) * (1 - target_phaseA_ratio))
    placed_voxels = 0
    
    # Create island mask
    size = int(feature_size * 2 + 2)
    z, y, x = np.indices((size, size, size))
    cz, cy, cx = size//2, size//2, size//2
    sphere = (z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= feature_size**2
    coords = np.argwhere(sphere)
    center = np.array(sphere.shape) // 2
    offsets = coords - center
    
    attempts = 0
    # Random placement allowing overlap
    while placed_voxels < target_voxels and attempts < 100000:
        attempts += 1
        cz, cy, cx = builder.rng.integers(0, grid_size, size=3)
        t_coords = offsets + np.array([cz, cy, cx])
        tz, ty, tx = t_coords[:, 0] % grid_size, t_coords[:, 1] % grid_size, t_coords[:, 2] % grid_size
        
        grid[tz, ty, tx] = 1
        placed_voxels = np.sum(grid == 1)
        
    actual_phaseA_ratio = 1.0 - (placed_voxels / (grid_size**3))
    return None, grid, 0.0, actual_phaseA_ratio

def build_island_sea_grid(grid_size, feature_size=8, target_phaseA_ratio=0.7, diffusion_factor=0.0):
    """Island-sea structure (Random sphere placement by Boolean model: Island is Phase A)"""
    target_phaseB_ratio = 1.0 - target_phaseA_ratio
    _, grid, _, actual_phaseB_ratio = build_sea_island_grid(grid_size, feature_size, target_phaseB_ratio, diffusion_factor)
    grid ^= 1
    actual_phaseA_ratio = 1.0 - actual_phaseB_ratio
    return None, grid, 0.0, actual_phaseA_ratio
