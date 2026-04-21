import numpy as np

from . import builder

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
        cz, cy, cx = builder.rng.integers(0, grid_size, size=3)
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
