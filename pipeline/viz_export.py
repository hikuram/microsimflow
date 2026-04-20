import os

import matplotlib.pyplot as plt

def save_thumbnail_png(grid, filename):
    """
    Save the Z-axis center slice as a PNG.
    Capping values at 4 ensures that Filler 2 (ID 5) and beyond 
    are colored identically to Filler 1 (ID 4).
    """
    num_phases = 5
    vmin, vmax = -0.5, 4.5
    custom_cmap = plt.get_cmap('viridis', num_phases)

    z_mid = grid.shape[0] // 2
    # Clip all filler IDs (4, 5, 6...) to 4 for consistent visualization
    slice_img = np.clip(grid[z_mid, :, :], 0, 4)
    
    # Save the pure 2D array directly to a PNG file without any Matplotlib figure overhead
    plt.imsave(filename, slice_img, cmap=custom_cmap, vmin=vmin, vmax=vmax, origin='upper')
    print(f"Saved clean thumbnail: {filename}")

def export_vtm_wrapper(vti_filepath, vtm_filepath):
    """
    Creates a lightweight VTM (MultiBlock) wrapper that references the original VTI file.
    This safely bypasses ParaView's extent caching issue during volume rendering.
    """
    # Calculate relative path from 'pvd/' directory to the original VTI file (../file.vti)
    rel_vti_path = f"../{os.path.basename(vti_filepath)}"
    
    vtm_content = f"""<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">
  <vtkMultiBlockDataSet>
    <DataSet index="0" file="{rel_vti_path}"/>
  </vtkMultiBlockDataSet>
</VTKFile>
"""
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(vtm_filepath), exist_ok=True)
    
    with open(vtm_filepath, 'w', encoding='utf-8') as f:
        f.write(vtm_content)

def update_pvd_file(pvd_filepath, dataset_records):
    """
    Creates or updates a ParaView Data (.pvd) file to group VTM wrappers as a time-series.
    This ensures ParaView correctly handles varying grid extents during deformation.
    """
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(pvd_filepath), exist_ok=True)
    
    with open(pvd_filepath, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for timestep, filepath in dataset_records:
            # Extract basename for the 'file' attribute.
            # Since both PVD and VTM are in the 'pvd/' directory, only the filename is needed.
            rel_filename = os.path.basename(filepath)
            f.write(f'    <DataSet timestep="{timestep}" group="" part="0" file="{rel_filename}"/>\n')
            
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

def export_common_legend(filename="common_legend.png"):
    """
    Exports a standalone legend image with fixed 5 phases.
    Moving this outside the loop ensures consistency across all experiments.
    """
    if os.path.exists(filename):
        return
        
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
        'axes.labelsize': 12, 'ytick.labelsize': 12, 
        'savefig.bbox': 'tight'
    })

    # Standardized 5 phases
    ids = [0, 1, 2, 3, 4]
    labels = ['Polymer A', 'Polymer B', 'Secondary Inter', 'Primary Inter', 'Filler']
    num_phases = 5
    
    vmin, vmax = -0.5, 4.5
    custom_cmap = plt.get_cmap('viridis', num_phases)

    fig, ax = plt.subplots(figsize=(1.5, 4)) 
    ax.axis('off')
    
    # Create a dummy ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, ticks=ids, fraction=1.0, pad=0.0)
    cbar.ax.set_yticklabels(labels)
    cbar.ax.set_title("Phase ID", pad=15, fontweight='bold')

    plt.savefig(filename, dpi=200, transparent=False, facecolor='white')
    plt.close(fig)
    print(f"Saved common legend: {filename}")
