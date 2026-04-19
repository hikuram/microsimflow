# RSA Filler Placement Algorithm Architecture

This document outlines the design and implementation of the Hybrid Random Sequential Adsorption (RSA) algorithm used in `microsimflow`.

To achieve high-density packing, physical accuracy, and seamless integration with downstream mechanical deformations, this algorithm abandons the traditional "paint-and-forget" voxel approach. Instead, it relies on a strict separation of **Kinematics (skeleton and rotation)** and **Voxel Rendering (rasterization)**, coupled with bitwise memory management.

-----

## 1\. Filler Geometry & Kinematic Standardization

Instead of drawing fillers directly into the massive global grid, the algorithm first generates a "stamp" in a localized bounding box. During this phase, the exact Center of Mass (CM) is mathematically standardized. *(Note: Topology-adaptive fibers bypass this module as they grow directly in the global grid).*

| **Shape** | **Step 1: Skeleton / Orientation** | **Step 2: Voxel Rendering** | **Step 3: Kinematics Definition** | **Step 4: CM Standardization** |
| :--- | :--- | :--- | :--- | :--- |
| **Sphere** | Isotropic geometry; no orientation calculation needed. | Fill voxels satisfying `x^2 + y^2 + z^2 <= r^2` in the local grid. | No rotation or deformation. Local coordinates are fixed at `[0,0,0]`. | The geometric center aligns perfectly with the true Center of Mass (CM). |
| **Flake** | Generate a rotation matrix `R_orig` from random Euler angles. | Apply inverse rotation to the coordinate grid and evaluate the disk equation (supports polar decomposition later). | Store `R_orig` to preserve rotational state. Local coordinate is `[0,0,0]`. | Geometric center is the CM. During affine stretch, polar decomposition extracts the pure rigid-body rotation to track orientation. |
| **Staggered** (Layered Flakes) | Use `R_orig` as the base. Determine local centers by shifting along the Z-axis (layer thickness) and applying random XY offsets. | On the inverse-rotated grid, draw multiple disks based on the distance from each layer's local center. | Store the list of local center coordinates for all layers. | Calculate the true CM from the multi-layer distribution and strictly correct the center offset. |
| **Rigid Fiber** | Generate a random directional vector. Create a straight backbone coordinate list by stepping forward until length *L* is reached. | Apply a fast "spherical brush" to each point on the backbone (bypassing slow dilation operations). | Store the straight backbone coordinate list. | Calculate the average of all backbone points to find the true CM and apply offset correction. |
| **Flexible Fiber** | Based on straight steps, but injects random noise (bend angles) at certain probabilities to deflect the direction vector. | Apply the fast spherical brush to the bent backbone. | Store the bent backbone coordinate list (used later to enforce inextensibility during affine stretch). | Calculate the true CM of the bent shape, then crop empty margins of the voxel mask to standardize the bounding box. |
| **Agglomerate** | Call the `Flexible Fiber` skeleton generator multiple times, shifting each by a random offset from a central point. | Generate masks for all individual fibers and overlay them in a single local grid. | Store a combined list of all fiber backbones. | Calculate the CM from all coordinates of all fibers in the bundle and correct the massive mask's offset. |

-----

## 2\. High-Performance Placement Logic (The RSA Core)

To prevent combinatorial explosion and computational bottlenecks during high-volume fraction packing, the placement loop utilizes advanced memory and caching optimizations.

| **Step** | **Process** | **Implementation & Optimization Strategy** |
| :--- | :--- | :--- |
| **1** | **Pre-extract Shells** | The dilation required to calculate tunneling zones (`tunnel_radius`) is performed exactly once on the small, local stamp *before* placement, extracting the shell coordinates as relative offsets. |
| **2** | **Geometry Caching** | If a placement attempt fails (e.g., due to a collision), the generated stamp mask and shell offsets are not discarded. They are cached and reused for up to 50 attempts, drastically reducing the cost of building complex shapes like flakes or agglomerates. |
| **3** | **Smart Sampling** | Before the loop begins, the algorithm lists all valid coordinates within the Polymer A phase. Random starting points (`cz, cy, cx`) are selected strictly from this list, eliminating wasted collision checks in invalid background phases. |
| **4** | **Protrusion & Overlap Check** | Utilizing **Numba JIT compilation**, the core collision engine (`_check_and_place_fast`) operates purely on C-level array indexing. It instantly evaluates boundary protrusion and filler overlaps without ever performing expensive matrix copies or Python slicing. |
| **5** | **Voxel Writing** | Upon clearing the checks, the algorithm writes the filler ID and shell data directly to the global `comp_grid` and `shell_count_grid`. |
| **6** | **Placement Registry** | Upon success, it is not just voxels that are saved. The exact kinematics (`geom_data`) and target coordinates are logged in the `placement_registry`. Subsequent processes (like mechanical stretching) read this registry to redraw the geometry, ensuring perfect rigid-body mechanics. |

-----

## 3\. Unified Placement & Physics-Aware Interface Generation

To maintain strict physical integrity, **the placement phase (Phase 1) is completely agnostic to the physics mode.** The algorithm guarantees that newly placed fillers *never* overwrite existing fillers, thereby protecting the target Volume Fraction (VF).

Instead of overwriting, the algorithm utilizes bitwise operations on an unsigned 8-bit integer array (`shell_count_grid`) to invisibly track overlaps and shell proximity. The physical meaning of these bits is interpreted later during the interface generation phase.

### Phase 1: Unified Placement Tracking (All Modes)

  * **Direct Overlap Flag:** If a newly placed filler overlaps an existing one, the highest bit is set (`|= 128`).
  * **Shell Proximity Count:** The expanded shell around the filler increments the lower 7 bits (`+= 1`).

### Phase 2: Physics-Specific Resolution

Once all fillers are placed, the bitwise data is decoded to construct physical interfaces tailored to the selected solver mode.

| **Process** | **Thermal Mode** (Heat Conduction) | **Electrical / Mechanics Mode** (Conductivity / Stiffness) |
| :--- | :--- | :--- |
| **Primary Interface**<br>*(Direct Contact)* | Extracts the `128` bit flag. Applies a Distance Transform to filter out massive bulk overlaps. Only thin contact zones (thickness \<= 1.5) are converted to Thermal Contact Resistance (Interface 1). Bulk overlaps are protected as filler material. | Extracts the lower 7 bits to find areas where shells overlap (`count >= 2`). Forms a temporary "Unified Interface." |
| **Secondary Interface**<br>*(Proximity/Tunneling)* | Extracts the lower 7 bits. In the polymer space, areas with `count >= 2` are identified as Kapitza bridge candidates (Interface 2). | (Determined during the separation step below). |
| **Targeted Cleanup & Separation** | **Isolated Execution:**<br>1. Protects the Primary interface (representing true physical touching).<br>2. Cleans up only the Secondary interface by removing 1-voxel islands and spikes to stabilize the FVM solver. | **Unified Execution & Split:**<br>1. Cleans up the temporary Unified Interface (removes slivers and spikes).<br>2. Dilates the filler bodies by 1 voxel.<br>3. Intersections between the dilation and the Unified Interface become the **Primary Interface**.<br>4. The remaining untouched Unified Interface becomes the **Secondary Interface** (Tunneling gap). |
