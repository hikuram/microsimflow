# CLI Usage & Custom Pipelines

This document explains how to use the Microsimflow pipeline to design custom simulations, import real experimental images, and manage recalculations.

## 1. Running Custom Pipelines (Digital Generation)

You can easily design and run a custom simulation by passing arguments directly to `run_pipeline.py`. Use `python3 run_pipeline.py --help` to see all available parameters.

### Recipe Formatting
Recipe strings follow the format:
`type:volume_fraction:param1=value1:param2=value2:...`

### Supported Filler Types & Parameters

Below is a comprehensive reference of all supported filler types (`type`) and their specific parameters. 

> **Note:** If an optional parameter is not explicitly defined in your recipe string, the algorithm will use its default value.

| Type (`type`) | Description | Required Parameters | Optional Parameters (Default Values) |
| :--- | :--- | :--- | :--- |
| **`sphere`** | Perfect spherical filler. | `radius` | None |
| **`flake`** | Disk-like platelet. Supports orientation. | `radius`, `thickness` | `mean_dir` (0,0,1), `kappa` (0.0) |
| **`staggered`** | Multiple overlapping flakes mimicking exfoliated stacks. | None | `radius` (15), `layer_thickness` (2), `min_layers` (1), `max_layers` (4), `max_offset_pct` (30), `mean_dir` (0,0,1), `kappa` (0.0) |
| **`rigidfiber`** | Straight rigid cylinder. | `length`, `radius` | `mean_dir` (0,0,1), `kappa` (0.0) |
| **`flexfiber`** | Flexible fiber generated via a random walk with bending probability. | None | `length` (90), `radius` (2), `max_bend_deg` (90), `max_total_bends` (10) |
| **`irregfiber`** | Straight rigid fiber with an extruded irregular cross-section (e.g., C-shape, bean). | `length` | `shape` (ellipse)*, `radius` (5.0), `ratio` (0.5), `mean_dir` (0,0,1), `kappa` (0.0)<br>*`shape` accepts: `ellipse`, `bean`, `c-shape`* |
| **`agglom`** | A localized bundle/agglomerate of multiple flexible fibers. | None | `num_fibers` (5), `length` (90), `radius` (2), `max_bend_deg` (90), `max_total_bends` (10) |

### Orientation Control (Pseudo-Watson Distribution)
Orientation is controlled via `mean_dir` and `kappa` for `flake`, `staggered`, `rigidfiber`, and `irregfiber`.
* `kappa = 0`: Isotropic random orientation.
* `kappa > 0`: Polar alignment (vMF style) strictly along the `mean_dir` axis.
* `kappa < 0`: Girdle/Equatorial alignment where vectors distribute in the plane perpendicular to the `mean_dir` axis (Pseudo-Watson).

> **⚠️ Note on Coordinate System for `mean_dir`:**
> Due to the underlying NumPy grid indexing `(Z, Y, X)`, the axis assignment is inverted:
> * `mean_dir=0,0,1` aligns with the **X-axis**.
> * `mean_dir=0,1,0` aligns with the **Y-axis**.
> * `mean_dir=1,0,0` aligns with the **Z-axis**.

#### Example: Aligned Rigid Fibers (Polar, X-axis)
```bash
python3 run_pipeline.py \
  --size 200 --bg_type single --physics_mode thermal --solver skip \
  --recipe "rigidfiber:0.04:length=60:radius=2:mean_dir=0,0,1:kappa=20" \
  --basename aligned_rigidfiber --csv_log results.csv
```

#### Example: Equatorial Flakes (Avoiding Z-axis)
Forces flake normals into the XY-plane, meaning the flakes stand parallel to the Z-axis.
```bash
python3 run_pipeline.py \
  --size 200 --bg_type single --physics_mode thermal --solver skip \
  --recipe "flake:0.03:radius=14:thickness=2:mean_dir=1,0,0:kappa=-30" \
  --basename girdle_flake --csv_log results.csv
```

#### Example: Irregular Extruded Fibers (Bean-shape, Y-axis)
```bash
python3 run_pipeline.py \
  --size 200 --bg_type single --physics_mode thermal --solver skip \
  --recipe "irregfiber:0.04:length=70:radius=5:shape=bean:ratio=0.55:mean_dir=0,1,0:kappa=25" \
  --basename irregfiber_bean --csv_log results.csv
```

---

## 2. Real Image Import & Analysis

Microsimflow provides a decoupled workflow to ingest real experimental data (e.g., CT, FIB-SEM scans) and evaluate them using the exact same structure metrics and solvers as the digital twins. This is split into two independent steps to allow for trial-and-error during image thresholding.

### Step 2.1: Pre-processing & Interface Extraction (`import_image.py`)

This standalone utility converts raw 3D images into the unified 4-phase MicroSimFlow format (`.raw` and `.nf`), explicitly extracting Primary (Contact) and Secondary (Tunneling) interfaces.

```bash
python3 utils/import_image.py \
  --input data/experimental_ct.tif \
  --format tiff \
  --voxel_size 5e-7 \
  --threshold 128 \
  --pattern dilation \
  --tunnel_radius 2 \
  --enforce_pbc \
  --out_dir imported_models/

```

**Key Arguments:**

* `--pattern`: Determines how interfaces are mapped.
* `erosion`: Minimizes contact core to represent high contact resistance without destroying the conduction network.
* `dilation` (default): Expands contact zones to heal voxelization defects.


* `--enforce_pbc`: Crucial for accurate FEM homogenization. Applies a 3D mirror reflection (increasing volume 8x) to enforce perfect Periodic Boundary Conditions, preventing artificial resistance walls at the bounding box edges.

### Step 2.2: Execution & Solver Analysis (`run_imported.py`)

Once the image is successfully converted into a `.raw` model, feed it into the analysis pipeline. This script reuses the core logic to guarantee 1:1 comparable metrics with the RSA-generated models.

```bash
python3 run_imported.py \
  --import_path imported_models/experimental_ct_pbc_final \
  --solver both \
  --physics_mode thermal \
  --advanced_metrics \
  --csv_log experimental_results.csv

```

*(Note: Exclude the `.raw` or `.nf` extension when providing `--import_path`)*

---

## 3. Recalculation Mode

You can re-run the computational solvers (chfem, PuMA) or extract new structural descriptors on already generated microstructures (both RSA-generated and imported real images) without rebuilding the geometry. This is highly useful for testing different material properties or recovering from previous solver errors.

To use this mode, use the standalone `run_recalc.py` script. It will read the specified `--csv_log`, locate the corresponding `.raw` and `.nf` files, re-run the solvers, and update the CSV in place.

> **Note:** A backup of your CSV (`[your_csv_name].csv.backup`) is automatically created before any recalculation begins.

**Basic Recalculation:**

```bash
# Re-run both solvers and update basic metrics using the existing properties
python3 run_recalc.py --csv_log comparison_results.csv --solver both

```

**Recalculation with Advanced Metrics and Property Overwrite:**
If you want to test new physical properties (e.g., thermal conductivity) on the exact same geometries and extract heavy morphological descriptors (PoreSpy), use the `--overwrite_props` and `--advanced_metrics` flags.

```bash
# Overwrite specific properties, extract PoreSpy metrics, and re-run only the PuMA solver
python3 run_recalc.py \
  --csv_log comparison_results.csv \
  --overwrite_props \
  --prop_A "0.5" \
  --prop_inter "50.0" \
  --advanced_metrics \
  --solver puma

```

**Key Arguments for Recalculation:**

* `--overwrite_props`: Overwrites the material properties in the existing `.nf` files with the new ones provided via CLI arguments (`--prop_A`, etc.).
* `--advanced_metrics`: Triggers PoreSpy morphological calculations (Specific Surface Area, Local Thickness, Autocorrelation Length).
* `--csv_log`: The target CSV file containing the list of models to process.

---

## 4. Rendering a Review Dashboard

For quick result review, you can render a self-contained HTML dashboard from a CSV log. The dashboard uses fixed styling, inline data bars, and client-side column sorting implemented with embedded JavaScript. No browser automation or external CDN assets are required.

```bash
python3 render_results_dashboard.py \
  --csv comparison_results.csv \
  --output comparison_results_dashboard.html \
  --sort-by Connectivity_Ratio \
  --descending

```

**What the dashboard provides:**

* A compact review table for the latest CSV results.
* Inline data bars for ratio, fraction, conductivity-like, and count columns.
* Click-to-sort column headers for quick manual inspection in a browser.
* A fully self-contained `.html` output that can be opened offline.

**Useful arguments:**

* `--csv`: Input CSV file path.
* `--output`: Output HTML file path.
* `--columns`: Optional explicit column list and order.
* `--sort-by`: Optional initial sort column.
* `--descending`: Use descending order for the initial sort.
* `--max-rows`: Maximum number of rows to include in the HTML table.
* `--title`, `--subtitle`: Optional labels for the dashboard header.
