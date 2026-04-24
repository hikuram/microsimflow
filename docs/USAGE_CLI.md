# CLI Usage & Custom Pipelines

This document explains how to use `run_pipeline.py` to design custom simulations, define complex filler recipes, and manage recalculations.

## 1. Running Custom Pipelines

You can easily design and run a custom simulation by passing arguments directly to `run_pipeline.py`. Use `python3 run_pipeline.py --help` to see all available parameters.

### Recipe Formatting
Recipe strings follow the format:
`type:volume_fraction:param1=value1:param2=value2:...`

### Orientation Control (Pseudo-Watson Distribution)
Orientation is controlled via `mean_dir` and `kappa` for `flake`, `rigidfiber`, and `irregfiber`.
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

## 2. Recalculation Mode

You can re-run the computational solvers (chfem, PuMA) on already generated microstructures without undergoing the computationally expensive geometry generation phase. This is highly useful for testing different material properties, running a different solver, or recovering from previous solver errors.

The same pass also recomputes the CSV structure descriptors (`Contact_Ratio`, `Tunneling_Ratio`, `Connectivity_Ratio`, and related voxel counts) from the reconstructed `final_grid` loaded from each `.raw` file.

To use this mode, provide the `--recalc` flag. The script will read the specified `--csv_log`, locate the corresponding `.raw` and `.nf` files in the directory, re-run the solvers, and update the CSV in place.

> **Note:** A backup of your CSV (`[your_csv_name].csv.backup`) is automatically created before any recalculation begins.

**Basic Recalculation:**

```bash
# Re-run both solvers using the existing properties found in the .nf files
python3 run_pipeline.py --recalc --csv_log comparison_results.csv --solver both
```

**Recalculation with Property Overwrite:**
If you want to test new physical properties (e.g., thermal conductivity) on the exact same geometries, use the `--overwrite_props` flag. This forces the script to ignore the old `.nf` properties and apply the new ones provided via the CLI.

```bash
# Overwrite specific properties and re-run only the PuMA solver
python3 run_pipeline.py --recalc --csv_log comparison_results.csv --overwrite_props --prop_A "0.5" --prop_inter "50.0" --solver puma
```

**Key Arguments for Recalculation:**

* `--recalc`: Activates recalculation mode. *(Note: This is mutually exclusive with `--recipe`. You cannot build new models and recalculate old ones in the same command).*
* `--overwrite_props`: Overwrites the material properties in the existing `.nf` files with the new ones provided via CLI arguments.
* `--csv_log`: The target CSV file containing the list of models to process.

## 3. Rendering a Review Dashboard

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
