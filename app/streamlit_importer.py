from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

try:
    from app.ui_shared import (
        CACHE_DIR_NAME,
        PROJECT_ROOT,
        clean_console_log,
        enforce_preview_cache_limits,
        ensure_project_dir,
        get_file_mtime_ns,
        load_and_generate_3d_figure,
        parse_raw_shape,
        read_grid_shape,
        render_csv_dashboard,
        render_preview_cache_controls,
        render_output_downloads,
        resolve_project_path,
        run_project_command,
        safe_base_from_final_raw,
    )
except ModuleNotFoundError:
    from ui_shared import (
        CACHE_DIR_NAME,
        PROJECT_ROOT,
        clean_console_log,
        enforce_preview_cache_limits,
        ensure_project_dir,
        get_file_mtime_ns,
        load_and_generate_3d_figure,
        parse_raw_shape,
        read_grid_shape,
        render_csv_dashboard,
        render_preview_cache_controls,
        render_output_downloads,
        resolve_project_path,
        run_project_command,
        safe_base_from_final_raw,
    )


DEFAULT_MATERIAL_PROPS_BY_MODE = {
    "thermal": ("0.3", "0.3", "30.0", "3.0"),
    "electrical": ("1e-4", "1e-4", "1e-1", "1e-3"),
    "mechanics": ("1.0 0.35", "1.0 0.35", "100.0 0.25", "10.0 0.30"),
    "permeability": ("0.3", "0.3", "30.0", "3.0"),
}
DEFAULT_FILLER_PROP_BY_MODE = {
    "thermal": "300.0",
    "electrical": "1e4",
    "mechanics": "1000.0 0.20",
    "permeability": "1000.0 0.20",
}


def default_material_props(physics_mode: str) -> tuple[str, str, str, str]:
    return DEFAULT_MATERIAL_PROPS_BY_MODE.get(
        physics_mode,
        DEFAULT_MATERIAL_PROPS_BY_MODE["electrical"],
    )


def reset_importer_props_for_physics() -> None:
    physics_mode = st.session_state.importer_physics_mode or "electrical"
    p_a, p_b, p_pri, p_sec = default_material_props(physics_mode)
    st.session_state.importer_prop_a = p_a
    st.session_state.importer_prop_b = p_b
    st.session_state.importer_prop_pri = p_pri
    st.session_state.importer_prop_sec = p_sec
    st.session_state.importer_filler_prop = DEFAULT_FILLER_PROP_BY_MODE.get(
        physics_mode,
        DEFAULT_FILLER_PROP_BY_MODE["electrical"],
    )


def directory_signature(directory: str) -> tuple[tuple[str, int, int], ...]:
    path = resolve_project_path(directory)
    if not path.exists() or not path.is_dir():
        return tuple()
    entries: list[tuple[str, int, int]] = []
    for file_path in sorted(path.glob("*")):
        if file_path.suffix.lower() not in {".raw", ".nf"}:
            continue
        try:
            stat = file_path.stat()
            entries.append((file_path.name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    return tuple(entries)


@st.cache_data(show_spinner=False, ttl="2m")
def scan_directory(directory: str, signature: tuple[tuple[str, int, int], ...]) -> pd.DataFrame:
    del signature
    path = resolve_project_path(directory)
    if not path.exists() or not path.is_dir():
        return pd.DataFrame()

    model_data = []
    for raw_path in sorted(glob.glob(str(path / "*_final.raw"))):
        raw = Path(raw_path)
        base_stem = str(raw)[:-4]
        nf_path = base_stem + ".nf"
        if not os.path.exists(nf_path) and base_stem.endswith("_final"):
            alt_nf = base_stem[:-6] + "_meta.nf"
            if os.path.exists(alt_nf):
                nf_path = alt_nf
        has_nf = os.path.exists(nf_path)
        grid_str = "Unknown"
        voxel_str = "Unknown"
        if has_nf:
            try:
                with open(nf_path, "r", encoding="utf-8") as file:
                    meta = json.load(file)
                grid_str = str(meta.get("grid_size", grid_str))
                voxel_str = str(meta.get("voxel_size_m", voxel_str))
            except Exception:
                pass
        model_data.append(
            {
                "Model Name": raw.name,
                "Meta": "yes" if has_nf else "no",
                "Grid": grid_str,
                "Voxel": voxel_str,
                "Raw Path": str(raw),
                "Nf Path": nf_path if has_nf else None,
            }
        )
    if not model_data:
        return pd.DataFrame()
    return pd.DataFrame(model_data).sort_values("Model Name").reset_index(drop=True)


def init_state() -> None:
    enforce_preview_cache_limits()
    ensure_project_dir(CACHE_DIR_NAME)
    st.session_state.setdefault("importer_active_preview", None)
    st.session_state.setdefault("import_log", None)
    st.session_state.setdefault("import_status", None)
    st.session_state.setdefault("solver_log", None)
    st.session_state.setdefault("solver_status", None)
    st.session_state.setdefault("importer_physics_mode", "electrical")
    st.session_state.setdefault("importer_solver", "chfem")
    p_a, p_b, p_pri, p_sec = default_material_props(
        st.session_state.importer_physics_mode
    )
    st.session_state.setdefault("importer_prop_a", p_a)
    st.session_state.setdefault("importer_prop_b", p_b)
    st.session_state.setdefault("importer_prop_pri", p_pri)
    st.session_state.setdefault("importer_prop_sec", p_sec)
    st.session_state.setdefault(
        "importer_filler_prop",
        DEFAULT_FILLER_PROP_BY_MODE[st.session_state.importer_physics_mode],
    )


def clear_import_log() -> None:
    st.session_state.import_log = None
    st.session_state.import_status = None


def clear_solver_log() -> None:
    st.session_state.solver_log = None
    st.session_state.solver_status = None


def render_page_header() -> None:
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        st.markdown("# :material/image_search: microsimflow Importer")
        if st.button(":material/restart_alt: Reset", type="tertiary"):
            st.session_state.clear()
            st.rerun()
    st.caption("Import, binarize, and analyze real micro-CT or FIB images.")


def render_image_preview(file_path: str, threshold: int, file_format: str, raw_shape: str) -> None:
    path = resolve_project_path(file_path)
    if not path.exists() or not path.is_file():
        st.info("Select an existing image file to preview.", icon=":material/info:")
        return
    try:
        if file_format == "raw":
            shape = [int(part) for part in parse_raw_shape(raw_shape)]
            arr = np.fromfile(path, dtype=np.uint8).reshape(tuple(shape))
            image = arr[shape[0] // 2, :, :]
        else:
            image = np.array(Image.open(path).convert("L"))
        fig_prev = px.imshow(image >= threshold, color_continuous_scale="gray")
        fig_prev.update_layout(
            coloraxis_showscale=False,
            margin={"l": 0, "r": 0, "b": 0, "t": 0},
            height=320,
        )
        st.plotly_chart(fig_prev)
    except Exception as exc:
        st.info(f"No valid preview is available. {exc}")




def render_importer() -> None:
    init_state()

    with st.sidebar:
        st.markdown("### :material/folder_open: Paths and settings")
        with st.container(border=True):
            target_dir = st.text_input("Model directory", value="imported_models")
            csv_file_path = st.text_input("CSV log path", value="imported_results.csv")
            fallback_grid = st.number_input(
                "Fallback grid size",
                value=100,
                step=10,
                min_value=10,
                help="Used when metadata is missing.",
            )
            if st.button("Refresh model scan", icon=":material/refresh:"):
                scan_directory.clear()

        st.space("small")
        st.markdown("### :material/science: Solver execution")
        with st.container(border=True):
            physics_mode = st.segmented_control(
                "Physics mode",
                ["thermal", "electrical", "mechanics", "permeability"],
                key="importer_physics_mode",
                on_change=reset_importer_props_for_physics,
            )
            solver = st.segmented_control(
                "Solver execution",
                ["skip", "chfem", "puma", "both"],
                key="importer_solver",
            )
            st.markdown("**Material properties**")
            prop_col1, prop_col2 = st.columns(2)
            with prop_col1:
                prop_a = st.text_input("Prop A", key="importer_prop_a")
                prop_pri = st.text_input(
                    "Prop primary interface",
                    key="importer_prop_pri",
                )
            with prop_col2:
                prop_b = st.text_input("Prop B", key="importer_prop_b")
                prop_sec = st.text_input(
                    "Prop secondary interface",
                    key="importer_prop_sec",
                )
            filler_prop = st.text_input(
                "Filler prop",
                key="importer_filler_prop",
                help="Filler phase property passed to solver execution.",
            )
            with st.expander("Advanced solver options", icon=":material/tune:"):
                advanced_metrics = st.toggle("Calculate advanced metrics", value=False)
                skip_vti = st.toggle("Skip VTI export", value=False)
                pbc_pad = st.number_input("PBC pad", value=20, min_value=0, step=1)
                void_phases = ""
                if physics_mode == "permeability":
                    void_phases = st.text_input("Void phases", value="0")

        st.space("small")
        render_preview_cache_controls()
        st.space("small")
        render_output_downloads(csv_file_path)

    render_page_header()
    active_tab = st.segmented_control(
        "Navigation",
        [
            ":material/add_photo_alternate: 1. Image importer",
            ":material/visibility: 2. Viewer and solver",
            ":material/dashboard: 3. Dashboard",
        ],
        default=":material/add_photo_alternate: 1. Image importer",
        label_visibility="collapsed",
    )

    if active_tab == ":material/add_photo_alternate: 1. Image importer":
        input_col, preview_col = st.columns([1, 1])
        with input_col:
            with st.container(border=True):
                st.markdown("#### :material/input: Input image and setup")
                input_method = st.segmented_control(
                    "Input source",
                    ["Server path", "Upload file"],
                    default="Server path",
                )
                file_path = ""
                if input_method == "Server path":
                    file_path = st.text_input("File path", value="./sample.tif")
                else:
                    uploaded_file = st.file_uploader(
                        "TIFF, PNG, or RAW",
                        type=["tif", "tiff", "raw", "png"],
                    )
                    if uploaded_file:
                        upload_dir = resolve_project_path("temp_uploads")
                        upload_dir.mkdir(exist_ok=True)
                        safe_name = Path(uploaded_file.name).name
                        upload_path = upload_dir / safe_name
                        upload_path.write_bytes(uploaded_file.getbuffer())
                        file_path = str(upload_path)

                fmt_col, voxel_col = st.columns(2)
                file_format = fmt_col.selectbox("Format", ["tiff", "raw"])
                voxel_size = voxel_col.number_input("Voxel size (m)", 1e-8, format="%.2e")
                raw_shape = ""
                if file_format == "raw":
                    raw_shape = st.text_input("RAW shape Z Y X", "100 100 100")

            st.space("small")
            with st.container(border=True):
                st.markdown("#### :material/imagesearch_roller: Processing rules")
                threshold = st.slider("Binarization threshold", 0, 255, 128)
                rule_col, radius_col = st.columns(2)
                pattern = rule_col.selectbox("Interface pattern", ["dilation", "erosion"])
                tunnel_radius = radius_col.number_input("Tunnel radius", min_value=1, value=2)
                st.markdown("**SNOW segmentation**")
                snow_col1, snow_col2 = st.columns(2)
                with snow_col1:
                    snow_sigma = st.number_input(
                        "SNOW sigma",
                        value=0.4,
                        step=0.1,
                        format="%.2f",
                    )
                    enforce_pbc = st.toggle("Enforce PBC", value=False)
                with snow_col2:
                    snow_r_max = st.number_input(
                        "SNOW r max",
                        value=4,
                        step=1,
                        min_value=1,
                    )
                    save_debug_slices = st.toggle("Save debug slices", value=False)

        with preview_col:
            with st.container(border=True):
                st.markdown("#### :material/preview: Preview and execute")
                render_image_preview(file_path, int(threshold), file_format, raw_shape)

                @st.fragment
                def import_execution_block() -> None:
                    with st.container(horizontal=True):
                        if st.button("Extract and import model", type="primary", icon=":material/rocket_launch:"):
                            path = resolve_project_path(file_path)
                            if not path.exists():
                                st.session_state.import_status = "error"
                                st.session_state.import_log = "Error: input file not found."
                            else:
                                try:
                                    cmd = [
                                        sys.executable,
                                        "import_image.py",
                                        "--input",
                                        str(path),
                                        "--format",
                                        file_format,
                                        "--voxel_size",
                                        str(voxel_size),
                                        "--threshold",
                                        str(threshold),
                                        "--pattern",
                                        pattern,
                                        "--tunnel_radius",
                                        str(tunnel_radius),
                                        "--snow_sigma",
                                        str(snow_sigma),
                                        "--snow_r_max",
                                        str(snow_r_max),
                                        "--out_dir",
                                        target_dir,
                                    ]
                                    if file_format == "raw":
                                        cmd.extend(["--raw_shape", *parse_raw_shape(raw_shape)])
                                    if enforce_pbc:
                                        cmd.append("--enforce_pbc")
                                    if save_debug_slices:
                                        cmd.append("--save_debug_slices")
                                    with st.spinner("Processing image..."):
                                        res = run_project_command(cmd)
                                    st.session_state.import_log = clean_console_log(res.stdout)
                                    st.session_state.import_status = (
                                        "success" if res.returncode == 0 else "error"
                                    )
                                    if res.returncode == 0:
                                        scan_directory.clear()
                                except Exception as exc:
                                    st.session_state.import_log = f"Execution failed: {exc}"
                                    st.session_state.import_status = "error"
                            st.rerun()

                        if st.session_state.import_log is not None:
                            st.button("Clear log", icon=":material/clear_all:", on_click=clear_import_log)

                    if st.session_state.import_log is not None:
                        if st.session_state.import_status == "success":
                            st.success("Import completed successfully.", icon=":material/check_circle:")
                        else:
                            st.error("Import failed.", icon=":material/error:")
                        with st.expander("Import log", expanded=True, icon=":material/terminal:"):
                            st.code(st.session_state.import_log)

                import_execution_block()

    elif active_tab == ":material/visibility: 2. Viewer and solver":
        df_models = scan_directory(target_dir, directory_signature(target_dir))
        if df_models.empty:
            st.info("No models found in the target directory.", icon=":material/info:")
        else:
            with st.container(border=True):
                st.markdown("#### :material/folder_managed: Model explorer")
                st.dataframe(
                    df_models[["Model Name", "Meta", "Grid", "Voxel"]],
                    height=220,
                    hide_index=True,
                )
                selected_model = st.selectbox("Select target model", df_models["Model Name"].tolist())
                target_row = df_models[df_models["Model Name"] == selected_model].iloc[0]
                raw_path = str(target_row["Raw Path"])
                base_path = safe_base_from_final_raw(raw_path)

                @st.fragment
                def solver_execution_block() -> None:
                    log_placeholder = st.empty()
                    with st.container(horizontal=True):
                        if st.button("Stage 3D preview", type="secondary", icon=":material/visibility:"):
                            st.session_state.importer_active_preview = selected_model
                            st.rerun()
                        if st.button("Run solver", type="primary", icon=":material/play_arrow:"):
                            cmd = [
                                sys.executable,
                                "run_imported.py",
                                "--import_path",
                                base_path,
                                "--solver",
                                solver,
                                "--physics_mode",
                                physics_mode,
                                "--csv_log",
                                csv_file_path,
                                "--prop_A",
                                prop_a,
                                "--prop_B",
                                prop_b,
                                "--prop_inter",
                                prop_pri,
                                "--prop_inter2",
                                prop_sec,
                                "--prop_filler",
                                filler_prop,
                                "--pbc_pad",
                                str(pbc_pad),
                            ]
                            if advanced_metrics:
                                cmd.append("--advanced_metrics")
                            if skip_vti:
                                cmd.append("--skip_vti")
                            if void_phases:
                                cmd.extend(["--void_phases", *[part.strip() for part in void_phases.split(",")]])
                            with st.spinner(f"Running {solver} solver..."):
                                res = run_project_command(cmd)
                            st.session_state.solver_log = clean_console_log(res.stdout)
                            st.session_state.solver_status = (
                                "success" if res.returncode == 0 else "error"
                            )
                            st.rerun()
                        if st.session_state.solver_log is not None:
                            st.button("Clear log", icon=":material/clear_all:", on_click=clear_solver_log)

                    if st.session_state.solver_log is not None:
                        if st.session_state.solver_status == "success":
                            st.success("Analysis completed.", icon=":material/check_circle:")
                        else:
                            st.error("Solver execution error.", icon=":material/error:")
                        with log_placeholder.expander("Solver log", expanded=True, icon=":material/terminal:"):
                            st.code(st.session_state.solver_log)

                solver_execution_block()

            st.space("medium")

            @st.fragment
            def preview_block() -> None:
                if st.session_state.importer_active_preview != selected_model:
                    return
                with st.container(border=True):
                    st.markdown(f"### :material/analytics: 3D microstructure: {selected_model}")
                    nz, ny, nx = read_grid_shape(target_row["Nf Path"], int(fallback_grid))
                    try:
                        with st.spinner("Computing 3D isosurfaces..."):
                            fig_3d, final_grid = load_and_generate_3d_figure(
                                raw_path,
                                nz,
                                ny,
                                nx,
                                get_file_mtime_ns(raw_path),
                            )
                        col_3d, col_2d = st.columns([1.2, 1])
                        with col_3d:
                            st.markdown("**3D isosurface**")
                            if fig_3d:
                                st.plotly_chart(fig_3d)
                            else:
                                st.warning("No filler phases were detected.")
                        with col_2d:
                            st.markdown("**2D full slice**")
                            z_slice = st.slider("Z-axis", 0, nz - 1, nz // 2, label_visibility="collapsed")
                            color_scale = [
                                [0, "#1f2937"],
                                [0.2, "#1f2937"],
                                [0.2, "#4b5563"],
                                [0.4, "#4b5563"],
                                [0.4, "#f59e0b"],
                                [0.8, "#f59e0b"],
                                [0.8, "#3b82f6"],
                                [1, "#3b82f6"],
                            ]
                            fig_2d = px.imshow(
                                final_grid[z_slice, :, :],
                                color_continuous_scale=color_scale,
                                origin="lower",
                            )
                            fig_2d.update_layout(
                                coloraxis_showscale=False,
                                margin={"l": 0, "r": 0, "b": 0, "t": 0},
                                height=450,
                            )
                            st.plotly_chart(fig_2d)
                    except ValueError as exc:
                        st.error(f"Voxel grid dimensions mismatch. {exc}")

            preview_block()

    elif active_tab == ":material/dashboard: 3. Dashboard":
        html_out = csv_file_path.replace(".csv", ".html")
        with st.container(border=True):
            st.markdown("### :material/insert_chart: Results dashboard")
            st.caption("Inspect the CSV log and build a standalone HTML report.")
            if st.button("Generate or refresh HTML report", type="primary", icon=":material/refresh:"):
                if resolve_project_path(csv_file_path).exists():
                    with st.spinner("Rendering HTML report components..."):
                        res = run_project_command(
                            [sys.executable, "render_results_dashboard.py", "--csv", csv_file_path]
                        )
                    if res.returncode == 0:
                        st.success("Dashboard static build complete.", icon=":material/check_circle:")
                    else:
                        st.error("Dashboard renderer failed.", icon=":material/error:")
                    with st.expander("Renderer output log", icon=":material/terminal:"):
                        st.code(clean_console_log(res.stdout))
                else:
                    st.error("Target CSV log file not found.")

        st.space("small")
        with st.container(border=True):
            render_csv_dashboard(csv_file_path, html_out=html_out)


def main() -> None:
    st.set_page_config(
        page_title="microsimflow Importer",
        page_icon=":material/image_search:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_importer()


if __name__ == "__main__":
    main()
