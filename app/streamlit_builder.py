from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from app.ui_shared import (
        PROJECT_ROOT,
        STATUS_COMPLETED,
        STATUS_ERROR,
        STATUS_PENDING,
        STATUS_RUNNING,
        CACHE_DIR_NAME,
        clean_console_log,
        enforce_preview_cache_limits,
        get_file_mtime_ns,
        load_and_generate_3d_figure,
        render_csv_dashboard,
        render_preview_cache_controls,
        render_output_downloads,
        resolve_project_path,
        ensure_project_dir,
        run_project_command,
    )
except ModuleNotFoundError:
    from ui_shared import (
        PROJECT_ROOT,
        STATUS_COMPLETED,
        STATUS_ERROR,
        STATUS_PENDING,
        STATUS_RUNNING,
        CACHE_DIR_NAME,
        clean_console_log,
        enforce_preview_cache_limits,
        get_file_mtime_ns,
        load_and_generate_3d_figure,
        render_csv_dashboard,
        render_preview_cache_controls,
        render_output_downloads,
        resolve_project_path,
        ensure_project_dir,
        run_project_command,
    )

QUEUE_COLUMNS = ["Recipe", "Stretch", "Seed", "Status"]
DEFAULT_FILLER_PARAMS = {
    "rigidfiber": "length=60:radius=2:mean_dir=0,0,1:kappa=0.0",
    "flake": "radius=10:thickness=2:mean_dir=0,0,1:kappa=0.0",
    "sphere": "radius=5",
    "irregfiber": "length=60:shape=bean:radius=5:ratio=0.5:mean_dir=0,0,1:kappa=0.0",
    "flexfiber": "length=90:radius=2:max_bend_deg=90:max_total_bends=10",
    "agglomerate": "num_fibers=5:length=90:radius=2:max_bend_deg=90:max_total_bends=10",
    "staggered": "radius=15:layer_thickness=2:min_layers=1:max_layers=4:max_offset_pct=30:mean_dir=0,0,1:kappa=0.0",
}
DEFAULT_FILLER_PROP_BY_MODE = {
    "thermal": "300.0",
    "electrical": "1e4",
    "mechanics": "1000.0_0.20",
    "permeability": "1000.0_0.20",
}


def init_state() -> None:
    enforce_preview_cache_limits()
    ensure_project_dir(CACHE_DIR_NAME)
    st.session_state.setdefault(
        "recipe_queue",
        pd.DataFrame(columns=QUEUE_COLUMNS),
    )
    st.session_state.setdefault("active_preview", None)
    st.session_state.setdefault("execution_logs", {})
    st.session_state.setdefault("last_queue_message", None)
    st.session_state.setdefault("filler_params_dict", DEFAULT_FILLER_PARAMS.copy())
    st.session_state.setdefault("sweep_f_type", "rigidfiber")
    st.session_state.setdefault(
        "sweep_params",
        st.session_state.filler_params_dict["rigidfiber"],
    )
    st.session_state.setdefault("builder_physics_mode", "electrical")
    st.session_state.setdefault("builder_solver", "chfem")
    st.session_state.setdefault(
        "builder_default_filler_prop",
        DEFAULT_FILLER_PROP_BY_MODE[st.session_state.builder_physics_mode],
    )
    st.session_state.setdefault("sweep_filler_prop", st.session_state.builder_default_filler_prop)


def on_builder_physics_mode_change() -> None:
    mode = st.session_state.builder_physics_mode or "electrical"
    default_prop = DEFAULT_FILLER_PROP_BY_MODE.get(mode, "1e4")
    st.session_state.builder_default_filler_prop = default_prop
    st.session_state.sweep_filler_prop = default_prop


def on_builder_default_filler_prop_change() -> None:
    st.session_state.sweep_filler_prop = st.session_state.builder_default_filler_prop


def on_filler_type_change() -> None:
    filler_type = st.session_state.sweep_f_type or "rigidfiber"
    st.session_state.sweep_f_type = filler_type
    st.session_state.sweep_params = st.session_state.filler_params_dict.get(
        filler_type, ""
    )


def on_params_change() -> None:
    filler_type = st.session_state.sweep_f_type or "rigidfiber"
    st.session_state.filler_params_dict[filler_type] = st.session_state.sweep_params


def reset_current_params() -> None:
    filler_type = st.session_state.sweep_f_type or "rigidfiber"
    default_params = DEFAULT_FILLER_PARAMS.get(filler_type, "")
    st.session_state.filler_params_dict[filler_type] = default_params
    st.session_state.sweep_params = default_params


def build_recipe_text(filler_type: str, volume_fraction: float, params: str, filler_prop: str) -> str:
    parts = [part.strip() for part in params.split(":") if part.strip()]
    parts = [part for part in parts if not part.startswith("prop=")]
    prop_value = filler_prop.strip().replace(" ", "_")
    if prop_value:
        parts.append(f"prop={prop_value}")
    recipe_prefix = f"{filler_type}:{volume_fraction:.4f}"
    if parts:
        return f"{recipe_prefix}:{':'.join(parts)}"
    return recipe_prefix


def generate_and_add_sweep() -> None:
    vf_min = float(st.session_state.sweep_vf_min)
    vf_max = float(st.session_state.sweep_vf_max)
    stretch_min = float(st.session_state.sweep_s_min)
    stretch_max = float(st.session_state.sweep_s_max)
    if vf_min > vf_max or stretch_min > stretch_max:
        st.session_state.last_queue_message = (
            "Min values must be less than or equal to max values."
        )
        return

    filler_type = st.session_state.sweep_f_type or "rigidfiber"
    params = st.session_state.sweep_params.strip()
    filler_prop = st.session_state.sweep_filler_prop.strip()
    vfs = np.linspace(vf_min, vf_max, int(st.session_state.sweep_vf_steps))
    stretches = np.linspace(
        stretch_min,
        stretch_max,
        int(st.session_state.sweep_s_steps),
    )
    seed_start = int(st.session_state.sweep_seed_start)
    seed_count = int(st.session_state.sweep_seed_count)

    new_rows = []
    for seed_offset in range(seed_count):
        current_seed = seed_start + seed_offset if seed_start >= 0 else -1
        for stretch in stretches:
            for vf in vfs:
                new_rows.append(
                    {
                        "Recipe": build_recipe_text(filler_type, float(vf), params, filler_prop),
                        "Stretch": float(stretch),
                        "Seed": current_seed,
                        "Status": STATUS_PENDING,
                    }
                )

    if not new_rows:
        st.session_state.last_queue_message = "No sweep rows were generated."
        return

    st.session_state.recipe_queue = pd.concat(
        [st.session_state.recipe_queue, pd.DataFrame(new_rows)],
        ignore_index=True,
    )
    st.session_state.last_queue_message = f"Added {len(new_rows)} step(s) to the queue."


def clear_queue() -> None:
    st.session_state.recipe_queue = pd.DataFrame(columns=QUEUE_COLUMNS)
    st.session_state.active_preview = None
    st.session_state.execution_logs = {}
    st.session_state.last_queue_message = "Queue cleared."


def clear_log() -> None:
    st.session_state.execution_logs = {}


def render_page_header() -> None:
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        st.markdown("# :material/layers: microsimflow Builder")
        if st.button(":material/restart_alt: Reset", type="tertiary"):
            st.session_state.clear()
            st.rerun()
    st.caption("Design 3D microstructures and manage simulation sweeps.")


def render_queue_metrics(queue_df: pd.DataFrame) -> None:
    total = len(queue_df)
    completed = int((queue_df["Status"] == STATUS_COMPLETED).sum()) if total else 0
    running = int((queue_df["Status"] == STATUS_RUNNING).sum()) if total else 0
    errors = int((queue_df["Status"] == STATUS_ERROR).sum()) if total else 0
    pending = int((queue_df["Status"] == STATUS_PENDING).sum()) if total else 0
    with st.container(horizontal=True):
        st.metric("Total", f"{total:,}", border=True)
        st.metric("Pending", f"{pending:,}", border=True)
        st.metric("Running", f"{running:,}", border=True)
        st.metric("Completed", f"{completed:,}", border=True)
        st.metric("Errors", f"{errors:,}", border=True)


def run_cli_for_step(
    recipe: str,
    stretch: float,
    seed: int,
    target_basename: str,
    *,
    grid_size: int,
    voxel_size: float,
    bg_type: str,
    phase_a_ratio: float,
    physics_mode: str,
    solver: str,
    csv_file_path: str,
    writer_opt: str,
    vti_fields_opt: str,
    prop_a: str,
    prop_b: str,
    prop_pri: str,
    prop_sec: str,
    tunnel_radius: int,
    contact_radius: int,
    poisson_ratio: float,
    deformation_mode: str,
    fine_volume_tol: float,
    fine_max_tilt_deg: float,
    fine_ledger_cap: float,
    void_phases_opt: str,
    skip_structure_metrics: bool,
    advanced_metrics: bool,
):
    cmd = [
        sys.executable,
        "-m",
        "run_pipeline",
        "--size",
        str(grid_size),
        "--voxel_size",
        str(voxel_size),
        "--bg_type",
        bg_type,
        "--phaseA_ratio",
        str(phase_a_ratio),
        "--physics_mode",
        physics_mode,
        "--solver",
        solver,
        "--basename",
        target_basename,
        "--csv_log",
        csv_file_path,
        "--writer",
        writer_opt,
        "--vti_fields",
        vti_fields_opt,
        "--prop_A",
        prop_a,
        "--prop_B",
        prop_b,
        "--prop_inter2",
        prop_sec,
        "--prop_inter",
        prop_pri,
        "--tunnel_radius",
        str(tunnel_radius),
        "--contact_radius",
        str(contact_radius),
        "--poisson_ratio",
        str(poisson_ratio),
        "--deformation_mode",
        deformation_mode,
        "--fine_volume_tol",
        str(fine_volume_tol),
        "--fine_max_tilt_deg",
        str(fine_max_tilt_deg),
        "--fine_ledger_cap",
        str(fine_ledger_cap),
        "--stretch_ratios",
        str(stretch),
    ]
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    if void_phases_opt:
        cmd.extend(["--void_phases", *[part.strip() for part in void_phases_opt.split(",")]])
    if skip_structure_metrics:
        cmd.append("--skip_structure_metrics")
    if advanced_metrics:
        cmd.append("--advanced_metrics")
    cmd.extend(["--recipe", recipe])
    return run_project_command(cmd)




def render_builder() -> None:
    init_state()

    with st.sidebar:
        st.markdown("### :material/settings: Global settings")
        with st.container(border=True):
            grid_col, voxel_col = st.columns(2)
            grid_size = grid_col.number_input("Grid size", value=100, step=10, min_value=10)
            voxel_size = voxel_col.number_input("Voxel size (m)", value=1e-8, format="%.1e")
            bg_type = st.segmented_control(
                "Background type",
                ["single", "gyroid", "lamellar", "cylinder", "bcc", "sea_island", "island_sea"],
                default="gyroid",
            )
            phase_a_ratio = st.slider("Background ratio A", 0.1, 0.9, 0.5)

        st.space("small")
        st.markdown("### :material/science: Solver and physics")
        with st.container(border=True):
            physics_mode = st.segmented_control(
                "Physics mode",
                ["thermal", "electrical", "mechanics", "permeability"],
                key="builder_physics_mode",
                on_change=on_builder_physics_mode_change,
            )
            solver = st.segmented_control(
                "Solver execution",
                ["skip", "chfem", "puma", "both"],
                key="builder_solver",
            )
            st.text_input(
                "Default filler prop",
                key="builder_default_filler_prop",
                on_change=on_builder_default_filler_prop_change,
                help="Applied to newly generated recipes. Spaces are converted to underscores.",
            )
            st.markdown("**Material properties**")
            if physics_mode == "mechanics":
                p_a, p_b, p_pri, p_sec = "1.0 0.35", "1.0 0.35", "100.0 0.25", "10.0 0.30"
            elif physics_mode == "electrical":
                p_a, p_b, p_pri, p_sec = "1e-4", "1e-4", "1e-1", "1e-3"
            else:
                p_a, p_b, p_pri, p_sec = "0.3", "0.3", "30.0", "3.0"
            mat_col1, mat_col2 = st.columns(2)
            with mat_col1:
                prop_a = st.text_input("Prop A", p_a)
                prop_pri = st.text_input("Prop primary interface", p_pri)
            with mat_col2:
                prop_b = st.text_input("Prop B", p_b)
                prop_sec = st.text_input("Prop secondary interface", p_sec)
            void_phases_opt = ""
            if physics_mode == "permeability":
                void_phases_opt = st.text_input("Void phases", value="0")

        with st.expander("Advanced export and metrics", icon=":material/save:"):
            writer_opt = st.selectbox("Export format", ["vti", "zstd", "arrow"], index=0)
            vti_fields_opt = st.radio("Embed fields to VTI", ["off", "on"], index=0, horizontal=True)
            skip_structure_metrics = st.toggle("Skip structure metrics", value=False)
            advanced_metrics = st.toggle("Calculate advanced metrics", value=False)

        with st.expander("Deformation and microstructure", icon=":material/transform:"):
            poisson_ratio = st.number_input("Poisson ratio", value=0.4, step=0.1)
            deformation_mode = st.radio("Deformation mode", ["fine", "coarse"], index=0, horizontal=True)
            radius_col, contact_col = st.columns(2)
            tunnel_radius = radius_col.number_input("Tunnel radius", value=2, step=1, min_value=1)
            contact_radius = contact_col.number_input("Contact radius", value=1, step=1, min_value=1)
            fine_volume_tol = st.number_input("Fine volume tolerance", value=0.01, format="%.3f")
            fine_max_tilt_deg = st.number_input("Fine max tilt (deg)", value=0.10, format="%.2f")
            fine_ledger_cap = st.number_input("Fine ledger cap", value=0.01, format="%.3f")

        st.space("small")
        csv_file_path = st.text_input("CSV log path", value="comparison_results.csv", icon=":material/description:")
        base_name = st.text_input("Export basename", value="model", icon=":material/drive_file_rename_outline:")
        st.space("small")
        render_preview_cache_controls()
        st.space("small")
        render_output_downloads(csv_file_path)

    render_page_header()
    active_tab = st.segmented_control(
        "Navigation",
        [
            ":material/construction: 1. Recipe and sweep",
            ":material/visibility: 2. Execute and preview",
            ":material/dashboard: 3. Dashboard",
        ],
        default=":material/construction: 1. Recipe and sweep",
        label_visibility="collapsed",
    )

    if active_tab == ":material/construction: 1. Recipe and sweep":
        with st.container(border=True):
            st.markdown("#### :material/tune: Configure sweep parameters")
            filler_options = list(st.session_state.filler_params_dict.keys())
            st.pills(
                "Filler type",
                filler_options,
                key="sweep_f_type",
                on_change=on_filler_type_change,
            )

            param_col, prop_col = st.columns([3, 1])
            with param_col:
                input_col, reset_col = st.columns([6, 1])
                input_col.text_input(
                    "Parameters",
                    key="sweep_params",
                    on_change=on_params_change,
                    help="Colon-separated key=value pairs for the selected filler.",
                )
                reset_col.button(
                    "Default",
                    key="reset_sweep_params",
                    icon=":material/undo:",
                    type="tertiary",
                    width="content",
                    on_click=reset_current_params,
                    help="Reset the selected filler's parameters.",
                )
            with prop_col:
                st.text_input(
                    "Filler prop",
                    key="sweep_filler_prop",
                    help="Optional recipe prop override. Spaces are converted to underscores.",
                )

            vf_col, stretch_col, seed_col = st.columns(3)
            with vf_col:
                st.markdown("**Volume fraction Vf**")
                c1, c2, c3 = st.columns(3)
                c1.number_input("Min", value=0.05, step=0.01, key="sweep_vf_min", format="%.3f")
                c2.number_input("Max", value=0.05, step=0.01, key="sweep_vf_max", format="%.3f")
                c3.number_input("Steps", value=1, step=1, min_value=1, key="sweep_vf_steps")
            with stretch_col:
                st.markdown("**Stretch ratio**")
                c4, c5, c6 = st.columns(3)
                c4.number_input("Min", value=1.0, step=0.1, key="sweep_s_min", format="%.2f")
                c5.number_input("Max", value=1.0, step=0.1, key="sweep_s_max", format="%.2f")
                c6.number_input("Steps", value=1, step=1, min_value=1, key="sweep_s_steps")
            with seed_col:
                st.markdown("**Random seed**")
                c7, c8 = st.columns(2)
                c7.number_input("Start", value=-1, step=1, key="sweep_seed_start", help="-1 uses random seeds")
                c8.number_input("Count", value=1, step=1, min_value=1, key="sweep_seed_count")

            with st.container(horizontal=True, horizontal_alignment="right"):
                st.button(
                    "Generate and append",
                    type="primary",
                    on_click=generate_and_add_sweep,
                    icon=":material/add_circle:",
                )
            if st.session_state.last_queue_message:
                st.info(st.session_state.last_queue_message, icon=":material/info:")

        st.space("small")
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.markdown("#### :material/format_list_bulleted: Simulation queue")
            st.button("Clear queue", icon=":material/delete_sweep:", on_click=clear_queue)

        queue_df = st.session_state.recipe_queue.copy()
        if queue_df.empty:
            st.info("Add a sweep recipe to create simulation steps.", icon=":material/info:")
        else:
            render_queue_metrics(queue_df)

        edited_queue = st.data_editor(
            queue_df,
            key="queue_editor",
            num_rows="dynamic",
            hide_index=True,
            disabled=["Status"],
            column_config={
                "Recipe": st.column_config.TextColumn("Microstructure recipe", width="large"),
                "Stretch": st.column_config.NumberColumn("Stretch", format="%.2f", width="small"),
                "Seed": st.column_config.NumberColumn("Seed", format="%d", width="small", help="-1 uses random seeds"),
                "Status": st.column_config.TextColumn("Status", width="small"),
            },
            height=320,
        )
        if isinstance(edited_queue, pd.DataFrame) and not edited_queue.equals(st.session_state.recipe_queue):
            st.session_state.recipe_queue = edited_queue.reindex(columns=QUEUE_COLUMNS)

    elif active_tab == ":material/visibility: 2. Execute and preview":
        if st.session_state.recipe_queue.empty:
            st.info("Please add recipes in step 1 first.", icon=":material/info:")
        else:
            with st.container(border=True):
                st.markdown("### :material/play_circle: Pipeline execution control")
                options = [
                    f"[{idx}] {row.Recipe} (L={row.Stretch:.2f}, Seed={int(row.Seed)})"
                    for idx, row in st.session_state.recipe_queue.iterrows()
                ]
                selected_label = st.selectbox("Target step", options)
                target_idx = options.index(selected_label)

                @st.fragment
                def execution_block() -> None:
                    queue_placeholder = st.empty()
                    log_placeholder = st.empty()
                    with st.container(horizontal=True):
                        if st.button("Stage preview", type="secondary", icon=":material/visibility:"):
                            st.session_state.active_preview = selected_label
                            st.rerun()

                        if st.button("Run selected step", type="primary", icon=":material/play_arrow:"):
                            row = st.session_state.recipe_queue.iloc[target_idx]
                            hash_text = f"{row['Recipe']}_{row['Stretch']}_{row['Seed']}_{grid_size}"
                            hash_str = hashlib.md5(hash_text.encode()).hexdigest()[:8]
                            enforce_preview_cache_limits()
                            ensure_project_dir(CACHE_DIR_NAME)
                            out_base = os.path.join(CACHE_DIR_NAME, f"{base_name}_{hash_str}")
                            st.session_state.recipe_queue.at[target_idx, "Status"] = STATUS_RUNNING
                            queue_placeholder.dataframe(st.session_state.recipe_queue, hide_index=True)
                            res = run_cli_for_step(
                                row["Recipe"],
                                float(row["Stretch"]),
                                int(row["Seed"]),
                                out_base,
                                grid_size=int(grid_size),
                                voxel_size=float(voxel_size),
                                bg_type=bg_type,
                                phase_a_ratio=float(phase_a_ratio),
                                physics_mode=physics_mode,
                                solver=solver,
                                csv_file_path=csv_file_path,
                                writer_opt=writer_opt,
                                vti_fields_opt=vti_fields_opt,
                                prop_a=prop_a,
                                prop_b=prop_b,
                                prop_pri=prop_pri,
                                prop_sec=prop_sec,
                                tunnel_radius=int(tunnel_radius),
                                contact_radius=int(contact_radius),
                                poisson_ratio=float(poisson_ratio),
                                deformation_mode=deformation_mode,
                                fine_volume_tol=float(fine_volume_tol),
                                fine_max_tilt_deg=float(fine_max_tilt_deg),
                                fine_ledger_cap=float(fine_ledger_cap),
                                void_phases_opt=void_phases_opt,
                                skip_structure_metrics=bool(skip_structure_metrics),
                                advanced_metrics=bool(advanced_metrics),
                            )
                            st.session_state.execution_logs[target_idx] = clean_console_log(res.stdout)
                            st.session_state.recipe_queue.at[target_idx, "Status"] = (
                                STATUS_COMPLETED if res.returncode == 0 else STATUS_ERROR
                            )
                            queue_placeholder.empty()
                            st.rerun()

                        if st.button("Run all pending steps", type="secondary", icon=":material/fast_forward:"):
                            for idx, row in st.session_state.recipe_queue.iterrows():
                                if row["Status"] == STATUS_COMPLETED:
                                    continue
                                st.session_state.recipe_queue.at[idx, "Status"] = STATUS_RUNNING
                                queue_placeholder.dataframe(st.session_state.recipe_queue, hide_index=True)
                                hash_text = f"{row['Recipe']}_{row['Stretch']}_{row['Seed']}_{grid_size}"
                                hash_str = hashlib.md5(hash_text.encode()).hexdigest()[:8]
                                enforce_preview_cache_limits()
                                ensure_project_dir(CACHE_DIR_NAME)
                                out_base = os.path.join(CACHE_DIR_NAME, f"{base_name}_{hash_str}")
                                res = run_cli_for_step(
                                    row["Recipe"],
                                    float(row["Stretch"]),
                                    int(row["Seed"]),
                                    out_base,
                                    grid_size=int(grid_size),
                                    voxel_size=float(voxel_size),
                                    bg_type=bg_type,
                                    phase_a_ratio=float(phase_a_ratio),
                                    physics_mode=physics_mode,
                                    solver=solver,
                                    csv_file_path=csv_file_path,
                                    writer_opt=writer_opt,
                                    vti_fields_opt=vti_fields_opt,
                                    prop_a=prop_a,
                                    prop_b=prop_b,
                                    prop_pri=prop_pri,
                                    prop_sec=prop_sec,
                                    tunnel_radius=int(tunnel_radius),
                                    contact_radius=int(contact_radius),
                                    poisson_ratio=float(poisson_ratio),
                                    deformation_mode=deformation_mode,
                                    fine_volume_tol=float(fine_volume_tol),
                                    fine_max_tilt_deg=float(fine_max_tilt_deg),
                                    fine_ledger_cap=float(fine_ledger_cap),
                                    void_phases_opt=void_phases_opt,
                                    skip_structure_metrics=bool(skip_structure_metrics),
                                    advanced_metrics=bool(advanced_metrics),
                                )
                                st.session_state.execution_logs[idx] = clean_console_log(res.stdout)
                                st.session_state.recipe_queue.at[idx, "Status"] = (
                                    STATUS_COMPLETED if res.returncode == 0 else STATUS_ERROR
                                )
                                if res.returncode != 0:
                                    break
                            queue_placeholder.empty()
                            st.rerun()

                        if target_idx in st.session_state.execution_logs:
                            st.button("Clear logs", icon=":material/clear_all:", on_click=clear_log)

                    if target_idx in st.session_state.execution_logs:
                        with log_placeholder.expander(
                            f"Output log for step {target_idx}",
                            expanded=True,
                            icon=":material/terminal:",
                        ):
                            st.code(st.session_state.execution_logs[target_idx])

                execution_block()

            st.space("medium")

            @st.fragment
            def preview_block() -> None:
                if not st.session_state.active_preview:
                    return
                try:
                    active_idx = options.index(st.session_state.active_preview)
                    row = st.session_state.recipe_queue.iloc[active_idx]
                except ValueError:
                    st.warning("Selected preview step is no longer in the queue.")
                    return

                stretch = float(row["Stretch"])
                hash_text = f"{row['Recipe']}_{stretch}_{row['Seed']}_{grid_size}"
                hash_str = hashlib.md5(hash_text.encode()).hexdigest()[:8]
                enforce_preview_cache_limits()
                ensure_project_dir(CACHE_DIR_NAME)
                raw_file = os.path.join(
                    CACHE_DIR_NAME,
                    f"{base_name}_{hash_str}_L{stretch:.2f}.raw",
                )
                raw_path = resolve_project_path(raw_file)

                with st.container(border=True):
                    st.markdown(f"### :material/analytics: Visualization: {st.session_state.active_preview}")
                    if not raw_path.exists():
                        st.info(
                            "No geometry binary found for this step. Run the step first.",
                            icon=":material/info:",
                        )
                        return

                    lam_nu = stretch ** (-float(poisson_ratio))
                    nz = max(1, int(round(int(grid_size) * lam_nu)))
                    ny = max(1, int(round(int(grid_size) * lam_nu)))
                    nx = max(1, int(round(int(grid_size) * stretch)))
                    try:
                        with st.spinner("Computing 3D isosurfaces..."):
                            fig_3d, final_grid = load_and_generate_3d_figure(
                                str(raw_path),
                                nz,
                                ny,
                                nx,
                                get_file_mtime_ns(raw_path),
                            )
                        col_3d, col_2d = st.columns([1.2, 1])
                        with col_3d:
                            st.markdown("**3D isosurface preview**")
                            if fig_3d:
                                st.plotly_chart(fig_3d)
                            else:
                                st.warning("No continuous filler phases were detected.")
                        with col_2d:
                            st.markdown("**2D voxel slice explorer**")
                            z_slice = st.slider("Z-axis slice", 0, nz - 1, nz // 2)
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
        page_title="microsimflow Builder",
        page_icon=":material/hive:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_builder()


if __name__ == "__main__":
    main()
