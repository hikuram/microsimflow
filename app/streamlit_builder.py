from __future__ import annotations

import hashlib
import itertools
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

SUGGESTED_PARAMS_BY_TYPE = {
    "rigidfiber": ["length", "radius", "mean_dir", "kappa", "protrusion_coef", "prop"],
    "flexfiber": ["length", "radius", "max_bend_deg", "max_total_bends", "protrusion_coef", "prop"],
    "flake": ["radius", "thickness", "mean_dir", "kappa", "protrusion_coef", "prop"],
    "sphere": ["radius", "protrusion_coef", "prop"],
    "irregfiber": ["length", "shape", "radius", "ratio", "mean_dir", "kappa", "prop"],
    "agglomerate": ["num_fibers", "length", "radius", "max_bend_deg", "prop"],
    "staggered": ["radius", "layer_thickness", "min_layers", "max_layers", "max_offset_pct", "mean_dir", "kappa", "prop"],
}

PARAM_DEFAULT_FALLBACKS = {
    "protrusion_coef": "0.0025",
}

def stage_key(stage_id: int, name: str) -> str:
    return f"filler_stage_{stage_id}_{name}"


def init_filler_stage(stage_id: int) -> None:
    filler_type_key = stage_key(stage_id, "type")
    params_key = stage_key(stage_id, "params")
    vf_min_key = stage_key(stage_id, "vf_min")
    vf_max_key = stage_key(stage_id, "vf_max")
    vf_steps_key = stage_key(stage_id, "vf_steps")

    default_type = "rigidfiber"
    st.session_state.setdefault(filler_type_key, default_type)
    st.session_state.setdefault(params_key, DEFAULT_FILLER_PARAMS[default_type])
    st.session_state.setdefault(vf_min_key, 0.05)
    st.session_state.setdefault(vf_max_key, 0.05)
    st.session_state.setdefault(vf_steps_key, 1)


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
    st.session_state.setdefault("builder_physics_mode", "electrical")
    st.session_state.setdefault("builder_solver", "chfem")
    st.session_state.setdefault(
        "builder_default_filler_prop",
        DEFAULT_FILLER_PROP_BY_MODE[st.session_state.builder_physics_mode],
    )
    st.session_state.setdefault("filler_stage_ids", [0])
    st.session_state.setdefault("next_filler_stage_id", 1)
    for stage_id in st.session_state.filler_stage_ids:
        init_filler_stage(int(stage_id))


def on_builder_physics_mode_change() -> None:
    mode = st.session_state.builder_physics_mode or "electrical"
    default_prop = DEFAULT_FILLER_PROP_BY_MODE.get(mode, "1e4")
    st.session_state.builder_default_filler_prop = default_prop


def add_filler_stage() -> None:
    stage_id = int(st.session_state.next_filler_stage_id)
    st.session_state.filler_stage_ids.append(stage_id)
    st.session_state.next_filler_stage_id = stage_id + 1
    init_filler_stage(stage_id)


def remove_filler_stage(stage_id: int) -> None:
    if len(st.session_state.filler_stage_ids) <= 1:
        st.session_state.last_queue_message = "Keep at least one filler stage."
        return
    st.session_state.filler_stage_ids = [
        int(value)
        for value in st.session_state.filler_stage_ids
        if int(value) != int(stage_id)
    ]


def on_stage_filler_type_change(stage_id: int) -> None:
    type_key = stage_key(stage_id, "type")
    params_key = stage_key(stage_id, "params")
    filler_type = st.session_state.get(type_key) or "rigidfiber"
    st.session_state[params_key] = DEFAULT_FILLER_PARAMS.get(filler_type, "")


def reset_stage_params(stage_id: int) -> None:
    type_key = stage_key(stage_id, "type")
    params_key = stage_key(stage_id, "params")
    filler_type = st.session_state.get(type_key) or "rigidfiber"
    st.session_state[params_key] = DEFAULT_FILLER_PARAMS.get(filler_type, "")


def parse_param_entries(params: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for part in params.split(":"):
        item = part.strip()
        if not item:
            continue
        key, value = item.split("=", 1) if "=" in item else (item, "")
        entries.append((key.strip(), value.strip()))
    return entries


def get_existing_param_keys(params: str) -> set[str]:
    return {key for key, _ in parse_param_entries(params)}


def get_default_param_value(filler_type: str, param_name: str) -> str:
    for key, value in parse_param_entries(DEFAULT_FILLER_PARAMS.get(filler_type, "")):
        if key == param_name:
            return value

    if param_name == "prop":
        return normalize_prop_value(
            str(st.session_state.get("builder_default_filler_prop", ""))
        )

    return PARAM_DEFAULT_FALLBACKS.get(param_name, "")


def append_param_to_recipe(stage_id: int, param_name: str) -> None:
    if not param_name:
        return

    filler_type = st.session_state.get(stage_key(stage_id, "type")) or "rigidfiber"
    params_key = stage_key(stage_id, "params")
    current_val = st.session_state.get(params_key, "").strip()
    if param_name in get_existing_param_keys(current_val):
        return

    default_value = get_default_param_value(filler_type, param_name)
    new_entry = f"{param_name}={default_value}" if default_value else f"{param_name}="

    if current_val and not current_val.endswith(":"):
        st.session_state[params_key] = f"{current_val}:{new_entry}"
    else:
        st.session_state[params_key] = f"{current_val}{new_entry}"


def add_selected_param(stage_id: int) -> None:
    candidate_key = stage_key(stage_id, "param_candidate")
    selected_param = st.session_state.get(candidate_key)
    param_name = str(selected_param or "")
    if not param_name:
        return

    params_key = stage_key(stage_id, "params")
    current_params = st.session_state.get(params_key, "")
    if param_name in get_existing_param_keys(current_params):
        st.session_state[candidate_key] = None
        return

    append_param_to_recipe(stage_id, param_name)


def normalize_prop_value(value: str) -> str:
    return value.strip().replace(" ", "_")


def build_recipe_text(
    filler_type: str,
    volume_fraction: float,
    params: str,
) -> str:
    raw_parts = [part.strip() for part in params.split(":") if part.strip()]
    parts = []
    has_prop = False

    for part in raw_parts:
        if part.startswith("prop="):
            prop_value = normalize_prop_value(part.split("=", 1)[1])
            if prop_value:
                parts.append(f"prop={prop_value}")
                has_prop = True
        else:
            parts.append(part)

    if not has_prop:
        default_prop = normalize_prop_value(
            str(st.session_state.get("builder_default_filler_prop", ""))
        )
        if default_prop:
            parts.append(f"prop={default_prop}")

    recipe_prefix = f"{filler_type}:{volume_fraction:.4f}"
    if parts:
        return f"{recipe_prefix}:{':'.join(parts)}"
    return recipe_prefix


def build_stage_recipe(stage_id: int, volume_fraction: float) -> str:
    filler_type = st.session_state.get(stage_key(stage_id, "type")) or "rigidfiber"
    params = st.session_state.get(stage_key(stage_id, "params"), "").strip()
    return build_recipe_text(filler_type, volume_fraction, params)


def generate_and_add_sweep() -> None:
    stretch_min = float(st.session_state.sweep_s_min)
    stretch_max = float(st.session_state.sweep_s_max)
    if stretch_min > stretch_max:
        st.session_state.last_queue_message = (
            "Stretch min must be less than or equal to max."
        )
        return

    stage_ids = [int(value) for value in st.session_state.filler_stage_ids]
    vf_arrays = []
    for stage_id in stage_ids:
        vf_min = float(st.session_state[stage_key(stage_id, "vf_min")])
        vf_max = float(st.session_state[stage_key(stage_id, "vf_max")])
        vf_steps = int(st.session_state[stage_key(stage_id, "vf_steps")])
        if vf_min > vf_max:
            st.session_state.last_queue_message = (
                "Filler volume fraction min must be less than or equal to max."
            )
            return
        vf_arrays.append(np.linspace(vf_min, vf_max, vf_steps))

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
            for vf_tuple in itertools.product(*vf_arrays):
                recipes = [
                    build_stage_recipe(stage_id, float(vf))
                    for stage_id, vf in zip(stage_ids, vf_tuple, strict=True)
                    if float(vf) > 0
                ]
                if not recipes:
                    continue
                new_rows.append(
                    {
                        "Recipe": " ".join(recipes),
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
    feature_size: float,
    diffusion_factor: float,
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
        "--feature_size",
        str(feature_size),
        "--diffusion_factor",
        str(diffusion_factor),
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
        cmd.extend(
            ["--void_phases", *[part.strip() for part in void_phases_opt.split(",")]]
        )
    if skip_structure_metrics:
        cmd.append("--skip_structure_metrics")
    if advanced_metrics:
        cmd.append("--advanced_metrics")
    recipe_parts = [part.strip() for part in str(recipe).split() if part.strip()]
    if recipe_parts:
        cmd.append("--recipe")
        cmd.extend(recipe_parts)
    return run_project_command(cmd)


def render_builder() -> None:
    init_state()

    with st.sidebar:
        st.markdown("### :material/settings: Global settings")
        with st.container(border=True):
            grid_col, voxel_col = st.columns(2)
            grid_size = grid_col.number_input(
                "Grid size",
                value=100,
                step=10,
                min_value=10,
            )
            voxel_size = voxel_col.number_input(
                "Voxel size (m)",
                value=1e-8,
                format="%.1e",
            )
            bg_type = st.segmented_control(
                "Background type",
                [
                    "single",
                    "gyroid",
                    "lamellar",
                    "cylinder",
                    "bcc",
                    "sea_island",
                    "island_sea",
                ],
                default="gyroid",
            )
            phase_a_ratio = st.slider("Background ratio A", 0.1, 0.9, 0.5)

            feat_col, diff_col = st.columns(2)
            feature_size = feat_col.number_input(
                "Feature size (px)",
                value=10.0,
                min_value=1.0,
                step=1.0,
                help="Wavelength for TPMS, Radius for Islands"
            )
            diffusion_factor = diff_col.number_input(
                "Diffusion factor",
                value=0.0,
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                format="%.2f",
                help="0.0 = Sharp interface. >0 = Mutual diffusion"
            )

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
            st.markdown("**Material properties**")
            if physics_mode == "mechanics":
                p_a, p_b, p_pri, p_sec = (
                    "1.0 0.35",
                    "1.0 0.35",
                    "100.0 0.25",
                    "10.0 0.30",
                )
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
            st.text_input(
                "Default filler prop",
                key="builder_default_filler_prop",
                help=(
                    "Applied when a filler stage Parameters field does not "
                    "include prop=. Spaces are converted to underscores."
                ),
            )

        with st.expander("Advanced export and metrics", icon=":material/save:"):
            writer_opt = st.selectbox("Export format", ["vti", "zstd", "arrow"], index=0)
            vti_fields_opt = st.radio(
                "Embed fields to VTI",
                ["off", "on"],
                index=0,
                horizontal=True,
            )
            skip_structure_metrics = st.toggle("Skip structure metrics", value=False)
            advanced_metrics = st.toggle("Calculate advanced metrics", value=False)

        with st.expander("Deformation and microstructure", icon=":material/transform:"):
            poisson_ratio = st.number_input("Poisson ratio", value=0.4, step=0.1)
            deformation_mode = st.radio(
                "Deformation mode",
                ["fine", "coarse"],
                index=0,
                horizontal=True,
            )
            radius_col, contact_col = st.columns(2)
            tunnel_radius = radius_col.number_input(
                "Tunnel radius",
                value=2,
                step=1,
                min_value=1,
            )
            contact_radius = contact_col.number_input(
                "Contact radius",
                value=1,
                step=1,
                min_value=1,
            )
            fine_volume_tol = st.number_input(
                "Fine volume tolerance",
                value=0.01,
                format="%.3f",
            )
            fine_max_tilt_deg = st.number_input(
                "Fine max tilt (deg)",
                value=0.10,
                format="%.2f",
            )
            fine_ledger_cap = st.number_input(
                "Fine ledger cap",
                value=0.01,
                format="%.3f",
            )

        st.space("small")
        csv_file_path = st.text_input(
            "CSV log path",
            value="comparison_results.csv",
            icon=":material/description:",
        )
        base_name = st.text_input(
            "Export basename",
            value="model",
            icon=":material/drive_file_rename_outline:",
        )
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
            with st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            ):
                st.markdown("#### :material/tune: Configure sweep parameters")
                st.button(
                    "Add filler",
                    icon=":material/add:",
                    type="tertiary",
                    on_click=add_filler_stage,
                )

            st.caption(
                "Each filler stage is added to the same model. "
                "Volume-fraction sweeps are expanded as combinations; "
                "stretch and random seed are shared."
            )

            filler_options = list(DEFAULT_FILLER_PARAMS.keys())
            stage_ids = [int(value) for value in st.session_state.filler_stage_ids]
            for stage_number, stage_id in enumerate(stage_ids, start=1):
                with st.container(border=True):
                    with st.container(
                        horizontal=True,
                        horizontal_alignment="distribute",
                        vertical_alignment="center",
                    ):
                        st.markdown(f"**Filler stage {stage_number}**")
                        st.button(
                            "Remove",
                            key=f"remove_stage_{stage_id}",
                            icon=":material/close:",
                            type="tertiary",
                            width="content",
                            disabled=len(stage_ids) <= 1,
                            on_click=remove_filler_stage,
                            args=(stage_id,),
                        )

                    st.pills(
                        "Filler type",
                        filler_options,
                        key=stage_key(stage_id, "type"),
                        on_change=on_stage_filler_type_change,
                        args=(stage_id,),
                    )

                    filler_type = st.session_state.get(stage_key(stage_id, "type")) or "rigidfiber"

                    input_col, reset_col = st.columns([6, 1])
                    input_col.text_input(
                        "Parameters",
                        key=stage_key(stage_id, "params"),
                        help=(
                            "Colon-separated key=value pairs. Add prop=... here "
                            "to override the sidebar default for this filler stage."
                        ),
                    )
                    reset_col.button(
                        "Default",
                        key=f"reset_stage_params_{stage_id}",
                        icon=":material/undo:",
                        type="tertiary",
                        width="content",
                        on_click=reset_stage_params,
                        args=(stage_id,),
                        help="Reset this stage's filler parameters.",
                    )

                    suggestions = SUGGESTED_PARAMS_BY_TYPE.get(filler_type, ["prop"])
                    add_col, select_col = st.columns([1, 5])
                    add_col.button(
                        "+ Add key",
                        key=f"add_stage_param_{stage_id}",
                        type="secondary",
                        width="stretch",
                        help="Add the selected parameter key.",
                        on_click=add_selected_param,
                        args=(stage_id,),
                    )
                    selected_candidate = select_col.selectbox(
                        "Add parameter key",
                        suggestions,
                        index=None,
                        key=stage_key(stage_id, "param_candidate"),
                        placeholder="Choose a parameter to add",
                        label_visibility="collapsed"
                    )
                    if selected_candidate in get_existing_param_keys(
                        st.session_state.get(stage_key(stage_id, "params"), "")
                    ):
                        select_col.caption("This key is already included.")

                    st.markdown("**Volume fraction Vf**")
                    c1, c2, c3 = st.columns(3)
                    c1.number_input(
                        "Min",
                        value=0.05,
                        step=0.01,
                        key=stage_key(stage_id, "vf_min"),
                        format="%.3f",
                    )
                    c2.number_input(
                        "Max",
                        value=0.05,
                        step=0.01,
                        key=stage_key(stage_id, "vf_max"),
                        format="%.3f",
                    )
                    c3.number_input(
                        "Steps",
                        value=1,
                        step=1,
                        min_value=1,
                        key=stage_key(stage_id, "vf_steps"),
                    )

            st.divider()
            stretch_col, seed_col = st.columns(2)
            with stretch_col:
                st.markdown("**Shared stretch ratio**")
                c4, c5, c6 = st.columns(3)
                c4.number_input(
                    "Min",
                    value=1.0,
                    step=0.1,
                    key="sweep_s_min",
                    format="%.2f",
                )
                c5.number_input(
                    "Max",
                    value=1.0,
                    step=0.1,
                    key="sweep_s_max",
                    format="%.2f",
                )
                c6.number_input(
                    "Steps",
                    value=1,
                    step=1,
                    min_value=1,
                    key="sweep_s_steps",
                )
            with seed_col:
                st.markdown("**Shared random seed**")
                c7, c8 = st.columns(2)
                c7.number_input(
                    "Start",
                    value=-1,
                    step=1,
                    key="sweep_seed_start",
                    help="-1 uses random seeds",
                )
                c8.number_input(
                    "Count",
                    value=1,
                    step=1,
                    min_value=1,
                    key="sweep_seed_count",
                )

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
        with st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        ):
            st.markdown("#### :material/format_list_bulleted: Simulation queue")
            st.button("Clear queue", icon=":material/delete_sweep:", on_click=clear_queue)

        queue_df = st.session_state.recipe_queue.copy()
        if queue_df.empty:
            st.info(
                "Add a sweep recipe to create simulation steps.",
                icon=":material/info:",
            )
        else:
            render_queue_metrics(queue_df)

        edited_queue = st.data_editor(
            queue_df,
            key="queue_editor",
            num_rows="dynamic",
            hide_index=True,
            disabled=["Status"],
            column_config={
                "Recipe": st.column_config.TextColumn(
                    "Microstructure recipe",
                    width="large",
                ),
                "Stretch": st.column_config.NumberColumn(
                    "Stretch",
                    format="%.2f",
                    width="small",
                ),
                "Seed": st.column_config.NumberColumn(
                    "Seed",
                    format="%d",
                    width="small",
                    help="-1 uses random seeds",
                ),
                "Status": st.column_config.TextColumn("Status", width="small"),
            },
            height=320,
        )
        if isinstance(edited_queue, pd.DataFrame) and not edited_queue.equals(
            st.session_state.recipe_queue
        ):
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
                        if st.button(
                            "Stage preview",
                            type="secondary",
                            icon=":material/visibility:",
                        ):
                            st.session_state.active_preview = selected_label
                            st.rerun()

                        if st.button(
                            "Run selected step",
                            type="primary",
                            icon=":material/play_arrow:",
                        ):
                            row = st.session_state.recipe_queue.iloc[target_idx]
                            hash_text = (
                                f"{row['Recipe']}_{row['Stretch']}_"
                                f"{row['Seed']}_{grid_size}"
                            )
                            hash_str = hashlib.md5(hash_text.encode()).hexdigest()[:8]
                            enforce_preview_cache_limits()
                            ensure_project_dir(CACHE_DIR_NAME)
                            out_base = os.path.join(
                                CACHE_DIR_NAME,
                                f"{base_name}_{hash_str}",
                            )
                            st.session_state.recipe_queue.at[target_idx, "Status"] = (
                                STATUS_RUNNING
                            )
                            queue_placeholder.dataframe(
                                st.session_state.recipe_queue,
                                hide_index=True,
                            )
                            res = run_cli_for_step(
                                row["Recipe"],
                                float(row["Stretch"]),
                                int(row["Seed"]),
                                out_base,
                                grid_size=int(grid_size),
                                voxel_size=float(voxel_size),
                                bg_type=bg_type,
                                phase_a_ratio=float(phase_a_ratio),
                                feature_size=float(feature_size),
                                diffusion_factor=float(diffusion_factor),
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
                            st.session_state.execution_logs[target_idx] = (
                                clean_console_log(res.stdout)
                            )
                            st.session_state.recipe_queue.at[target_idx, "Status"] = (
                                STATUS_COMPLETED
                                if res.returncode == 0
                                else STATUS_ERROR
                            )
                            queue_placeholder.empty()
                            st.rerun()

                        if st.button(
                            "Run all pending steps",
                            type="secondary",
                            icon=":material/fast_forward:",
                        ):
                            for idx, row in st.session_state.recipe_queue.iterrows():
                                if row["Status"] == STATUS_COMPLETED:
                                    continue
                                st.session_state.recipe_queue.at[idx, "Status"] = (
                                    STATUS_RUNNING
                                )
                                queue_placeholder.dataframe(
                                    st.session_state.recipe_queue,
                                    hide_index=True,
                                )
                                hash_text = (
                                    f"{row['Recipe']}_{row['Stretch']}_"
                                    f"{row['Seed']}_{grid_size}"
                                )
                                hash_str = hashlib.md5(
                                    hash_text.encode()
                                ).hexdigest()[:8]
                                enforce_preview_cache_limits()
                                ensure_project_dir(CACHE_DIR_NAME)
                                out_base = os.path.join(
                                    CACHE_DIR_NAME,
                                    f"{base_name}_{hash_str}",
                                )
                                res = run_cli_for_step(
                                    row["Recipe"],
                                    float(row["Stretch"]),
                                    int(row["Seed"]),
                                    out_base,
                                    grid_size=int(grid_size),
                                    voxel_size=float(voxel_size),
                                    bg_type=bg_type,
                                    phase_a_ratio=float(phase_a_ratio),
                                    feature_size=float(feature_size),
                                    diffusion_factor=float(diffusion_factor),
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
                                st.session_state.execution_logs[idx] = clean_console_log(
                                    res.stdout
                                )
                                st.session_state.recipe_queue.at[idx, "Status"] = (
                                    STATUS_COMPLETED
                                    if res.returncode == 0
                                    else STATUS_ERROR
                                )
                                if res.returncode != 0:
                                    break
                            queue_placeholder.empty()
                            st.rerun()

                        if target_idx in st.session_state.execution_logs:
                            st.button(
                                "Clear logs",
                                icon=":material/clear_all:",
                                on_click=clear_log,
                            )

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
                    st.markdown(
                        "### :material/analytics: "
                        f"Visualization: {st.session_state.active_preview}"
                    )
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
                                [0.0, "#440154"], [0.2, "#440154"],   # Polymer A (ID: 0)
                                [0.2, "#31688e"], [0.4, "#31688e"],   # Polymer B (ID: 1)
                                [0.4, "#21918c"], [0.6, "#21918c"],   # Secondary Interface (ID: 2)
                                [0.6, "#85d44a"], [0.8, "#85d44a"],   # Primary Interface (ID: 3)
                                [0.8, "#fde725"], [1.0, "#fde725"],   # Fillers (ID: 4+)
                            ]
                            fig_2d = px.imshow(
                                final_grid[z_slice, :, :],
                                color_continuous_scale=color_scale,
                                zmin=-0.5,
                                zmax=4.5,
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
            if st.button(
                "Generate or refresh HTML report",
                type="primary",
                icon=":material/refresh:",
            ):
                if resolve_project_path(csv_file_path).exists():
                    with st.spinner("Rendering HTML report components..."):
                        res = run_project_command(
                            [
                                sys.executable,
                                "render_results_dashboard.py",
                                "--csv",
                                csv_file_path,
                            ]
                        )
                    if res.returncode == 0:
                        st.success(
                            "Dashboard static build complete.",
                            icon=":material/check_circle:",
                        )
                    else:
                        st.error("Dashboard renderer failed.", icon=":material/error:")
                    with st.expander(
                        "Renderer output log",
                        icon=":material/terminal:",
                    ):
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
