import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import subprocess
import hashlib
from skimage.measure import marching_cubes

# ==========================================
# 1. ページ設定
# ==========================================
st.set_page_config(
    page_title="microsimflow Builder", 
    page_icon=":material/hive:", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 共通ユーティリティ (ログ＆キャッシュ)
# ==========================================
def clean_console_log(raw_bytes):
    if not raw_bytes: return ""
    text = raw_bytes.decode('utf-8', errors='replace')
    lines, current_line = [], ""
    for char in text:
        if char == '\n':
            if current_line.strip(): lines.append(current_line)
            current_line = ""
        elif char == '\r': current_line = ""
        else: current_line += char
    if current_line.strip(): lines.append(current_line)
    return '\n'.join(lines)

@st.cache_data(show_spinner=False)
def load_and_generate_3d_figure(raw_path, nz, ny, nx):
    final_grid = np.fromfile(raw_path, dtype=np.uint8).reshape((nz, ny, nx))
    pixz, pixy, pixx = min(nz, 100), min(ny, 100), min(nx, 100)
    preview_grid = final_grid[:pixz, :pixy, :pixx]
    
    traces = []
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']
    filler_ids = np.unique(preview_grid)[np.unique(preview_grid) >= 4]
    
    for i, f_id in enumerate(filler_ids):
        mask = (preview_grid == f_id).astype(float)
        if np.any(mask):
            verts, faces, _, _ = marching_cubes(mask, level=0.5)
            traces.append(go.Mesh3d(
                x=verts[:, 2], y=verts[:, 1], z=verts[:, 0], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=colors[i % len(colors)], opacity=0.7, flatshading=False,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.4), 
                lightposition=dict(x=pixx, y=pixy, z=pixz*2)
            ))
    if not traces: return None, final_grid
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode='data', xaxis=dict(range=[0, pixx]), yaxis=dict(range=[0, pixy]), zaxis=dict(range=[0, pixz])),
        margin=dict(l=0, r=0, b=0, t=0), height=450
    )
    return fig, final_grid

# ==========================================
# 3. セッションステート & コールバック
# ==========================================
st.session_state.setdefault("recipe_queue", pd.DataFrame(columns=["Recipe", "Stretch", "Seed", "Status"]))
st.session_state.setdefault("active_preview", None)
st.session_state.setdefault("execution_logs", {})

default_filler_params = {
    'rigidfiber': "length=60:radius=2:mean_dir=0,0,1:kappa=0.0", 
    'flake': "radius=10:thickness=2:mean_dir=0,0,1:kappa=0.0", 
    'sphere': "radius=5", 
    'irregfiber': "length=60:shape=bean:radius=5:ratio=0.5:mean_dir=0,0,1:kappa=0.0",
    'flexfiber': "length=90:radius=2:max_bend_deg=90:max_total_bends=10",
    'agglomerate': "num_fibers=5:length=90:radius=2:max_bend_deg=90:max_total_bends=10",
    'staggered': "radius=15:layer_thickness=2:min_layers=1:max_layers=4:max_offset_pct=30:mean_dir=0,0,1:kappa=0.0"
}
st.session_state.setdefault("filler_params_dict", default_filler_params)

if "sweep_f_type" not in st.session_state:
    st.session_state.sweep_f_type = "rigidfiber"
if "sweep_params" not in st.session_state:
    st.session_state.sweep_params = st.session_state.filler_params_dict["rigidfiber"]

def on_filler_type_change():
    f_type = st.session_state.sweep_f_type
    st.session_state.sweep_params = st.session_state.filler_params_dict.get(f_type, "")

def on_params_change():
    f_type = st.session_state.sweep_f_type
    st.session_state.filler_params_dict[f_type] = st.session_state.sweep_params

def generate_and_add_sweep():
    f_type = st.session_state.sweep_f_type
    params = st.session_state.sweep_params
    vfs = np.linspace(st.session_state.sweep_vf_min, st.session_state.sweep_vf_max, st.session_state.sweep_vf_steps)
    stretches = np.linspace(st.session_state.sweep_s_min, st.session_state.sweep_s_max, st.session_state.sweep_s_steps)
    seed_start = st.session_state.sweep_seed_start
    seed_count = st.session_state.sweep_seed_count
    
    new_rows = []
    for seed_offset in range(seed_count):
        current_seed = seed_start + seed_offset if seed_start >= 0 else -1
        for s in stretches:
            for v in vfs:
                new_rows.append({
                    "Recipe": f"{f_type}:{v:.4f}:{params}", 
                    "Stretch": s,
                    "Seed": current_seed,
                    "Status": "⏳ Pending"
                })
                
    new_df = pd.DataFrame(new_rows)
    st.session_state.recipe_queue = pd.concat([st.session_state.recipe_queue, new_df], ignore_index=True)

def clear_queue():
    st.session_state.recipe_queue = pd.DataFrame(columns=["Recipe", "Stretch", "Seed", "Status"])
    st.session_state.active_preview = None
    st.session_state.execution_logs = {}

def clear_log():
    st.session_state.execution_logs = {}

# ==========================================
# 4. サイドバー: Global Settings
# ==========================================
with st.sidebar:
    st.markdown("### :material/settings: Global settings")
    with st.container(border=True):
        c_sb1, c_sb2 = st.columns(2)
        grid_size = c_sb1.number_input("Grid size", value=100, step=10)
        voxel_size = c_sb2.number_input("Voxel size (m)", value=1e-8, format="%.1e")
        bg_type = st.segmented_control("Background type", ['single', 'gyroid', 'lamellar', 'cylinder', 'bcc', 'sea_island', 'island_sea'], default='gyroid')
        phaseA_ratio = st.slider("Background ratio (A)", 0.1, 0.9, 0.5)

    st.space("small")
    st.markdown("### :material/science: Solver & Physics")
    with st.container(border=True):
        physics_mode = st.segmented_control("Physics mode", ['thermal', 'electrical', 'mechanics', 'permeability'], default='thermal')
        solver = st.segmented_control("Solver execution", ['skip', 'chfem', 'puma', 'both'], default='both')

        with st.expander("Material properties", icon=":material/tune:"):
            if physics_mode == 'mechanics': p_a, p_b, p_pri, p_sec = "1.0 0.35", "1.0 0.35", "100.0 0.25", "10.0 0.30"
            elif physics_mode == 'electrical': p_a, p_b, p_pri, p_sec = "1e-4", "1e-4", "1e-1", "1e-3"
            else: p_a, p_b, p_pri, p_sec = "0.3", "0.3", "30.0", "3.0"
            prop_A = st.text_input("Prop A", p_a)
            prop_B = st.text_input("Prop B", p_b)
            prop_pri = st.text_input("Prop primary interface", p_pri)
            prop_sec = st.text_input("Prop secondary interface", p_sec)
            
            void_phases_opt = ""
            if physics_mode == 'permeability':
                void_phases_opt = st.text_input("Void phases (comma-separated)", value="0")

    st.space("small")
    with st.expander("Advanced Export & Metrics", icon=":material/save:"):
        writer_opt = st.selectbox("Export format (Writer)", ["vti", "zstd", "arrow"], index=0)
        vti_fields_opt = st.radio("Embed physical fields to VTI", ["off", "on"], index=0, horizontal=True)
        skip_structure_metrics = st.toggle("Skip structure metrics calculation", value=False)
        advanced_metrics = st.toggle("Calculate advanced PoreSpy metrics", value=False)

    with st.expander("Deformation & Microstructure", icon=":material/transform:"):
        poisson_ratio = st.number_input("Poisson's ratio", value=0.4, step=0.1)
        deformation_mode = st.radio("Deformation mode", ["fine", "coarse"], index=0, horizontal=True)
        c_mic1, c_mic2 = st.columns(2)
        tunnel_radius = c_mic1.number_input("Tunnel radius", value=2, step=1)
        contact_radius = c_mic2.number_input("Contact radius", value=1, step=1)
        fine_volume_tol = st.number_input("Fine mode volume tolerance", value=0.01, format="%.3f")
        fine_max_tilt_deg = st.number_input("Fine mode max tilt (deg)", value=0.10, format="%.2f")
        fine_ledger_cap = st.number_input("Fine mode ledger cap", value=0.01, format="%.3f")

    st.space("small")
    csv_file_path = st.text_input("CSV log path", value="comparison_results.csv", icon=":material/description:")
    base_n = st.text_input("Export basename", value="model", icon=":material/drive_file_rename_outline:")

# ==========================================
# 5. メインコンテンツ (SPAルーター)
# ==========================================
st.markdown("# :material/layers: microsimflow Builder")
st.caption("Design 3D microstructures and manage multi-property simulation sweeps.")

# タブの代わりに確実なルーティング機能を持つSegmented Controlを採用
active_tab = st.segmented_control(
    "Navigation",
    [
        ":material/construction: 1. Recipe & Sweep", 
        ":material/visibility: 2. Exec & Preview", 
        ":material/dashboard: 3. Dashboard"
    ],
    default=":material/construction: 1. Recipe & Sweep",
    label_visibility="collapsed"
)

# ------------------------------------------
# Tab 1: レシピ設計 & スイープジェネレータ
# ------------------------------------------
if active_tab == ":material/construction: 1. Recipe & Sweep":
    with st.container(border=True):
        st.markdown("#### :material/tune: Configure Sweep Parameters")
        
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            st.selectbox(
                "Filler type", 
                list(st.session_state.filler_params_dict.keys()), 
                key="sweep_f_type", 
                on_change=on_filler_type_change
            )
        with col_f2:
            st.text_input(
                "Parameters", 
                key="sweep_params", 
                on_change=on_params_change
            )
        
        col_sweep1, col_sweep2, col_sweep3 = st.columns(3)
        with col_sweep1:
            st.markdown("**Volume Fraction (Vf)**")
            c1, c2, c3 = st.columns(3)
            c1.number_input("Min", value=0.05, step=0.01, key="sweep_vf_min", format="%.3f")
            c2.number_input("Max", value=0.05, step=0.01, key="sweep_vf_max", format="%.3f")
            c3.number_input("Steps", value=1, step=1, min_value=1, key="sweep_vf_steps")
        with col_sweep2:
            st.markdown("**Stretch Ratio (λ)**")
            c4, c5, c6 = st.columns(3)
            c4.number_input("Min", value=1.0, step=0.1, key="sweep_s_min", format="%.2f")
            c5.number_input("Max", value=1.0, step=0.1, key="sweep_s_max", format="%.2f")
            c6.number_input("Steps", value=1, step=1, min_value=1, key="sweep_s_steps")
        with col_sweep3:
            st.markdown("**Random Seed**")
            c7, c8 = st.columns(2)
            c7.number_input("Start", value=-1, step=1, key="sweep_seed_start", help="-1 implies random")
            c8.number_input("Count", value=1, step=1, min_value=1, key="sweep_seed_count", help="Ensemble variations per param set")

        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button("Generate & Append to Queue", type="primary", on_click=generate_and_add_sweep, icon=":material/add_circle:")

    st.space("small")
    st.markdown("#### :material/format_list_bulleted: Simulation Queue")
    st.button("Clear Queue", icon=":material/delete_sweep:", on_click=clear_queue)

    st.data_editor(
        st.session_state.recipe_queue,
        key="queue_editor",
        num_rows="dynamic",
        hide_index=True,
        disabled=["Status"],
        column_config={
            "Recipe": st.column_config.TextColumn("Microstructure Recipe", width="large"), 
            "Stretch": st.column_config.NumberColumn("Stretch (λ)", format="%.2f", width="small"),
            "Seed": st.column_config.NumberColumn("Seed", format="%d", width="small", help="-1 = Random"),
            "Status": st.column_config.TextColumn("Status", width="small")
        },
        height=300
    )

# ------------------------------------------
# Tab 2: 実行 & 3D/2D プレビュー
# ------------------------------------------
elif active_tab == ":material/visibility: 2. Exec & Preview":
    if st.session_state.recipe_queue.empty:
        st.info("Please add recipes in Tab 1 first.", icon=":material/info:")
    else:
        with st.container(border=True):
            st.markdown("### :material/play_circle: Pipeline Execution Control")
            
            options = [f"[{i}] {r.Recipe} (L={r.Stretch:.2f}, Seed={r.Seed})" for i, r in st.session_state.recipe_queue.iterrows()]
            selected_label = st.selectbox("Target step to preview or execute", options)
            target_idx = options.index(selected_label)
            
            def run_cli_for_step(r_str, s_val, seed_val, target_basename):
                cmd = [
                    "python3", "-m", "run_pipeline",
                    "--size", str(grid_size), "--voxel_size", str(voxel_size),
                    "--bg_type", bg_type, "--phaseA_ratio", str(phaseA_ratio),
                    "--physics_mode", physics_mode, "--solver", solver,
                    "--basename", target_basename, "--csv_log", csv_file_path,
                    "--writer", writer_opt, "--vti_fields", vti_fields_opt,
                    "--prop_A", prop_A, "--prop_B", prop_B, 
                    "--prop_inter2", prop_sec, "--prop_inter", prop_pri,
                    "--tunnel_radius", str(tunnel_radius), "--contact_radius", str(contact_radius),
                    "--poisson_ratio", str(poisson_ratio), "--deformation_mode", deformation_mode,
                    "--fine_volume_tol", str(fine_volume_tol), "--fine_max_tilt_deg", str(fine_max_tilt_deg),
                    "--fine_ledger_cap", str(fine_ledger_cap), "--stretch_ratios", str(s_val)
                ]
                if seed_val >= 0: cmd.extend(["--seed", str(seed_val)])
                if void_phases_opt: cmd.extend(["--void_phases"] + [x.strip() for x in void_phases_opt.split(",")])
                if skip_structure_metrics: cmd.append("--skip_structure_metrics")
                if advanced_metrics: cmd.append("--advanced_metrics")
                cmd.extend(["--recipe"] + r_str.split())
                
                run_env = os.environ.copy()
                run_env["PYTHONUNBUFFERED"] = "1"
                return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)

            @st.fragment
            def execution_block():
                queue_placeholder = st.empty()
                log_placeholder = st.empty()
                
                with st.container(horizontal=True):
                    if st.button("👁️‍🗨️ Stage preview", type="secondary"):
                        st.session_state.active_preview = selected_label
                        st.rerun()

                    if st.button("▶ Run selected step", type="primary", icon=":material/play_arrow:"):
                        row = st.session_state.recipe_queue.iloc[target_idx]
                        hash_str = hashlib.md5(f"{row['Recipe']}_{row['Stretch']}_{row['Seed']}_{grid_size}".encode()).hexdigest()[:8]
                        b_name = os.path.join(".gui_preview_cache", f"{base_n}_{hash_str}")
                        
                        st.session_state.recipe_queue.at[target_idx, "Status"] = "🚀 Running"
                        queue_placeholder.dataframe(st.session_state.recipe_queue, hide_index=True)
                        
                        res = run_cli_for_step(row['Recipe'], row['Stretch'], row['Seed'], b_name)
                        st.session_state.execution_logs[target_idx] = clean_console_log(res.stdout)
                        
                        if res.returncode == 0:
                            st.session_state.recipe_queue.at[target_idx, "Status"] = "✅ Completed"
                        else:
                            st.session_state.recipe_queue.at[target_idx, "Status"] = "❌ Error"
                        
                        queue_placeholder.empty()
                        st.rerun()

                    if st.button("⏭ Run all batch steps", type="secondary", icon=":material/fast_forward:"):
                        for i, row in st.session_state.recipe_queue.iterrows():
                            if "Completed" in row["Status"]: continue
                                
                            st.session_state.recipe_queue.at[i, "Status"] = "🚀 Running"
                            queue_placeholder.dataframe(st.session_state.recipe_queue, hide_index=True)
                            
                            h = hashlib.md5(f"{row['Recipe']}_{row['Stretch']}_{row['Seed']}_{grid_size}".encode()).hexdigest()[:8]
                            b_name = os.path.join(".gui_preview_cache", f"{base_n}_{h}")
                            res = run_cli_for_step(row['Recipe'], row['Stretch'], row['Seed'], b_name)
                            
                            st.session_state.execution_logs[i] = clean_console_log(res.stdout)
                            if res.returncode == 0:
                                st.session_state.recipe_queue.at[i, "Status"] = "✅ Completed"
                            else:
                                st.session_state.recipe_queue.at[i, "Status"] = "❌ Error"
                                break
                                
                        queue_placeholder.empty()
                        st.rerun()
                        
                    if target_idx in st.session_state.execution_logs:
                        st.button("Clear Log", icon=":material/clear_all:", on_click=clear_log)

                if target_idx in st.session_state.execution_logs:
                    with log_placeholder.expander(f"Output Log for Step [{target_idx}]", expanded=True, icon=":material/terminal:"):
                        st.code(st.session_state.execution_logs[target_idx])

            execution_block()

        st.space("medium")
        
        @st.fragment
        def preview_block():
            if st.session_state.active_preview:
                try:
                    active_idx = options.index(st.session_state.active_preview)
                    row = st.session_state.recipe_queue.iloc[active_idx]
                except ValueError:
                    st.warning("Selected preview step is no longer in the queue. Please re-stage.")
                    return

                s_val = row['Stretch']
                h_str = hashlib.md5(f"{row['Recipe']}_{s_val}_{row['Seed']}_{grid_size}".encode()).hexdigest()[:8]
                raw_f = os.path.join(".gui_preview_cache", f"{base_n}_{h_str}_L{s_val:.2f}.raw")
                
                with st.container(border=True):
                    st.markdown(f"### :material/analytics: Visualization: {st.session_state.active_preview}")
                    
                    if not os.path.exists(raw_f):
                        st.info("No geometry binary found for this step. Please run the step first.", icon=":material/info:")
                    else:
                        lam_nu = s_val ** (-poisson_ratio)
                        nz, ny, nx = max(1, int(round(grid_size * lam_nu))), max(1, int(round(grid_size * lam_nu))), max(1, int(round(grid_size * s_val)))
                        
                        try:
                            with st.spinner("Computing 3D isosurfaces..."):
                                fig_3d, final_grid = load_and_generate_3d_figure(raw_path=raw_f, nz=nz, ny=ny, nx=nx)
                            
                            col_3d, col_2d = st.columns([1.2, 1])
                            with col_3d:
                                st.markdown("**3D Isosurface Preview**")
                                if fig_3d: st.plotly_chart(fig_3d)
                                else: st.warning("No continuous filler phases detected to render.")

                            with col_2d:
                                st.markdown("**2D Voxel Slice Explorer**")
                                z_slice = st.slider("Z-Axis Slice Selector", 0, nz-1, nz//2)
                                c_scale = [
                                    [0, "#1f2937"], [0.2, "#1f2937"], 
                                    [0.2, "#4b5563"], [0.4, "#4b5563"], 
                                    [0.4, "#f59e0b"], [0.8, "#f59e0b"], 
                                    [0.8, "#3b82f6"], [1, "#3b82f6"]
                                ]
                                fig_2d = px.imshow(final_grid[z_slice, :, :], color_continuous_scale=c_scale, origin='lower')
                                fig_2d.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0), height=450)
                                st.plotly_chart(fig_2d)
                        except ValueError:
                            st.error("Voxel grid dimensions mismatch. If you modified Grid Size, please re-run the pipeline.")

        preview_block()

# ------------------------------------------
# Tab 3: HTML ダッシュボード
# ------------------------------------------
elif active_tab == ":material/dashboard: 3. Dashboard":
    with st.container(border=True):
        st.markdown("### :material/insert_chart: Results Dashboard Generator")
        st.caption("Compile your central simulation log into an interactive visual HTML dashboard report.")
        
        html_out = csv_file_path.replace('.csv', '.html')
        
        if st.button("🔄 Generate / Refresh dashboard", type="primary", icon=":material/refresh:"):
            if os.path.exists(csv_file_path):
                with st.spinner("Rendering HTML report components..."):
                    res = subprocess.run(["python3", "render_results_dashboard.py", "--csv", csv_file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    st.success("Dashboard static build complete!", icon=":material/check_circle:")
                    with st.expander("Renderer output log", icon=":material/terminal:"): 
                        st.code(clean_console_log(res.stdout))
            else: 
                st.error("Target CSV log file not found. Ensure you have executed at least one solver step.")
            
        if os.path.exists(html_out):
            st.space("small")
            with open(html_out, "rb") as f:
                st.download_button(
                    label="🌐 Download HTML dashboard report", 
                    data=f.read(), 
                    file_name=os.path.basename(html_out), 
                    mime="text/html",
                    icon=":material/download:"
                )
            st.caption("Tip: Open the downloaded HTML file in a new browser tab to access full interactive features.")
