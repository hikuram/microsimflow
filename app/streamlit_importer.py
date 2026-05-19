import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
import json
import subprocess
from skimage.measure import marching_cubes
from PIL import Image

# ==========================================
# 1. ページ設定
# ==========================================
st.set_page_config(
    page_title="microsimflow Importer", 
    page_icon=":material/image_search:", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 共通ユーティリティ (ログ＆キャッシュ)
# ==========================================
def clean_console_log(raw_bytes):
    """バイナリ出力からキャリッジリターンを処理し、綺麗なテキストに変換"""
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
def scan_directory(directory):
    if not os.path.exists(directory) or not os.path.isdir(directory): return pd.DataFrame()
    model_data = []
    for raw_path in glob.glob(os.path.join(directory, "*.raw")):
        base_stem = raw_path[:-4]
        nf_path = base_stem + ".nf"
        if not os.path.exists(nf_path) and base_stem.endswith("_final"):
            alt_nf = base_stem[:-6] + "_meta.nf"
            if os.path.exists(alt_nf): nf_path = alt_nf
        has_nf = os.path.exists(nf_path)
        grid_str, voxel_str = "Unknown", "Unknown"
        if has_nf:
            try:
                with open(nf_path, 'r') as f: meta = json.load(f)
                grid_str, voxel_str = str(meta.get("grid_size", grid_str)), str(meta.get("voxel_size_m", voxel_str))
            except Exception: pass
        model_data.append({
            "Model Name": os.path.basename(raw_path), "Meta": "✅" if has_nf else "❌",
            "Grid": grid_str, "Voxel": voxel_str, "Raw Path": raw_path, "Nf Path": nf_path if has_nf else None
        })
    return pd.DataFrame(model_data).sort_values("Model Name").reset_index(drop=True) if model_data else pd.DataFrame()

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
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.4), lightposition=dict(x=pixx, y=pixy, z=pixz*2)
            ))
    if not traces: return None, final_grid
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode='data', xaxis=dict(range=[0, pixx]), yaxis=dict(range=[0, pixy]), zaxis=dict(range=[0, pixz])),
        margin=dict(l=0, r=0, b=0, t=0), height=450
    )
    return fig, final_grid

# ==========================================
# 3. セッションステート初期化
# ==========================================
st.session_state.setdefault("importer_active_preview", None)
st.session_state.setdefault("import_log", None)
st.session_state.setdefault("import_status", None)
st.session_state.setdefault("solver_log", None)
st.session_state.setdefault("solver_status", None)

def clear_import_log():
    st.session_state.import_log = None
    st.session_state.import_status = None

def clear_solver_log():
    st.session_state.solver_log = None
    st.session_state.solver_status = None

# ==========================================
# 4. サイドバー: Global Settings
# ==========================================
with st.sidebar:
    st.markdown("### :material/folder_open: Paths & Settings")
    with st.container(border=True):
        target_dir = st.text_input("Model Directory", value="imported_models")
        csv_file_path = st.text_input("CSV Log Path", value="imported_results.csv")
        fallback_grid = st.number_input("Fallback Grid Size", value=100, step=10, help="メタデータ(.nf)が無い場合のプレビュー用サイズ")

    st.space("small")
    st.markdown("### :material/science: Solver Execution")
    with st.container(border=True):
        physics_mode = st.segmented_control("Physics Mode", ['thermal', 'electrical', 'mechanics', 'permeability'], default='thermal')
        solver = st.segmented_control("Solver", ['skip', 'chfem', 'puma', 'both'], default='chfem')

# ==========================================
# 5. メインコンテンツ (SPAルーター)
# ==========================================
st.markdown("# :material/image_search: microsimflow Importer")
st.caption("Import, binarize, and analyze real micro-CT/FIB images.")

# SPAルーティング
active_tab = st.segmented_control(
    "Navigation",
    [
        ":material/add_photo_alternate: 1. Image Importer", 
        ":material/visibility: 2. Viewer & Solver", 
        ":material/dashboard: 3. Dashboard"
    ],
    default=":material/add_photo_alternate: 1. Image Importer",
    label_visibility="collapsed"
)

# ------------------------------------------
# Tab 1: 実画像インポーター
# ------------------------------------------
if active_tab == ":material/add_photo_alternate: 1. Image Importer":
    c_in1, c_in2 = st.columns([1, 1])
    
    with c_in1:
        with st.container(border=True):
            st.markdown("#### :material/input: 1. Input Image & Setup")
            input_method = st.segmented_control("Input Source", ["Server Path", "Upload File"], default="Server Path")
            
            file_path = ""
            if input_method == "Server Path":
                file_path = st.text_input("File Path:", value="./sample.tif")
            else:
                uploaded_file = st.file_uploader("TIFF/RAW", type=["tif", "tiff", "raw", "png"])
                if uploaded_file:
                    os.makedirs("temp_uploads", exist_ok=True)
                    file_path = os.path.join("temp_uploads", uploaded_file.name)
                    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            c_fmt1, c_fmt2 = st.columns(2)
            file_format = c_fmt1.selectbox("Format", ["tiff", "raw"])
            raw_shape = c_fmt2.text_input("RAW Shape (Z Y X)", "100 100 100") if file_format == "raw" else ""
            voxel_size = st.number_input("Voxel Size (m)", 1e-8, format="%.2e")
        
        st.space("small")
        with st.container(border=True):
            st.markdown("#### :material/imagesearch_roller: 2. Processing Rules")
            threshold = st.slider("Binarization Threshold", 0, 255, 128)
            c_p1, c_p2 = st.columns(2)
            pattern = c_p1.selectbox("Interface Pattern", ["dilation", "erosion"])
            tunnel_radius = c_p2.number_input("Tunnel Radius", min_value=1, value=2)

    with c_in2:
        with st.container(border=True):
            st.markdown("#### :material/preview: Preview & Execute")
            
            # 2Dプレビュー
            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    img = np.array(Image.open(file_path).convert("L"))
                    fig_prev = px.imshow(img >= threshold, color_continuous_scale="gray")
                    fig_prev.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0), height=300)
                    st.plotly_chart(fig_prev)
                except Exception:
                    st.info("No valid 2D preview available for this file type.")
            
            @st.fragment
            def import_execution_block():
                with st.container(horizontal=True):
                    if st.button("🚀 Extract & Import Model", type="primary"):
                        if not os.path.exists(file_path): 
                            st.session_state.import_status = "error"
                            st.session_state.import_log = "Error: Input file not found."
                        else:
                            with st.spinner("Processing image..."):
                                cmd = [
                                    "python3", "import_image.py", 
                                    "--input", file_path, "--format", file_format, 
                                    "--voxel_size", str(voxel_size), "--threshold", str(threshold), 
                                    "--pattern", pattern, "--tunnel_radius", str(tunnel_radius), 
                                    "--out_dir", target_dir
                                ]
                                if file_format == "raw": cmd.extend(["--raw_shape"] + raw_shape.split())
                                
                                run_env = os.environ.copy()
                                run_env["PYTHONUNBUFFERED"] = "1"
                                try:
                                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
                                    st.session_state.import_log = clean_console_log(res.stdout)
                                    st.session_state.import_status = "success" if res.returncode == 0 else "error"
                                except Exception as e:
                                    st.session_state.import_log = f"Execution failed: {e}"
                                    st.session_state.import_status = "error"
                                    
                        st.rerun() # リランで描画を更新し、キャッシュスキャンを誘発
                        
                    if st.session_state.import_log is not None:
                        st.button("Clear Log", icon=":material/clear_all:", on_click=clear_import_log)

                # ログの永続的表示
                if st.session_state.import_log is not None:
                    if st.session_state.import_status == "success":
                        st.success("Import completed successfully!", icon=":material/check_circle:")
                        # インポート成功時にスキャンキャッシュをクリア
                        st.cache_data.clear()
                    else:
                        st.error("Import failed.", icon=":material/error:")
                    
                    with st.expander("Import Log", expanded=True, icon=":material/terminal:"): 
                        st.code(st.session_state.import_log)

            import_execution_block()

# ------------------------------------------
# Tab 2: Viewer & Solver
# ------------------------------------------
elif active_tab == ":material/visibility: 2. Viewer & Solver":
    df_models = scan_directory(target_dir)
    
    if df_models.empty:
        st.info("No models found in the target directory.", icon=":material/info:")
    else:
        with st.container(border=True):
            st.markdown("#### :material/folder_managed: Model Explorer")
            st.dataframe(df_models[["Model Name", "Meta", "Grid", "Voxel"]], height=200, hide_index=True)
            
            c_sel1, c_sel2 = st.columns([3, 1])
            with c_sel1:
                selected_model = st.selectbox("Select Target Model", df_models["Model Name"].tolist())
            
            target_row = df_models[df_models["Model Name"] == selected_model].iloc[0]
            raw_path = target_row["Raw Path"]
            base_path = raw_path[:-10] # _final.raw を削る
            
            @st.fragment
            def solver_execution_block():
                log_placeholder = st.empty()
                with st.container(horizontal=True):
                    if st.button("👁️‍🗨️ Stage 3D Preview", type="secondary"):
                        st.session_state.importer_active_preview = selected_model
                        st.rerun()
                        
                    if st.button("▶️ Run Solver", type="primary", icon=":material/play_arrow:"):
                        with st.spinner(f"Running {solver} solver..."):
                            cmd = [
                                "python3", "run_imported.py",
                                "--import_path", base_path,
                                "--solver", solver,
                                "--physics_mode", physics_mode,
                                "--csv_log", csv_file_path
                            ]
                            run_env = os.environ.copy()
                            run_env["PYTHONUNBUFFERED"] = "1"
                            try:
                                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
                                st.session_state.solver_log = clean_console_log(res.stdout)
                                st.session_state.solver_status = "success" if res.returncode == 0 else "error"
                            except Exception as e:
                                st.session_state.solver_log = f"Execution failed: {e}"
                                st.session_state.solver_status = "error"
                        st.rerun()

                    if st.session_state.solver_log is not None:
                        st.button("Clear Log", icon=":material/clear_all:", on_click=clear_solver_log)

                if st.session_state.solver_log is not None:
                    if st.session_state.solver_status == "success":
                        st.success("Analysis completed!", icon=":material/check_circle:")
                    else:
                        st.error("Solver execution error.", icon=":material/error:")
                        
                    with log_placeholder.expander("Solver Log", expanded=True, icon=":material/terminal:"):
                        st.code(st.session_state.solver_log)

            solver_execution_block()

        st.space("medium")
        
        @st.fragment
        def preview_block():
            if st.session_state.importer_active_preview == selected_model:
                with st.container(border=True):
                    st.markdown(f"### :material/analytics: 3D Microstructure: {selected_model}")
                    
                    nz, ny, nx = fallback_grid, fallback_grid, fallback_grid
                    if target_row["Nf Path"]:
                        try:
                            with open(target_row["Nf Path"], 'r') as f:
                                meta = json.load(f)
                                if "grid_size" in meta and len(meta["grid_size"]) == 3: nz, ny, nx = meta["grid_size"]
                        except Exception: pass
                    
                    with st.spinner("Computing 3D isosurfaces..."):
                        fig_3d, final_grid = load_and_generate_3d_figure(raw_path, nz, ny, nx)

                    if final_grid is not None:
                        col_3d, col_2d = st.columns([1.2, 1])
                        
                        with col_3d:
                            st.markdown("**3D Isosurface**")
                            if fig_3d: st.plotly_chart(fig_3d)
                            else: st.warning("フィラーがありません。")

                        with col_2d:
                            st.markdown("**2D Full Slice**")
                            z_slice = st.slider("Z-Axis", 0, nz-1, nz//2, label_visibility="collapsed")
                            c_scale = [[0, "#1f2937"], [0.2, "#1f2937"], [0.2, "#4b5563"], [0.4, "#4b5563"], [0.4, "#f59e0b"], [0.8, "#f59e0b"], [0.8, "#3b82f6"], [1, "#3b82f6"]]
                            fig_2d = px.imshow(final_grid[z_slice, :, :], color_continuous_scale=c_scale, origin='lower')
                            fig_2d.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0), height=450)
                            st.plotly_chart(fig_2d)

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
