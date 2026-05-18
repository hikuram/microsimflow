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

st.set_page_config(page_title="microsimflow Importer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1, h2, h3 { margin-top: 0.5rem; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 0. ターミナルログのクリーニング関数
# ==========================================
def clean_console_log(raw_bytes):
    """
    バイナリで受け取った標準出力からターミナルの \r (行上書き) の挙動をシミュレートし、
    縦に伸びるプログレスバーを最終状態の1行だけに圧縮します。
    """
    if not raw_bytes:
        return ""
    text = raw_bytes.decode('utf-8', errors='replace')
    
    lines = []
    current_line = ""
    for char in text:
        if char == '\n':
            if current_line.strip():
                lines.append(current_line)
            current_line = ""
        elif char == '\r':
            current_line = ""  # 行の先頭に戻り、これまでバッファした文字を捨てる（上書き）
        else:
            current_line += char
            
    if current_line.strip():
        lines.append(current_line)
        
    return '\n'.join(lines)

# ==========================================
# 1. キャッシュされた重い処理
# ==========================================
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

if "active_preview" not in st.session_state:
    st.session_state.active_preview = None

# ==========================================
# サイドバー
# ==========================================
st.sidebar.markdown("### 📁 Paths & Settings")
target_dir = st.sidebar.text_input("Model Directory:", value="imported_models")
csv_file_path = st.sidebar.text_input("CSV Log Path:", value="imported_results.csv")
fallback_grid = st.sidebar.number_input("Fallback Grid Size:", value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Solver Execution")
physics_mode = st.sidebar.selectbox("Physics Mode", ['thermal', 'electrical', 'mechanics', 'permeability'])
solver = st.sidebar.selectbox("Solver", ['skip', 'chfem', 'puma', 'both'], index=1)

# ==========================================
# メインタブ構成
# ==========================================
st.title("microsimflow Image Importer & Analyzer")
tab1, tab2, tab3 = st.tabs(["🔬 1. Image Importer", "👁️‍🗨️ 2. Viewer & Solver", "📊 3. Dashboard"])

# ------------------------------------------
# Tab 1: 実画像インポーター
# ------------------------------------------
with tab1:
    c_in1, c_in2 = st.columns([1, 1])
    with c_in1:
        st.subheader("1. Input Image & Setup")
        input_method = st.radio("入力", ["サーバー内パス", "アップロード"], horizontal=True, label_visibility="collapsed")
        file_path = st.text_input("File Path:", value="./sample.tif") if input_method == "サーバー内パス" else ""
        if input_method == "アップロード":
            uploaded_file = st.file_uploader("TIFF/RAW", type=["tif", "tiff", "raw", "png"])
            if uploaded_file:
                os.makedirs("temp_uploads", exist_ok=True)
                file_path = os.path.join("temp_uploads", uploaded_file.name)
                with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        c_fmt1, c_fmt2 = st.columns(2)
        file_format = c_fmt1.selectbox("Format", ["tiff", "raw"])
        raw_shape = c_fmt2.text_input("RAW Shape (Z Y X)", "100 100 100") if file_format == "raw" else ""
        voxel_size = st.number_input("Voxel Size (m):", 1e-8, format="%.2e")
        
        st.subheader("2. Processing")
        threshold = st.slider("Binarization Threshold", 0, 255, 128)
        c_p1, c_p2 = st.columns(2)
        pattern = c_p1.selectbox("Interface Pattern", ["dilation", "erosion"])
        tunnel_radius = c_p2.number_input("Tunnel Radius", min_value=1, value=2)

    with c_in2:
        st.subheader("Preview & Execute")
        if os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                img = np.array(Image.open(file_path).convert("L"))
                fig_prev = px.imshow(img >= threshold, color_continuous_scale="gray")
                fig_prev.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0), height=300)
                st.plotly_chart(fig_prev, use_container_width=True)
            except Exception: pass

        if st.button("🚀 モデル抽出を実行 (import_image.py)", type="primary"):
            if not os.path.exists(file_path): st.error("ファイルがありません")
            else:
                with st.spinner("実行中..."):
                    cmd = ["python3", "import_image.py", "--input", file_path, "--format", file_format, "--voxel_size", str(voxel_size), "--threshold", str(threshold), "--pattern", pattern, "--tunnel_radius", str(tunnel_radius), "--out_dir", target_dir]
                    if file_format == "raw": cmd.extend(["--raw_shape"] + raw_shape.split())
                    try:
                        # ★ 変更: stderrをstdoutにマージし、バイナリで受け取る
                        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        if res.returncode == 0:
                            st.success("✅ インポート完了！")
                            st.cache_data.clear()
                        else:
                            st.error("❌ エラー発生")
                        with st.expander("実行ログ", expanded=True): 
                            st.code(clean_console_log(res.stdout))
                    except Exception as e:
                        st.error(f"実行失敗: {e}")

# ------------------------------------------
# Tab 2: 3D/2D Viewer & Solver (run_imported.py 連携)
# ------------------------------------------
with tab2:
    df_models = scan_directory(target_dir)
    
    if df_models.empty:
        st.info("ディレクトリにモデルがありません。")
    else:
        st.markdown("#### Model Explorer")
        st.dataframe(df_models[["Model Name", "Meta", "Grid", "Voxel"]], use_container_width=True, height=150)
        
        c_sel1, c_sel2, c_sel3 = st.columns([2, 1, 1])
        selected_model = c_sel1.selectbox("Target Model", df_models["Model Name"].tolist(), label_visibility="collapsed")
        target_row = df_models[df_models["Model Name"] == selected_model].iloc[0]
        
        raw_path = target_row["Raw Path"]
        base_path = raw_path[:-10] # _final.raw を削る
        
        if c_sel2.button("👁️‍🗨️ プレビュー描画", use_container_width=True):
            st.session_state.active_preview = selected_model
            
        if c_sel3.button("▶️ ソルバー実行", type="primary", use_container_width=True):
            with st.spinner(f"{solver} ソルバーを実行中..."):
                cmd = [
                    "python3", "run_imported.py",
                    "--import_path", base_path,
                    "--solver", solver,
                    "--physics_mode", physics_mode,
                    "--csv_log", csv_file_path
                ]
                try:
                    # ★ 変更: stderrをstdoutにマージし、バイナリで受け取る
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    if res.returncode == 0:
                        st.success("✅ 解析完了！")
                    else:
                        st.error("❌ 実行エラー")
                    with st.expander("実行ログ", expanded=True):
                        st.code(clean_console_log(res.stdout))
                except Exception as e:
                    st.error(f"実行失敗: {e}")

        st.markdown("---")
        
        if st.session_state.active_preview == selected_model:
            nz, ny, nx = fallback_grid, fallback_grid, fallback_grid
            if target_row["Nf Path"]:
                try:
                    with open(target_row["Nf Path"], 'r') as f:
                        meta = json.load(f)
                        if "grid_size" in meta and len(meta["grid_size"]) == 3: nz, ny, nx = meta["grid_size"]
                except Exception: pass
            
            with st.spinner("3Dモデル準備中..."):
                fig_3d, final_grid = load_and_generate_3d_figure(raw_path, nz, ny, nx)

            if final_grid is not None:
                col_3d, col_2d = st.columns([1.2, 1])
                
                with col_3d:
                    st.markdown("**3D Isosurface**")
                    if fig_3d: st.plotly_chart(fig_3d, use_container_width=True)
                    else: st.warning("フィラーがありません。")

                with col_2d:
                    st.markdown("**2D Full Slice**")
                    z_slice = st.slider("Z-Axis", 0, nz-1, nz//2, label_visibility="collapsed")
                    c_scale = [[0, "#1f2937"], [0.2, "#1f2937"], [0.2, "#4b5563"], [0.4, "#4b5563"], [0.4, "#f59e0b"], [0.8, "#f59e0b"], [0.8, "#3b82f6"], [1, "#3b82f6"]]
                    fig_2d = px.imshow(final_grid[z_slice, :, :], color_continuous_scale=c_scale, origin='lower')
                    fig_2d.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0), height=450)
                    st.plotly_chart(fig_2d, use_container_width=True)

# ------------------------------------------
# Tab 3: HTML ダッシュボード
# ------------------------------------------
with tab3:
    st.markdown("### Results Dashboard Generator")
    html_out = csv_file_path.replace('.csv', '.html')
    
    if st.button("🔄 生成・更新", type="primary"):
        if os.path.exists(csv_file_path):
            with st.spinner("生成中..."):
                res = subprocess.run(["python3", "render_results_dashboard.py", "--csv", csv_file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                st.success("完了")
                with st.expander("実行ログ"): st.code(clean_console_log(res.stdout))
        else: st.error("CSVがありません")
            
    if os.path.exists(html_out):
        with open(html_out, "rb") as f:
            st.download_button("🌐 HTMLダッシュボードをダウンロード", f.read(), os.path.basename(html_out), "text/html")
