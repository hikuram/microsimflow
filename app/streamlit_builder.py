import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import subprocess
import hashlib
from skimage.measure import marching_cubes

st.set_page_config(page_title="microsimflow Builder", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1, h2, h3 { margin-top: 0.5rem; margin-bottom: 0.5rem; }
    .stExpander { border-color: rgba(128, 128, 128, 0.2); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 0. 共通ユーティリティ (ログ＆キャッシュ)
# ==========================================
def clean_console_log(raw_bytes):
    """バイナリ出力から \r (キャリッジリターン) による行上書きを再現し、ログを綺麗に圧縮する"""
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
    """3D等値面の計算をキャッシュし、スライダー操作時の再計算ラグを防ぐ"""
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

# セッションステート
if "step_df" not in st.session_state:
    st.session_state.step_df = pd.DataFrame(columns=["Recipe", "Stretch"])
if "active_preview" not in st.session_state:
    st.session_state.active_preview = None

# ==========================================
# サイドバー: Global Settings
# ==========================================
st.sidebar.markdown("### ⚙️ Global Settings")
c_sb1, c_sb2 = st.sidebar.columns(2)
grid_size = c_sb1.number_input("Grid Size", value=100, step=10)
voxel_size = c_sb2.number_input("Voxel", value=1e-8, format="%.1e")

bg_type = st.sidebar.selectbox("BG Type", ['single', 'gyroid', 'lamellar', 'cylinder', 'bcc', 'sea_island', 'island_sea'])
phaseA_ratio = st.sidebar.slider("BG Ratio (A)", 0.1, 0.9, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Solver & Physics")
physics_mode = st.sidebar.selectbox("Physics Mode", ['thermal', 'electrical', 'mechanics', 'permeability'])
solver = st.sidebar.selectbox("Solver Execute", ['skip', 'chfem', 'puma', 'both'], index=0, help="skip以外を選ぶと、計算が実行され時間がかかります。")

with st.sidebar.expander("Material Properties", expanded=False):
    if physics_mode == 'mechanics': p_a, p_b, p_pri, p_sec = "1.0 0.35", "1.0 0.35", "100.0 0.25", "10.0 0.30"
    elif physics_mode == 'electrical': p_a, p_b, p_pri, p_sec = "1e-4", "1e-4", "1e-1", "1e-3"
    else: p_a, p_b, p_pri, p_sec = "0.3", "0.3", "30.0", "3.0"
    st.text_input("Prop A", p_a)
    st.text_input("Prop B", p_b)
    st.text_input("Prop Pri. Inter", p_pri)
    st.text_input("Prop Sec. Inter", p_sec)

st.sidebar.markdown("---")
csv_file_path = st.sidebar.text_input("CSV Log Path", value="comparison_results.csv")

# ==========================================
# メインタブ構成
# ==========================================
st.title("microsimflow Builder")
tab1, tab2, tab3 = st.tabs(["🏗️ 1. Recipe & Sweep", "👁️‍🗨️ 2. Exec & Preview", "📊 3. Dashboard"])

# ------------------------------------------
# Tab 1: レシピ設計 & スイープジェネレータ
# ------------------------------------------
with tab1:
    with st.expander("➕ Parameter Sweep Generator", expanded=True):
        c1, c2 = st.columns([1, 2])
        f_type = c1.selectbox("Filler Type", ['rigidfiber', 'flake', 'sphere', 'irregfiber', 'flexfiber', 'agglomerate', 'staggered'])
        default_params = {'rigidfiber': "length=60:radius=2:mean_dir=0,0,1:kappa=0.0", 'flake': "radius=10:thickness=2:mean_dir=0,0,1:kappa=0.0", 'sphere': "radius=5", 'irregfiber': "length=60:shape=bean:radius=5:ratio=0.5:mean_dir=0,0,1:kappa=0.0"}
        params = c2.text_input("Parameters", value=default_params.get(f_type, ""))

        c3, c4, c5, c6, c7, c8 = st.columns(6)
        vf_min = c3.number_input("Vf Min", value=0.05, step=0.01)
        vf_max = c4.number_input("Vf Max", value=0.05, step=0.01)
        vf_steps = c5.number_input("Vf Steps", value=1, step=1, min_value=1)
        s_min = c6.number_input("Stretch Min", value=1.0, step=0.1)
        s_max = c7.number_input("Stretch Max", value=1.0, step=0.1)
        s_steps = c8.number_input("Str. Steps", value=1, step=1, min_value=1)

        if st.button("🔽 スイープ展開してリストに追加", type="primary"):
            vfs, stretches, new_rows = np.linspace(vf_min, vf_max, vf_steps), np.linspace(s_min, s_max, s_steps), []
            for s in stretches:
                for v in vfs: new_rows.append({"Recipe": f"{f_type}:{v:.4f}:{params}", "Stretch": s})
            st.session_state.step_df = pd.concat([st.session_state.step_df, pd.DataFrame(new_rows)], ignore_index=True)
            st.rerun()

    st.markdown("#### Execution Steps")
    st.session_state.step_df = st.data_editor(
        st.session_state.step_df, num_rows="dynamic", use_container_width=True,
        column_config={"Recipe": st.column_config.TextColumn("Recipe String", width="large"), "Stretch": st.column_config.NumberColumn("Stretch Ratio", format="%.2f")},
        height=250
    )
    if st.button("🗑️ リストを全クリア"):
        st.session_state.step_df = pd.DataFrame(columns=["Recipe", "Stretch"])
        st.session_state.active_preview = None
        st.rerun()

# ------------------------------------------
# Tab 2: 実行 & 3D/2D プレビュー
# ------------------------------------------
with tab2:
    if st.session_state.step_df.empty:
        st.info("Tab 1 でレシピを追加してください。")
    else:
        c_run1, c_run2, c_run3, c_run4 = st.columns([3, 1, 1, 1])
        options = [f"[{i}] {r.Recipe} (L={r.Stretch:.2f})" for i, r in st.session_state.step_df.iterrows()]
        selected_label = c_run1.selectbox("Preview & Execute Target", options, label_visibility="collapsed")
        target_idx = options.index(selected_label)
        target_row = st.session_state.step_df.iloc[target_idx]
        
        recipe_str, stretch_val = target_row['Recipe'], target_row['Stretch']
        hash_str = hashlib.md5(f"{recipe_str}_{stretch_val}_{grid_size}".encode()).hexdigest()[:8]
        cache_dir = ".gui_preview_cache"
        os.makedirs(cache_dir, exist_ok=True)
        basename = os.path.join(cache_dir, f"model_{hash_str}")
        raw_file = f"{basename}_L{stretch_val:.2f}.raw"

        def run_cli_for_step(r_str, s_val, base_n):
            cmd = ["python3", "-m", "run_pipeline", "--size", str(grid_size), "--voxel_size", str(voxel_size), "--bg_type", bg_type, "--phaseA_ratio", str(phaseA_ratio), "--physics_mode", physics_mode, "--solver", solver, "--basename", base_n, "--csv_log", csv_file_path, "--stretch_ratios", str(s_val)] + ["--recipe"] + r_str.split()
            # ★ 変更: stderrをstdoutにマージし、バイナリで受け取る
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # プレビューボタン (状態を保持)
        if c_run2.button("👁️‍🗨️ プレビュー", use_container_width=True):
            st.session_state.active_preview = selected_label

        log_placeholder = st.empty()

        if c_run3.button("▶ 選択実行", type="primary", use_container_width=True):
            with st.spinner("パイプライン実行中..."):
                try:
                    res = run_cli_for_step(recipe_str, stretch_val, basename)
                    if res.returncode == 0: st.success("成功！")
                    else: st.error("エラー発生")
                    with log_placeholder.expander("実行ログ", expanded=True):
                        st.code(clean_console_log(res.stdout))
                except Exception as e: st.error(f"実行失敗: {e}")
                    
        if c_run4.button("⏭ 全ステップ", use_container_width=True):
            with st.spinner(f"全 {len(st.session_state.step_df)} ステップを実行中..."):
                all_logs = ""
                for i, row in st.session_state.step_df.iterrows():
                    h = hashlib.md5(f"{row['Recipe']}_{row['Stretch']}_{grid_size}".encode()).hexdigest()[:8]
                    res = run_cli_for_step(row['Recipe'], row['Stretch'], os.path.join(cache_dir, f"model_{h}"))
                    all_logs += f"--- Step {i} ---\n{clean_console_log(res.stdout)}\n"
                st.success("一括実行完了！")
                with log_placeholder.expander("全ステップ 実行ログ", expanded=False): st.code(all_logs)

        st.markdown("---")
        
        # プレビューエリア (状態保持されている場合のみ描画)
        if st.session_state.active_preview == selected_label:
            if not os.path.exists(raw_file):
                st.info(f"対象のバイナリがありません。実行ボタンを押してください。")
            else:
                poisson_ratio = 0.4
                lam_nu = stretch_val ** (-poisson_ratio)
                nz, ny, nx = max(1, int(round(grid_size * lam_nu))), max(1, int(round(grid_size * lam_nu))), max(1, int(round(grid_size * stretch_val)))
                
                try:
                    with st.spinner("3Dモデル準備中..."):
                        fig_3d, final_grid = load_and_generate_3d_figure(raw_path=raw_file, nz=nz, ny=ny, nx=nx)
                    
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
                except ValueError:
                    st.error("バイナリの形状が一致しません。Grid Sizeを変更した場合は再実行してください。")

# ------------------------------------------
# Tab 3: HTML ダッシュボード
# ------------------------------------------
with tab3:
    st.markdown("### Results Dashboard Generator")
    html_out = csv_file_path.replace('.csv', '.html')
    
    if st.button("🔄 ダッシュボード生成・更新", type="primary"):
        if os.path.exists(csv_file_path):
            with st.spinner("生成中..."):
                res = subprocess.run(["python3", "render_results_dashboard.py", "--csv", csv_file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                st.success("生成完了")
                with st.expander("実行ログ"): st.code(clean_console_log(res.stdout))
        else: st.error("CSVがありません")
            
    if os.path.exists(html_out):
        with open(html_out, "rb") as f:
            st.download_button("🌐 HTMLをダウンロードして開く", f.read(), os.path.basename(html_out), "text/html")
        st.caption("ダウンロードしたHTMLをブラウザの別タブで開くと、フル機能（画像ホバー等）で閲覧できます。")
