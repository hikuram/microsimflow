from __future__ import annotations

import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from skimage.measure import marching_cubes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PENDING = "Pending"
STATUS_RUNNING = "Running"
STATUS_COMPLETED = "Completed"
STATUS_ERROR = "Error"
CACHE_DIR_NAME = ".gui_preview_cache"
PREVIEW_CACHE_MAX_BYTES = 512 * 1024 * 1024
PREVIEW_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
DOWNLOAD_DIR_NAME = ".gui_downloads"


def resolve_project_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_project_dir(path_like: str | os.PathLike[str]) -> Path:
    path = resolve_project_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_bytes(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def display_project_path(path_like: str | os.PathLike[str]) -> str:
    path = resolve_project_path(path_like)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_directory_zip_file(
    path_like: str | os.PathLike[str],
    output_filename: str,
) -> Path:
    directory = ensure_project_dir(path_like)
    download_dir = ensure_project_dir(DOWNLOAD_DIR_NAME)
    zip_path = download_dir / output_filename
    temp_path = zip_path.with_suffix(".tmp")

    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{directory.name}/", b"")
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            arcname = Path(directory.name) / path.relative_to(directory)
            zf.write(path, arcname.as_posix())

    temp_path.replace(zip_path)
    return zip_path


def render_output_downloads(csv_path: str) -> None:
    csv_file = resolve_project_path(csv_path)
    csv_exists = csv_file.exists() and csv_file.is_file()
    cache_stats = get_preview_cache_stats()

    with st.expander("Downloads", icon=":material/download:"):
        st.caption(f"CSV: {display_project_path(csv_file)}")
        st.download_button(
            "Download CSV log",
            data=csv_file.read_bytes() if csv_exists else b"",
            file_name=csv_file.name,
            mime="text/csv",
            icon=":material/table_view:",
            disabled=not csv_exists,
        )
        if not csv_exists:
            st.caption("CSV log is not available yet.")

        st.caption(
            f"Cache: {cache_stats['path']} - {cache_stats['files']} files / "
            f"{format_bytes(int(cache_stats['bytes']))}"
        )
        if st.button(
            "Prepare cache ZIP",
            type="tertiary",
            icon=":material/archive:",
            width="content",
        ):
            zip_path = write_directory_zip_file(
                CACHE_DIR_NAME,
                f"{CACHE_DIR_NAME}.zip",
            )
            st.session_state.preview_cache_zip_path = str(zip_path)
            st.success(f"Prepared {display_project_path(zip_path)}.")

        zip_path_text = st.session_state.get("preview_cache_zip_path")
        if zip_path_text:
            zip_path = Path(zip_path_text)
            if zip_path.exists():
                st.download_button(
                    "Download prepared cache ZIP",
                    data=zip_path.read_bytes(),
                    file_name=zip_path.name,
                    mime="application/zip",
                    icon=":material/folder_zip:",
                )
                st.caption("Use Prepare cache ZIP again after cache changes.")



def _preview_cache_files() -> list[Path]:
    cache_dir = ensure_project_dir(CACHE_DIR_NAME)
    files: list[Path] = []
    for path in cache_dir.rglob("*"):
        if not path.is_file():
            continue
        files.append(path)
    return files


def get_preview_cache_stats() -> dict[str, int | str]:
    cache_dir = ensure_project_dir(CACHE_DIR_NAME)
    total_size = 0
    files = 0
    for path in _preview_cache_files():
        try:
            total_size += path.stat().st_size
            files += 1
        except OSError:
            continue
    return {
        "path": str(cache_dir.relative_to(PROJECT_ROOT)),
        "files": files,
        "bytes": total_size,
    }


def clear_preview_cache() -> tuple[int, int]:
    removed_files = 0
    removed_bytes = 0
    for path in _preview_cache_files():
        try:
            size = path.stat().st_size
            path.unlink()
            removed_files += 1
            removed_bytes += size
        except OSError:
            continue
    cache_dir = ensure_project_dir(CACHE_DIR_NAME)
    for path in sorted(cache_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return removed_files, removed_bytes


def enforce_preview_cache_limits(
    max_bytes: int = PREVIEW_CACHE_MAX_BYTES,
    max_age_seconds: int = PREVIEW_CACHE_MAX_AGE_SECONDS,
) -> tuple[int, int]:
    now = time.time()
    removed_files = 0
    removed_bytes = 0
    remaining: list[tuple[float, int, Path]] = []
    for path in _preview_cache_files():
        try:
            stat = path.stat()
        except OSError:
            continue
        age = now - stat.st_mtime
        if max_age_seconds > 0 and age > max_age_seconds:
            try:
                path.unlink()
                removed_files += 1
                removed_bytes += stat.st_size
            except OSError:
                pass
            continue
        remaining.append((stat.st_mtime, stat.st_size, path))

    total_size = sum(size for _, size, _ in remaining)
    for _, size, path in sorted(remaining):
        if total_size <= max_bytes:
            break
        try:
            path.unlink()
            removed_files += 1
            removed_bytes += size
            total_size -= size
        except OSError:
            pass
    return removed_files, removed_bytes


def render_preview_cache_controls() -> None:
    stats = get_preview_cache_stats()
    with st.expander("Preview cache", icon=":material/cached:"):
        st.caption(
            f"{stats['path']} - {stats['files']} files / "
            f"{format_bytes(int(stats['bytes']))}"
        )
        st.caption(
            "Auto cleanup keeps preview files under 512 MB and removes files "
            "older than 7 days. Final simulation outputs are not deleted."
        )
        col1, col2 = st.columns(2)
        if col1.button(
            "Clean now",
            type="tertiary",
            icon=":material/cleaning_services:",
            width="content",
        ):
            count, size = enforce_preview_cache_limits()
            st.success(f"Removed {count} file(s), {format_bytes(size)}.")
        if col2.button(
            "Clear all",
            type="tertiary",
            icon=":material/delete:",
            width="content",
        ):
            count, size = clear_preview_cache()
            st.warning(f"Removed {count} file(s), {format_bytes(size)}.")


def clean_console_log(raw_bytes: bytes | None) -> str:
    if not raw_bytes:
        return ""
    text = raw_bytes.decode("utf-8", errors="replace")
    lines: list[str] = []
    current_line = ""
    for char in text:
        if char == "\n":
            if current_line.strip():
                lines.append(current_line)
            current_line = ""
        elif char == "\r":
            current_line = ""
        else:
            current_line += char
    if current_line.strip():
        lines.append(current_line)
    return "\n".join(lines)


def get_file_mtime_ns(path_like: str | os.PathLike[str]) -> int:
    path = resolve_project_path(path_like)
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(show_spinner=False, ttl="1h", max_entries=4)
def load_and_generate_3d_figure(
    raw_path: str,
    nz: int,
    ny: int,
    nx: int,
    file_mtime_ns: int,
    max_preview_dim: int = 100,
) -> tuple[go.Figure | None, np.ndarray]:
    path = resolve_project_path(raw_path)
    expected_size = int(nz) * int(ny) * int(nx)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"RAW size mismatch: expected {expected_size} bytes, got {actual_size} bytes."
        )

    final_grid = np.fromfile(path, dtype=np.uint8).reshape((nz, ny, nx))
    pixz = min(nz, max_preview_dim)
    pixy = min(ny, max_preview_dim)
    pixx = min(nx, max_preview_dim)
    preview_grid = final_grid[:pixz, :pixy, :pixx]

    traces: list[go.Mesh3d] = []
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    filler_ids = np.unique(preview_grid)
    filler_ids = filler_ids[filler_ids >= 4]

    for i, filler_id in enumerate(filler_ids):
        mask = (preview_grid == filler_id).astype(float)
        if not np.any(mask) or mask.min() == mask.max():
            continue
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
        traces.append(
            go.Mesh3d(
                x=verts[:, 2],
                y=verts[:, 1],
                z=verts[:, 0],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=colors[i % len(colors)],
                opacity=0.72,
                flatshading=False,
                lighting={
                    "ambient": 0.4,
                    "diffuse": 0.8,
                    "specular": 0.3,
                    "roughness": 0.4,
                },
                lightposition={"x": pixx, "y": pixy, "z": pixz * 2},
                name=f"phase {int(filler_id)}",
            )
        )

    if not traces:
        return None, final_grid

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis": {"range": [0, pixx], "title": "X"},
            "yaxis": {"range": [0, pixy], "title": "Y"},
            "zaxis": {"range": [0, pixz], "title": "Z"},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        height=450,
        legend={"orientation": "h"},
    )
    return fig, final_grid


@st.cache_data(show_spinner=False, ttl="5m", max_entries=8)
def load_csv_log(csv_path: str, file_mtime_ns: int) -> pd.DataFrame:
    path = resolve_project_path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# =============================================================================
# Dashboard Components
# =============================================================================

@st.dialog("HTML Report Preview", width="large")
def show_html_report_dialog(html_content: str) -> None:
    """Display the HTML report inside a modal dialog using native st.html."""
    st.html(html_content)


@st.fragment
def render_interactive_explorer(df: pd.DataFrame) -> None:
    """
    A lightweight, clean scatter plot explorer allowing selection of X, Y, and color axes.
    Decorated with @st.fragment so axis switching doesn't trigger a full app reload.
    """
    st.markdown("#### :material/scatter_plot: Metrics Explorer")
    st.caption("Select axes to explore the relationships between structure and properties.")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Pick categorical columns with fewer than 20 unique values for color coding
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    color_candidates = [c for c in categorical_cols if df[c].nunique() < 20]
    color_options = ["None"] + color_candidates

    # Smart defaults for physical properties vs structure metrics
    default_x = numeric_cols.index("Filler_Frac") if "Filler_Frac" in numeric_cols else 0
    default_y = numeric_cols.index("Connectivity_Ratio") if "Connectivity_Ratio" in numeric_cols else min(1, len(numeric_cols)-1)

    # Use max 3 columns for a clean look
    col_x, col_y, col_color = st.columns(3)
    
    x_axis = col_x.selectbox("X-axis", options=numeric_cols, index=default_x)
    y_axis = col_y.selectbox("Y-axis", options=numeric_cols, index=default_y)
    color_by = col_color.selectbox("Color by", options=color_options, index=0)

    chart_kwargs = {"x": x_axis, "y": y_axis}
    if color_by != "None":
        chart_kwargs["color"] = color_by

    # Streamlit native chart (no use_container_width required, expands natively)
    st.scatter_chart(df, **chart_kwargs)


def render_csv_dashboard(csv_path: str, *, html_out: str | None = None) -> None:
    """Main rendering function for the unified dashboard."""
    path = resolve_project_path(csv_path)
    st.caption(f"CSV log path: `{display_project_path(path)}`")
    
    if not path.exists():
        st.info(
            "No CSV log found yet. Run at least one workflow step first.",
            icon=":material/info:",
        )
        return

    df = load_csv_log(str(path), get_file_mtime_ns(path))
    if df.empty:
        st.warning("The CSV log is empty.", icon=":material/warning:")
        return

    # Keep top-level overview metrics
    with st.container(horizontal=True):
        st.metric("Rows", f"{len(df):,}", border=True)
        st.metric("Columns", f"{len(df.columns):,}", border=True)
        if "Mode" in df.columns:
            st.metric("Modes", f"{df['Mode'].nunique():,}", border=True)
        if "Basename" in df.columns:
            st.metric("Models", f"{df['Basename'].nunique():,}", border=True)

    st.write("") # Lightweight spacing

    # --- 1. HTML Report Actions (Side-by-side buttons) ---
    if html_out:
        st.markdown("#### :material/description: HTML Report")
        html_path = resolve_project_path(html_out)
        
        with st.container(horizontal=True):
            if html_path.exists():
                html_data = html_path.read_text(encoding="utf-8")
                
                st.download_button(
                    "Download HTML",
                    data=html_data,
                    file_name=html_path.name,
                    mime="text/html",
                    icon=":material/download:",
                )
                
                if st.button("Preview in Dialog", icon=":material/visibility:"):
                    show_html_report_dialog(html_data)
            else:
                st.caption("Generate the HTML report first to download or preview.")
                
        st.write("") # Lightweight spacing

    # --- 2. Interactive Explorer ---
    render_interactive_explorer(df)
    st.write("") # Lightweight spacing

    # --- 3. Raw Data Table ---
    st.markdown("#### :material/table: Raw Data")
    st.dataframe(df, hide_index=True)


def run_project_command(args: Iterable[str]) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        list(args),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )


def safe_base_from_final_raw(raw_path: str) -> str:
    raw_path_str = str(resolve_project_path(raw_path))
    suffix = "_final.raw"
    if raw_path_str.endswith(suffix):
        return raw_path_str[: -len(suffix)]
    if raw_path_str.endswith(".raw"):
        return raw_path_str[:-4]
    return raw_path_str


def read_grid_shape(nf_path: str | None, fallback_grid: int) -> tuple[int, int, int]:
    if nf_path:
        path = resolve_project_path(nf_path)
        if path.exists():
            try:
                import json

                with path.open("r", encoding="utf-8") as file:
                    meta = json.load(file)
                grid_size = meta.get("grid_size")
                if isinstance(grid_size, list) and len(grid_size) == 3:
                    return tuple(int(x) for x in grid_size)  # type: ignore[return-value]
            except Exception:
                pass
    fallback = int(fallback_grid)
    return fallback, fallback, fallback


def parse_raw_shape(raw_shape: str) -> list[str]:
    parts = raw_shape.split()
    if len(parts) != 3:
        raise ValueError("RAW shape must contain exactly three integers: Z Y X.")
    values = [int(part) for part in parts]
    if any(value <= 0 for value in values):
        raise ValueError("RAW shape values must be positive integers.")
    return [str(value) for value in values]
