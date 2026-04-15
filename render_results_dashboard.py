#!/usr/bin/env python3
"""Render a sortable HTML review dashboard from a CSV results file.

This standalone script reads a results CSV file, builds a compact dashboard-like
HTML table with data bars, and writes a self-contained HTML file. The output is
intended for quick visual review in a browser without adding heavy dependencies.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


# Default columns are ordered for quick review rather than completeness.
DEFAULT_COLUMN_CANDIDATES: List[str] = [
    "Basename",
    "Grid_Size",
    "Recipe",
    "BG_Type",
    "Mode",
    "Stretch_ratio",
    "PolymerA_Frac",
    "PolymerB_Frac",
    "Secondary_Inter_Frac",
    "Primary_Inter_Frac",
    "Filler_Frac",
    "chfem_Txx",
    "chfem_Tyy",
    "chfem_Tzz",
    "puma_Txx",
    "puma_Tyy",
    "puma_Tzz",
    "Contact_Ratio",
    "Tunneling_Ratio",
    "Connectivity_Ratio",
    "N_Conductive_Clusters",
    "N_Largest_Cluster_Voxels",
    "N_Conductive_Candidate_Voxels",
    "N_Filler_Voxels",
    "N_Contact_Voxels",
    "N_Tunnel_Voxels",
    "chfem_Time_s",
    "puma_Time_s",
]

LINEAR_BAR_KEYWORDS: Tuple[str, ...] = ("ratio", "frac", "time")
LOG_BAR_KEYWORDS: Tuple[str, ...] = ("txx", "tyy", "tzz", "txy", "tyz", "tzx")
COUNT_BAR_KEYWORDS: Tuple[str, ...] = ("voxels", "clusters", "n_")
CATEGORICAL_KEYWORDS: Tuple[str, ...] = ("model", "recipe", "solver", "type", "mode", "grid_size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a self-contained HTML review dashboard from a CSV file."
    )
    parser.add_argument("--csv", required=True, help="Input CSV file path.")
    parser.add_argument("--output", required=True, help="Output HTML file path.")
    parser.add_argument("--columns", nargs="*", default=None, help="Optional list of columns to display.")
    parser.add_argument("--sort-by", default=None, help="Optional column name used for initial sorting.")
    parser.add_argument("--descending", action="store_true", help="Use descending order for the initial sort.")
    parser.add_argument("--max-rows", type=int, default=200, help="Maximum number of rows to include.")
    parser.add_argument("--title", default="Simulation Results Dashboard", help="Dashboard title.")
    parser.add_argument("--subtitle", default="Sortable summary with inline data bars.", help="Dashboard subtitle.")
    return parser.parse_args()


def read_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def select_columns(df: pd.DataFrame, user_columns: Optional[Sequence[str]]) -> List[str]:
    if user_columns:
        return [col for col in user_columns if col in df.columns]
    selected: List[str] = []
    for candidate in DEFAULT_COLUMN_CANDIDATES:
        if candidate in df.columns and candidate not in selected:
            selected.append(candidate)
    return selected if selected else list(df.columns[: min(15, len(df.columns))])


def split_recipe_column(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Split the 'Recipe' column by space into multiple columns dynamically."""
    if "Recipe" in df.columns and "Recipe" in columns:
        split_df = df["Recipe"].fillna("").astype(str).str.split(r"\s+", expand=True)
        if split_df.shape[1] > 0:
            recipe_cols = [f"Recipe_{i+1}" for i in range(split_df.shape[1])]
            split_df.columns = recipe_cols
            df = pd.concat([df.drop(columns=["Recipe"]), split_df], axis=1)
            idx = columns.index("Recipe")
            columns = columns[:idx] + recipe_cols + columns[idx+1:]
    return df, columns


def apply_initial_sort(df: pd.DataFrame, sort_by: Optional[str], descending: bool) -> pd.DataFrame:
    if sort_by and sort_by in df.columns:
        return df.sort_values(by=sort_by, ascending=not descending, na_position="last")
    return df


def detect_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    numeric_columns: List[str] = []
    for col in columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(col)
    return numeric_columns


def classify_bar_attributes(column_name: str) -> Optional[Tuple[str, str]]:
    """Determine the scaling mode and color theme based on column naming conventions."""
    lower = column_name.lower()
    if any(k in lower for k in COUNT_BAR_KEYWORDS):
        return ("log", "count")
    if any(k in lower for k in LOG_BAR_KEYWORDS):
        return ("log", "tensor")
    if any(k in lower for k in LINEAR_BAR_KEYWORDS):
        return ("linear", "ratio")
    return None


def is_categorical(column_name: str) -> bool:
    """Determine if a column should be rendered as a categorical badge."""
    return any(k in column_name.lower() for k in CATEGORICAL_KEYWORDS)


def deterministic_color_index(text: str, max_colors: int = 5) -> int:
    """Assign a deterministic color index based on string content."""
    return sum(ord(c) for c in text) % max_colors


def is_ratio_like(column_name: str) -> bool:
    lower_name = column_name.lower()
    return "ratio" in lower_name or "fraction" in lower_name or lower_name.startswith("vf")


def format_value(value: object, column_name: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lower_col = column_name.lower()
        if any(k in lower_col for k in COUNT_BAR_KEYWORDS):
            return f"{int(float(value)):,}"
        if is_ratio_like(column_name):
            return f"{value:.4f}"
        
        abs_value = abs(float(value))
        if abs_value == 0:
            return "0"
        if abs_value >= 10000 or abs_value < 1e-4:
            return f"{value:.3e}"
        if abs_value >= 100:
            return f"{value:.2f}"
        if abs_value >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def build_bar_normalizer(series: pd.Series, mode: str):
    """Create a normalization function (0.0 to 1.0) for inline data bars."""
    valid = series.dropna()
    if valid.empty:
        return lambda _: None

    if mode == "log":
        positive = valid[valid > 0]
        if positive.empty:
            return lambda _: None
        log_values = positive.map(math.log10)
        lo = float(log_values.min())
        hi = float(log_values.max())
        if math.isclose(lo, hi):
            return lambda value: 1.0 if pd.notna(value) and float(value) > 0 else None
        def normalize(value: object) -> Optional[float]:
            if pd.isna(value): return None
            value_f = float(value)
            if value_f <= 0: return None
            return max(0.0, min(1.0, (math.log10(value_f) - lo) / (hi - lo)))
        return normalize

    lo = float(valid.min())
    hi = float(valid.max())
    if math.isclose(lo, hi):
        return lambda value: 1.0 if pd.notna(value) else None
    def normalize(value: object) -> Optional[float]:
        if pd.isna(value): return None
        return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))
    return normalize


def prepare_bar_metadata(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Tuple[str, str, object]]:
    metadata: Dict[str, Tuple[str, str, object]] = {}
    numeric_series = {col: pd.to_numeric(df[col], errors="coerce") for col in columns}
    for col, series in numeric_series.items():
        attrs = classify_bar_attributes(col)
        if attrs is None: continue
        mode, theme = attrs
        metadata[col] = (mode, theme, build_bar_normalizer(series, mode))
    return metadata


def make_header_html(columns: Sequence[str], numeric_cols: List[str], bar_metadata: dict) -> str:
    header_cells = []
    for col in columns:
        escaped = html.escape(col)
        is_num = col in numeric_cols
        has_bar = col in bar_metadata
        
        # Enforce strict pixel widths for ALL columns to prevent flex/ratio bugs in table-layout: fixed
        if has_bar:
            align_class = "th-num"
            base_class = ""
            style = "width: 150px; min-width: 150px; max-width: 150px;"
            resizer_html = ""
        elif is_num:
            align_class = "th-num"
            base_class = ""
            style = "width: 100px; min-width: 100px; max-width: 100px;"
            resizer_html = ""
        else:
            align_class = "th-text"
            base_class = "resizable-th"
            style = "width: 50px; min-width: 50px; max-width: 50px;"
            resizer_html = '<div class="resizer" title="Drag to resize"></div>'
            
        header_cells.append(
            f'<th class="sortable {align_class} {base_class}" data-column="{escaped}" style="{style}">'
            f'<button type="button" title="{escaped}">'
            f'<span class="header-text">{escaped}</span>'
            f'<span class="sort-indicator"></span>'
            f'</button>'
            f'{resizer_html}'
            f'</th>'
        )
    return "\n".join(header_cells)


def render_plain_cell(display_text: str, sort_value: str, is_numeric: bool, is_cat: bool, raw_text: str, col_name: str) -> str:
    escaped_sort = html.escape(sort_value)
    escaped_raw = html.escape(raw_text)
    lower_col = col_name.lower()
    
    # Path simplification for Basename
    if lower_col == "basename" and "/" in raw_text:
        short_text = raw_text.rsplit("/", 1)[-1]
        content = f'<span class="cell-text">{html.escape(short_text)}</span>'
        return f'<td class="text-cell" data-sort-value="{escaped_sort}" title="{escaped_raw}">{content}</td>'

    # Badge rendering for categorical variables
    if is_cat and not is_numeric and display_text:
        color_idx = deterministic_color_index(display_text)
        content = f'<span class="badge badge-{color_idx}">{html.escape(display_text)}</span>'
        cell_class = "cat-cell"
    else:
        content = f'<span class="cell-text">{html.escape(display_text)}</span>'
        cell_class = "num-cell" if is_numeric else "text-cell"

    return f'<td class="{cell_class}" data-sort-value="{escaped_sort}" title="{escaped_raw}">{content}</td>'


def render_bar_cell(display_text: str, sort_value: str, ratio: Optional[float], theme: str, raw_text: str) -> str:
    escaped_text = html.escape(display_text)
    escaped_sort = html.escape(sort_value)
    escaped_raw = html.escape(raw_text)
    ratio_percent = 0.0 if ratio is None else max(0.0, min(100.0, ratio * 100.0))
    
    return (
        f'<td class="bar-cell" data-sort-value="{escaped_sort}" title="{escaped_raw}">'
        f'<div class="bar-shell">'
        f'<div class="bar-fill bar-{theme}" style="width:{ratio_percent:.2f}%"></div>'
        f'<span class="bar-label">{escaped_text}</span>'
        f'</div>'
        f'</td>'
    )


def make_body_html(
    df: pd.DataFrame, columns: Sequence[str], bar_metadata: Dict[str, Tuple[str, str, object]], numeric_cols: List[str]
) -> str:
    rows_html: List[str] = []
    for _, row in df.iterrows():
        cells_html: List[str] = []
        for col in columns:
            raw_value = row[col]
            is_num = col in numeric_cols
            is_cat = is_categorical(col)
            numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            display_text = format_value(raw_value, col)
            raw_text_for_tooltip = str(raw_value) if pd.notna(raw_value) else ""
            sort_value = "" if pd.isna(numeric_value) else repr(float(numeric_value))
            
            if col in bar_metadata:
                mode, theme, normalizer = bar_metadata[col]
                ratio = normalizer(numeric_value)
                cells_html.append(render_bar_cell(display_text, sort_value or display_text, ratio, theme, raw_text_for_tooltip))
            else:
                cells_html.append(render_plain_cell(display_text, sort_value or display_text, is_num, is_cat, raw_text_for_tooltip, col))
        rows_html.append("<tr>" + "".join(cells_html) + "</tr>")
    return "\n".join(rows_html)


def build_dashboard_html(df: pd.DataFrame, columns: Sequence[str], title: str, subtitle: str) -> str:
    numeric_cols = detect_numeric_columns(df, columns)
    bar_metadata = prepare_bar_metadata(df, columns)
    header_html = make_header_html(columns, numeric_cols, bar_metadata)
    body_html = make_body_html(df, columns, bar_metadata, numeric_cols)
    
    escaped_title = html.escape(title)
    escaped_subtitle = html.escape(subtitle)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
/* CSS Variables for Theming */
:root {{
  /* ---- Dark Theme (Default) ---- */
  --bg: #0b1020;
  --panel: #131a2a;
  --panel-2: #172033;
  --grid: #25324a;
  --text: #e5ecf6;
  --muted: #99a7bd;
  --accent: #66b3ff;
  
  /* Row Alternation & Hover */
  --row-odd: rgba(255, 255, 255, 0.01);
  --row-hover: rgba(102, 179, 255, 0.06);
  --sticky-bg: var(--panel);
  --sticky-odd: #151c2d;
  --sticky-hover: #192336;
  --sticky-border: #3b4b6b;

  /* Data Bar Gradients */
  --bar-ratio: linear-gradient(90deg, rgba(126,224,195,0.55), rgba(102,179,255,0.45));
  --bar-tensor: linear-gradient(90deg, rgba(199,126,224,0.55), rgba(255,102,179,0.45));
  --bar-count: linear-gradient(90deg, rgba(224,195,126,0.55), rgba(255,179,102,0.45));
  
  /* Categorical Badge Palettes */
  --badge-0-bg: rgba(102,179,255,0.15); --badge-0-fg: #66b3ff; --badge-0-border: rgba(102,179,255,0.3);
  --badge-1-bg: rgba(126,224,195,0.15); --badge-1-fg: #7ee0c3; --badge-1-border: rgba(126,224,195,0.3);
  --badge-2-bg: rgba(255,179,102,0.15); --badge-2-fg: #ffb366; --badge-2-border: rgba(255,179,102,0.3);
  --badge-3-bg: rgba(199,126,224,0.15); --badge-3-fg: #c77ee0; --badge-3-border: rgba(199,126,224,0.3);
  --badge-4-bg: rgba(255,102,179,0.15); --badge-4-fg: #ff66b3; --badge-4-border: rgba(255,102,179,0.3);
}}

[data-theme="light"] {{
  /* ---- Light Theme (CVD Friendly) ---- */
  --bg: #f8f9fa;
  --panel: #ffffff;
  --panel-2: #e9ecef;
  --grid: #dee2e6;
  --text: #212529;
  --muted: #6c757d;
  --accent: #0072B2;
  
  --row-odd: rgba(0, 0, 0, 0.02);
  --row-hover: rgba(0, 114, 178, 0.06);
  --sticky-bg: var(--panel);
  --sticky-odd: #fbfbfc;
  --sticky-hover: #f1f7fb;
  --sticky-border: #adb5bd;

  /* CVD Friendly Data Bar Gradients (Okabe-Ito inspired) */
  --bar-ratio: linear-gradient(90deg, rgba(86,180,233,0.4), rgba(0,158,115,0.4));
  --bar-tensor: linear-gradient(90deg, rgba(204,121,167,0.4), rgba(0,114,178,0.4));
  --bar-count: linear-gradient(90deg, rgba(240,228,66,0.6), rgba(213,94,0,0.4));

  /* CVD Friendly Badge Palettes (Okabe-Ito inspired) */
  --badge-0-bg: rgba(86, 180, 233, 0.15); --badge-0-fg: #0072B2; --badge-0-border: #56B4E9; /* Sky Blue */
  --badge-1-bg: rgba(0, 158, 115, 0.15); --badge-1-fg: #009E73; --badge-1-border: #009E73; /* Bluish Green */
  --badge-2-bg: rgba(213, 94, 0, 0.1); --badge-2-fg: #D55E00; --badge-2-border: #E69F00; /* Vermillion / Orange */
  --badge-3-bg: rgba(204, 121, 167, 0.15); --badge-3-fg: #CC79A7; --badge-3-border: #CC79A7; /* Reddish Purple */
  --badge-4-bg: rgba(240, 228, 66, 0.3); --badge-4-fg: #D55E00; --badge-4-border: #E69F00; /* Yellow */
}}

/* Global Reset & Typography */
* {{ box-sizing: border-box; transition: background-color 0.2s, color 0.2s; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font-family: Arial, "BIZ UD Gothic", sans-serif;
}}
.main {{ padding: 20px 24px 28px; }}

/* Header Area */
.header-container {{
  display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;
}}
.title {{ font-size: 24px; font-weight: 700; margin: 0 0 4px; }}
.subtitle {{ font-size: 13px; color: var(--muted); margin: 0; }}
.meta {{ margin-top: 8px; font-size: 12px; color: var(--muted); }}

.theme-toggle-btn {{
  background: var(--panel); border: 1px solid var(--grid); color: var(--text);
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: bold;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.theme-toggle-btn:hover {{ background: var(--panel-2); border-color: var(--accent); }}

/* Table Container */
.table-wrap {{
  border: 1px solid var(--grid); border-radius: 12px; overflow: auto;
  background: var(--panel); box-shadow: 0 8px 24px rgba(0,0,0,0.15);
  max-height: calc(100vh - 120px);
}}

/* Removing min-width:100% prevents the browser from auto-flexing columns when there is no horizontal scroll */
table {{ 
  width: max-content; 
  border-collapse: separate; 
  border-spacing: 0; 
  table-layout: fixed; 
}}

/* Stacking Context Strategy (Z-Index) */
thead th {{
  position: sticky; top: 0; z-index: 10; background: var(--panel-2);
  border-bottom: 1px solid var(--grid); padding: 0;
}}
thead th:first-child {{
  left: 0; z-index: 12; border-right: 1px solid var(--sticky-border);
}}
tbody td:first-child {{
  position: sticky; left: 0; z-index: 11; background: var(--sticky-bg);
  border-right: 1px solid var(--sticky-border); font-weight: bold;
}}
tbody tr:nth-child(odd) td:first-child {{ background: var(--sticky-odd); }}
tbody tr:hover td:first-child {{ background: var(--sticky-hover); }}

/* Resizer Handle */
.resizer {{
  position: absolute; right: 0; top: 0; bottom: 0; width: 6px;
  cursor: col-resize; background: rgba(255,255,255,0.05);
  z-index: 2; opacity: 0; transition: opacity 0.2s;
}}
th:hover .resizer, .resizer.resizing {{ opacity: 1; background: var(--accent); }}

thead th button {{
  appearance: none; width: 100%; border: 0; background: transparent; color: var(--text);
  cursor: pointer; font: inherit; font-size: 12px; font-weight: 600; padding: 8px 14px 8px 12px;
  display: flex; align-items: center; justify-content: space-between; gap: 6px;
}}
thead th button:hover {{ background: rgba(128,128,128,0.1); }}
.header-text {{
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden; text-overflow: ellipsis; white-space: normal; line-height: 1.3; text-align: left;
}}
.th-num .header-text {{ text-align: right; width: 100%; }}

/* Body Cells */
tbody td {{
  padding: 8px 12px; border-bottom: 1px solid var(--grid);
  white-space: nowrap; vertical-align: middle; font-size: 13px;
}}
.text-cell {{ text-align: left; overflow: hidden; text-overflow: ellipsis; }}
.cat-cell {{ text-align: left; overflow: hidden; text-overflow: ellipsis; }}
.num-cell {{ text-align: right; font-variant-numeric: tabular-nums; }}

tbody tr:nth-child(odd) td:not(:first-child) {{ background: var(--row-odd); }}
tbody tr:hover td:not(:first-child) {{ background: var(--row-hover); }}

.sort-indicator {{ color: var(--muted); display: inline-block; width: 10px; flex-shrink: 0; text-align: center; }}
th.sorted-asc .sort-indicator::after {{ content: "▲"; }}
th.sorted-desc .sort-indicator::after {{ content: "▼"; }}

/* Components */
.bar-shell {{
  position: relative; width: 100%; height: 24px; border-radius: 6px;
  overflow: hidden; background: rgba(128,128,128,0.1); border: 1px solid rgba(128,128,128,0.2);
}}
.bar-fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 0px; }}
.bar-ratio {{ background: var(--bar-ratio); }}
.bar-tensor {{ background: var(--bar-tensor); }}
.bar-count {{ background: var(--bar-count); }}
.bar-label {{
  position: relative; z-index: 1; display: flex; align-items: center; justify-content: flex-end;
  width: 100%; height: 100%; padding: 0 10px; font-size: 12px; font-variant-numeric: tabular-nums;
  text-shadow: 0 1px 2px rgba(255,255,255,0.1);
}}
[data-theme="dark"] .bar-label {{ text-shadow: 0 1px 2px rgba(0,0,0,0.5); }}

.badge {{
  display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px;
  font-weight: 600; max-width: 100%; overflow: hidden; text-overflow: ellipsis; border: 1px solid;
  vertical-align: middle;
}}
.badge-0 {{ background: var(--badge-0-bg); color: var(--badge-0-fg); border-color: var(--badge-0-border); }}
.badge-1 {{ background: var(--badge-1-bg); color: var(--badge-1-fg); border-color: var(--badge-1-border); }}
.badge-2 {{ background: var(--badge-2-bg); color: var(--badge-2-fg); border-color: var(--badge-2-border); }}
.badge-3 {{ background: var(--badge-3-bg); color: var(--badge-3-fg); border-color: var(--badge-3-border); }}
.badge-4 {{ background: var(--badge-4-bg); color: var(--badge-4-fg); border-color: var(--badge-4-border); }}
</style>
</head>
<body data-theme="dark">
<div class="main">
  <div class="header-container">
    <div>
      <h1 class="title">{escaped_title}</h1>
      <p class="subtitle">{escaped_subtitle}</p>
      <div class="meta">Rows shown: {len(df)} | Click a column header to sort.</div>
    </div>
    <button id="theme-toggle" class="theme-toggle-btn">☀️ Light Mode</button>
  </div>
  <div class="table-wrap">
    <table id="dashboard-table">
      <thead><tr>{header_html}</tr></thead>
      <tbody>{body_html}</tbody>
    </table>
  </div>
</div>
<script>
(function() {{
  // Theme Toggle Logic
  const toggleBtn = document.getElementById('theme-toggle');
  toggleBtn.addEventListener('click', () => {{
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    if (isDark) {{
      document.body.setAttribute('data-theme', 'light');
      toggleBtn.textContent = '🌙 Dark Mode';
    }} else {{
      document.body.setAttribute('data-theme', 'dark');
      toggleBtn.textContent = '☀️ Light Mode';
    }}
  }});

  // Table Logic
  const table = document.getElementById('dashboard-table');
  const headers = Array.from(table.querySelectorAll('thead th'));
  const tbody = table.querySelector('tbody');

  function parseSortValue(cell) {{
    const raw = cell.getAttribute('data-sort-value') || '';
    const numeric = Number(raw);
    if (raw !== '' && !Number.isNaN(numeric)) return {{ type: 'number', value: numeric }};
    return {{ type: 'string', value: raw.toLowerCase() }};
  }}

  // Sorting
  headers.forEach((th, columnIndex) => {{
    const btn = th.querySelector('button');
    if (!btn) return;
    btn.addEventListener('click', () => {{
      const nextDesc = !th.classList.contains('sorted-desc');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((rowA, rowB) => {{
        const a = parseSortValue(rowA.children[columnIndex]);
        const b = parseSortValue(rowB.children[columnIndex]);
        if (a.type === 'number' && b.type === 'number') return nextDesc ? b.value - a.value : a.value - b.value;
        if (a.value < b.value) return nextDesc ? 1 : -1;
        if (a.value > b.value) return nextDesc ? -1 : 1;
        return 0;
      }});
      headers.forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
      th.classList.add(nextDesc ? 'sorted-desc' : 'sorted-asc');
      rows.forEach(row => tbody.appendChild(row));
    }});
  }});

  // Column Resizer Logic
  const resizers = document.querySelectorAll('.resizer');
  resizers.forEach(resizer => {{
    let startX = 0; let startWidth = 0; let th = null; let colIndex = -1;

    resizer.addEventListener('mousedown', (e) => {{
      e.preventDefault(); e.stopPropagation(); 
      th = resizer.parentElement;
      startX = e.clientX;
      const rect = th.getBoundingClientRect();
      startWidth = rect.width;
      colIndex = Array.from(th.parentElement.children).indexOf(th);
      
      resizer.classList.add('resizing');
      document.body.style.cursor = 'col-resize';
      
      const onMouseMove = (moveEvent) => {{
        const currentWidth = Math.max(60, startWidth + (moveEvent.clientX - startX));
        const widthPx = currentWidth + 'px';
        th.style.width = widthPx;
        th.style.minWidth = widthPx;
        th.style.maxWidth = widthPx;
        
        const cells = table.querySelectorAll('tbody tr td:nth-child(' + (colIndex + 1) + ')');
        cells.forEach(cell => {{
           cell.style.width = widthPx;
           cell.style.minWidth = widthPx;
           cell.style.setProperty('max-width', widthPx, 'important');
        }});
      }};
      
      const onMouseUp = () => {{
        resizer.classList.remove('resizing');
        document.body.style.cursor = '';
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
      }};
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    }});
  }});
}})();
</script>
</body>
</html>
"""



def render_dashboard_from_csv(
    csv_path: Path | str,
    output_path: Path | str,
    columns: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = None,
    descending: bool = True,
    max_rows: int = 200,
    title: str = "Simulation Results Dashboard",
    subtitle: str = "Sortable summary with inline data bars.",
) -> Path:
    """Render a self-contained HTML dashboard directly from a CSV path."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = read_csv(csv_path)
    selected_columns = select_columns(df, columns)
    df, selected_columns = split_recipe_column(df, selected_columns)
    df = apply_initial_sort(df, sort_by, descending)
    df = df.loc[:, selected_columns].head(max_rows).copy()

    html_text = build_dashboard_html(df, selected_columns, title, subtitle)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path

def main() -> None:
    args = parse_args()
    output_path = render_dashboard_from_csv(
        csv_path=args.csv,
        output_path=args.output,
        columns=args.columns,
        sort_by=args.sort_by,
        descending=args.descending,
        max_rows=args.max_rows,
        title=args.title,
        subtitle=args.subtitle,
    )
    print(f"Saved HTML dashboard to: {output_path}")


if __name__ == "__main__":
    main()
