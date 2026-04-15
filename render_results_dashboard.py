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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# Default columns are ordered for quick review rather than completeness.
DEFAULT_COLUMN_CANDIDATES: List[str] = [
    "Model",
    "ModelName",
    "Basename",
    "Dir",
    "Recipe",
    "Solver",
    "BG_Type",
    "Physics_Mode",
    "VF_Target",
    "VF_Filler",
    "Filler_Volume_Fraction",
    "Chfem_Txx",
    "PuMA_Txx",
    "Chfem_Tyy",
    "PuMA_Tyy",
    "Chfem_Tzz",
    "PuMA_Tzz",
    "Contact_Ratio",
    "Tunneling_Ratio",
    "Connectivity_Ratio",
    "N_Conductive_Clusters",
    "N_Largest_Cluster_Voxels",
    "N_Conductive_Candidate_Voxels",
    "N_Filler_Voxels",
    "N_Contact_Voxels",
    "N_Tunnel_Voxels",
]

# Columns that are usually easier to compare with a linear data bar.
LINEAR_BAR_KEYWORDS: Tuple[str, ...] = (
    "ratio",
    "fraction",
    "vf",
    "volume_fraction",
    "voxels",
    "clusters",
)

# Columns that often span orders of magnitude and benefit from log scaling.
LOG_BAR_KEYWORDS: Tuple[str, ...] = (
    "cond",
    "sigma",
    "txx",
    "tyy",
    "tzz",
    "txy",
    "tyz",
    "tzx",
    "effective",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a self-contained HTML review dashboard from a CSV file."
    )
    parser.add_argument("--csv", required=True, help="Input CSV file path.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional list of columns to display in the given order.",
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        help="Optional column name used for initial sorting.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Use descending order for the initial sort.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=40,
        help="Maximum number of rows to include in the dashboard.",
    )
    parser.add_argument(
        "--title",
        default="microsimflow review dashboard",
        help="Dashboard title.",
    )
    parser.add_argument(
        "--subtitle",
        default="Sortable HTML summary with inline data bars.",
        help="Dashboard subtitle.",
    )
    return parser.parse_args()


def read_csv(csv_path: Path) -> pd.DataFrame:
    # Keep the original columns and let pandas infer types where possible.
    return pd.read_csv(csv_path)


def select_columns(df: pd.DataFrame, user_columns: Optional[Sequence[str]]) -> List[str]:
    if user_columns:
        return [col for col in user_columns if col in df.columns]

    selected: List[str] = []
    for candidate in DEFAULT_COLUMN_CANDIDATES:
        if candidate in df.columns and candidate not in selected:
            selected.append(candidate)

    if selected:
        return selected

    # Fall back to the first columns when none of the preferred names exist.
    return list(df.columns[: min(12, len(df.columns))])


def apply_initial_sort(
    df: pd.DataFrame,
    sort_by: Optional[str],
    descending: bool,
) -> pd.DataFrame:
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


def compute_numeric_series(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, pd.Series]:
    return {col: pd.to_numeric(df[col], errors="coerce") for col in columns}


def classify_bar_mode(column_name: str) -> Optional[str]:
    lower_name = column_name.lower()
    if any(token in lower_name for token in LOG_BAR_KEYWORDS):
        return "log"
    if any(token in lower_name for token in LINEAR_BAR_KEYWORDS):
        return "linear"
    return None


def is_ratio_like(column_name: str) -> bool:
    lower_name = column_name.lower()
    return "ratio" in lower_name or "fraction" in lower_name or lower_name.startswith("vf")


def format_value(value: object, column_name: str) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if is_ratio_like(column_name):
            return f"{value:.4f}"
        abs_value = abs(float(value))
        if abs_value == 0:
            return "0"
        if abs_value >= 1000 or abs_value < 1e-3:
            return f"{value:.3e}"
        if abs_value >= 100:
            return f"{value:.1f}"
        if abs_value >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"

    return str(value)


def build_bar_normalizer(series: pd.Series, mode: str):
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
            if pd.isna(value):
                return None
            value_f = float(value)
            if value_f <= 0:
                return None
            return max(0.0, min(1.0, (math.log10(value_f) - lo) / (hi - lo)))

        return normalize

    lo = float(valid.min())
    hi = float(valid.max())
    if math.isclose(lo, hi):
        return lambda value: 1.0 if pd.notna(value) else None

    def normalize(value: object) -> Optional[float]:
        if pd.isna(value):
            return None
        value_f = float(value)
        return max(0.0, min(1.0, (value_f - lo) / (hi - lo)))

    return normalize


def prepare_bar_metadata(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Tuple[str, object]]:
    metadata: Dict[str, Tuple[str, object]] = {}
    numeric_series = compute_numeric_series(df, columns)
    for col, series in numeric_series.items():
        mode = classify_bar_mode(col)
        if mode is None:
            continue
        metadata[col] = (mode, build_bar_normalizer(series, mode))
    return metadata


def make_header_html(columns: Sequence[str]) -> str:
    header_cells = []
    for col in columns:
        escaped = html.escape(col)
        header_cells.append(
            f'<th class="sortable" data-column="{escaped}"><button type="button">{escaped}<span class="sort-indicator"></span></button></th>'
        )
    return "\n".join(header_cells)


def render_plain_cell(display_text: str, sort_value: str) -> str:
    escaped_text = html.escape(display_text)
    escaped_sort = html.escape(sort_value)
    return f'<td data-sort-value="{escaped_sort}"><span class="cell-text">{escaped_text}</span></td>'


def render_bar_cell(display_text: str, sort_value: str, ratio: Optional[float], mode: str) -> str:
    escaped_text = html.escape(display_text)
    escaped_sort = html.escape(sort_value)
    ratio_percent = 0.0 if ratio is None else max(0.0, min(100.0, ratio * 100.0))
    mode_class = "bar-log" if mode == "log" else "bar-linear"
    return (
        f'<td class="bar-cell" data-sort-value="{escaped_sort}">'
        f'<div class="bar-shell">'
        f'<div class="bar-fill {mode_class}" style="width:{ratio_percent:.2f}%"></div>'
        f'<span class="bar-label">{escaped_text}</span>'
        f'</div>'
        f'</td>'
    )


def make_body_html(
    df: pd.DataFrame,
    columns: Sequence[str],
    bar_metadata: Dict[str, Tuple[str, object]],
) -> str:
    rows_html: List[str] = []
    for _, row in df.iterrows():
        cells_html: List[str] = []
        for col in columns:
            raw_value = row[col]
            numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            display_text = format_value(raw_value, col)
            sort_value = "" if pd.isna(numeric_value) else repr(float(numeric_value))
            if col in bar_metadata:
                mode, normalizer = bar_metadata[col]
                ratio = normalizer(numeric_value)
                cells_html.append(render_bar_cell(display_text, sort_value or display_text, ratio, mode))
            else:
                cells_html.append(render_plain_cell(display_text, sort_value or display_text))
        rows_html.append("<tr>" + "".join(cells_html) + "</tr>")
    return "\n".join(rows_html)


def build_dashboard_html(
    df: pd.DataFrame,
    columns: Sequence[str],
    title: str,
    subtitle: str,
) -> str:
    header_html = make_header_html(columns)
    bar_metadata = prepare_bar_metadata(df, columns)
    body_html = make_body_html(df, columns, bar_metadata)
    row_count = len(df)
    escaped_title = html.escape(title)
    escaped_subtitle = html.escape(subtitle)

    # Keep the document self-contained so it can be opened offline.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
:root {{
  --bg: #0b1020;
  --panel: #131a2a;
  --panel-2: #172033;
  --grid: #25324a;
  --text: #e5ecf6;
  --muted: #99a7bd;
  --accent: #66b3ff;
  --accent-2: #7ee0c3;
  --bar-log: linear-gradient(90deg, rgba(102,179,255,0.60), rgba(126,224,195,0.65));
  --bar-linear: linear-gradient(90deg, rgba(126,224,195,0.55), rgba(102,179,255,0.45));
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.main {{
  padding: 20px 24px 28px;
}}
.header {{
  margin-bottom: 16px;
}}
.title {{
  font-size: 24px;
  font-weight: 700;
  margin: 0 0 4px;
}}
.subtitle {{
  font-size: 13px;
  color: var(--muted);
  margin: 0;
}}
.meta {{
  margin-top: 8px;
  font-size: 12px;
  color: var(--muted);
}}
.table-wrap {{
  border: 1px solid var(--grid);
  border-radius: 14px;
  overflow: auto;
  background: var(--panel);
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.20);
}}
table {{
  width: max-content;
  min-width: 100%;
  border-collapse: separate;
  border-spacing: 0;
}}
thead th {{
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--panel-2);
  border-bottom: 1px solid var(--grid);
  padding: 0;
  text-align: left;
  white-space: nowrap;
}}
thead th button {{
  appearance: none;
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--text);
  cursor: pointer;
  font: inherit;
  font-weight: 600;
  padding: 10px 12px;
  text-align: left;
}}
thead th button:hover {{
  background: rgba(255, 255, 255, 0.03);
}}
tbody td {{
  padding: 9px 12px;
  border-bottom: 1px solid rgba(37, 50, 74, 0.75);
  white-space: nowrap;
  vertical-align: middle;
}}
tbody tr:nth-child(odd) td {{
  background: rgba(255, 255, 255, 0.01);
}}
tbody tr:hover td {{
  background: rgba(102, 179, 255, 0.06);
}}
.sort-indicator {{
  margin-left: 6px;
  color: var(--muted);
}}
th.sorted-asc .sort-indicator::after {{ content: "▲"; }}
th.sorted-desc .sort-indicator::after {{ content: "▼"; }}
.cell-text {{
  font-variant-numeric: tabular-nums;
}}
.bar-cell {{
  min-width: 164px;
}}
.bar-shell {{
  position: relative;
  min-width: 140px;
  height: 28px;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
}}
.bar-fill {{
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 7px;
}}
.bar-linear {{
  background: var(--bar-linear);
}}
.bar-log {{
  background: var(--bar-log);
}}
.bar-label {{
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  height: 100%;
  padding: 0 10px;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}}
.footer {{
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
}}
</style>
</head>
<body>
<div class="main">
  <div class="header">
    <h1 class="title">{escaped_title}</h1>
    <p class="subtitle">{escaped_subtitle}</p>
    <div class="meta">Rows shown: {row_count} | Click a column header to sort.</div>
  </div>
  <div class="table-wrap">
    <table id="dashboard-table">
      <thead>
        <tr>
          {header_html}
        </tr>
      </thead>
      <tbody>
        {body_html}
      </tbody>
    </table>
  </div>
  <div class="footer">This file is self-contained and can be reviewed offline in a web browser.</div>
</div>
<script>
(function() {{
  const table = document.getElementById('dashboard-table');
  const headers = Array.from(table.querySelectorAll('thead th'));
  const tbody = table.querySelector('tbody');

  function parseSortValue(cell) {{
    const raw = cell.getAttribute('data-sort-value') || '';
    const numeric = Number(raw);
    if (raw !== '' && !Number.isNaN(numeric)) {{
      return {{ type: 'number', value: numeric }};
    }}
    return {{ type: 'string', value: raw.toLowerCase() }};
  }}

  function clearHeaderState() {{
    headers.forEach((th) => th.classList.remove('sorted-asc', 'sorted-desc'));
  }}

  headers.forEach((th, columnIndex) => {{
    th.querySelector('button').addEventListener('click', () => {{
      const currentDesc = th.classList.contains('sorted-desc');
      const nextDesc = !currentDesc;
      const rows = Array.from(tbody.querySelectorAll('tr'));

      rows.sort((rowA, rowB) => {{
        const a = parseSortValue(rowA.children[columnIndex]);
        const b = parseSortValue(rowB.children[columnIndex]);
        if (a.type === 'number' && b.type === 'number') {{
          return nextDesc ? b.value - a.value : a.value - b.value;
        }}
        if (a.value < b.value) return nextDesc ? 1 : -1;
        if (a.value > b.value) return nextDesc ? -1 : 1;
        return 0;
      }});

      clearHeaderState();
      th.classList.add(nextDesc ? 'sorted-desc' : 'sorted-asc');
      rows.forEach((row) => tbody.appendChild(row));
    }});
  }});
}})();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    df = read_csv(csv_path)
    columns = select_columns(df, args.columns)
    df = apply_initial_sort(df, args.sort_by, args.descending)
    df = df.loc[:, columns].head(args.max_rows).copy()

    html_text = build_dashboard_html(df, columns, args.title, args.subtitle)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Saved HTML dashboard to: {output_path}")


if __name__ == "__main__":
    main()
