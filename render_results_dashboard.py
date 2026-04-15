#!/usr/bin/env python3
"""Render a review-friendly dashboard table from a CSV file.

This script reads a result CSV, selects a compact set of columns, and renders
an Excel-like dashboard table with in-cell data bars. The output is a PNG image
that can be quickly shared for review.

The implementation uses pandas for tabular processing and Playwright for HTML
rendering and PNG export. The script is intentionally standalone so that the
visual style can be tuned without touching the main simulation pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from playwright.async_api import async_playwright
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    raise SystemExit(
        "Playwright is required for this script. Install it with:\n"
        "  pip install playwright\n"
        "  python -m playwright install chromium"
    ) from exc


DEFAULT_COLUMNS: List[str] = [
    "Case_Name",
    "Case_ID",
    "Filler_Volume_Fraction",
    "Sigma_xx_chfem",
    "Sigma_xx_puma",
    "Contact_Ratio",
    "Tunneling_Ratio",
    "Connectivity_Ratio",
    "N_Conductive_Clusters",
    "N_Largest_Cluster_Voxels",
]

RATIO_COLUMNS = {
    "Filler_Volume_Fraction",
    "Contact_Ratio",
    "Tunneling_Ratio",
    "Connectivity_Ratio",
}

CONDUCTIVITY_CANDIDATES = [
    "Sigma_xx_chfem",
    "Sigma_xx_puma",
    "Sigma_xx",
    "Effective_Conductivity",
    "Txx",
]

COUNT_COLUMNS = {
    "N_Conductive_Clusters",
    "N_Largest_Cluster_Voxels",
    "N_Conductive_Candidate_Voxels",
    "N_Filler_Voxels",
    "N_Contact_Voxels",
    "N_Tunnel_Voxels",
}


# Column labels are shortened for dashboard readability.
DISPLAY_LABELS: Dict[str, str] = {
    "Case_Name": "Case",
    "Case_ID": "ID",
    "Filler_Volume_Fraction": "Filler VF",
    "Sigma_xx_chfem": "Sigma xx (chfem)",
    "Sigma_xx_puma": "Sigma xx (PuMA)",
    "Sigma_xx": "Sigma xx",
    "Effective_Conductivity": "Eff. Cond.",
    "Contact_Ratio": "Contact",
    "Tunneling_Ratio": "Tunnel",
    "Connectivity_Ratio": "Connectivity",
    "N_Conductive_Clusters": "Clusters",
    "N_Largest_Cluster_Voxels": "Largest Cluster",
    "N_Conductive_Candidate_Voxels": "Cond. Voxels",
    "N_Filler_Voxels": "Filler Voxels",
    "N_Contact_Voxels": "Contact Voxels",
    "N_Tunnel_Voxels": "Tunnel Voxels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a review dashboard PNG from a microsimflow CSV result file."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output PNG file.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30,
        help="Maximum number of rows to display in the dashboard.",
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        help="Column used for sorting. Defaults to a conductivity column if available.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order. Default is descending.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional explicit list of columns to display.",
    )
    parser.add_argument(
        "--title",
        default="microsimflow results dashboard",
        help="Title shown above the table.",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Optional subtitle. If omitted, a short auto summary is shown.",
    )
    return parser.parse_args()


def choose_columns(df: pd.DataFrame, requested_columns: Optional[Sequence[str]]) -> List[str]:
    if requested_columns:
        return [name for name in requested_columns if name in df.columns]
    return [name for name in DEFAULT_COLUMNS if name in df.columns]


def choose_sort_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested
    for candidate in CONDUCTIVITY_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for candidate in ["Connectivity_Ratio", "Contact_Ratio", "Filler_Volume_Fraction"]:
        if candidate in df.columns:
            return candidate
    return None


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_linear(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    min_value = numeric.min(skipna=True)
    max_value = numeric.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value <= min_value:
        return pd.Series(np.where(numeric.notna(), 1.0, np.nan), index=series.index, dtype=float)
    return (numeric - min_value) / (max_value - min_value)


def normalize_log10(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    positive = numeric.where(numeric > 0)
    if positive.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    logged = np.log10(positive)
    min_value = logged.min(skipna=True)
    max_value = logged.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value <= min_value:
        return pd.Series(np.where(positive.notna(), 1.0, np.nan), index=series.index, dtype=float)
    return (logged - min_value) / (max_value - min_value)


def infer_bar_columns(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, pd.Series]:
    bar_map: Dict[str, pd.Series] = {}
    for name in columns:
        if name in RATIO_COLUMNS:
            bar_map[name] = safe_numeric(df[name]).clip(lower=0.0, upper=1.0)
        elif name in COUNT_COLUMNS:
            bar_map[name] = normalize_linear(df[name])
        elif name in CONDUCTIVITY_CANDIDATES:
            bar_map[name] = normalize_log10(df[name])
    return bar_map


def format_value(name: str, value: object) -> str:
    if pd.isna(value):
        return "-"
    if name in RATIO_COLUMNS:
        return f"{float(value):.3f}"
    if name in COUNT_COLUMNS:
        return f"{int(round(float(value))):,}"
    if name in CONDUCTIVITY_CANDIDATES:
        return f"{float(value):.3e}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def build_table_html(
    df: pd.DataFrame,
    columns: Sequence[str],
    title: str,
    subtitle: str,
) -> str:
    bar_map = infer_bar_columns(df, columns)
    header_html = "".join(
        f"<th>{html.escape(DISPLAY_LABELS.get(name, name))}</th>" for name in columns
    )

    row_chunks: List[str] = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        cell_chunks = [f"<td class='row-index'>{row_index}</td>"]
        for name in columns:
            value = row.get(name)
            text = html.escape(format_value(name, value))
            if name in bar_map:
                bar_value = clamp01(float(bar_map[name].get(row.name, np.nan))) if not pd.isna(bar_map[name].get(row.name, np.nan)) else 0.0
                cell_html = (
                    "<td class='metric-cell'>"
                    f"<div class='bar-track'><div class='bar-fill' style='width:{bar_value * 100:.1f}%'></div></div>"
                    f"<span class='metric-text'>{text}</span>"
                    "</td>"
                )
            else:
                extra_class = " text-cell" if isinstance(value, str) else " number-cell"
                cell_html = f"<td class='{extra_class.strip()}'>{text}</td>"
            cell_chunks.append(cell_html)
        row_chunks.append(f"<tr>{''.join(cell_chunks)}</tr>")

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #ffffff;
    --fg: #1f2937;
    --muted: #6b7280;
    --line: #d1d5db;
    --stripe: #f9fafb;
    --bar-bg: #eef2ff;
    --bar-fill: linear-gradient(90deg, #93c5fd 0%, #60a5fa 100%);
  }}
  html, body {{
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: Arial, Helvetica, sans-serif;
  }}
  .wrap {{
    width: max-content;
    padding: 18px 20px 22px 20px;
  }}
  .title {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .subtitle {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 14px;
  }}
  table {{
    border-collapse: collapse;
    border: 1px solid var(--line);
    font-size: 12px;
    white-space: nowrap;
  }}
  thead th {{
    background: #f3f4f6;
    border-bottom: 1px solid var(--line);
    padding: 8px 10px;
    text-align: left;
    font-weight: 700;
  }}
  tbody td {{
    border-top: 1px solid #eceff1;
    padding: 6px 10px;
    vertical-align: middle;
  }}
  tbody tr:nth-child(even) td {{
    background: var(--stripe);
  }}
  .row-index {{
    color: var(--muted);
    text-align: right;
    min-width: 28px;
  }}
  .text-cell {{
    text-align: left;
    min-width: 120px;
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .number-cell {{
    text-align: right;
    min-width: 88px;
  }}
  .metric-cell {{
    position: relative;
    text-align: right;
    min-width: 120px;
    padding-right: 10px;
  }}
  .bar-track {{
    position: absolute;
    left: 8px;
    right: 8px;
    top: 50%;
    height: 16px;
    transform: translateY(-50%);
    background: var(--bar-bg);
    border-radius: 4px;
    overflow: hidden;
    opacity: 0.95;
  }}
  .bar-fill {{
    height: 100%;
    background: var(--bar-fill);
    border-radius: 4px;
  }}
  .metric-text {{
    position: relative;
    z-index: 1;
    font-variant-numeric: tabular-nums;
  }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="title">{html.escape(title)}</div>
    <div class="subtitle">{html.escape(subtitle)}</div>
    <table>
      <thead>
        <tr><th>#</th>{header_html}</tr>
      </thead>
      <tbody>
        {''.join(row_chunks)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def build_subtitle(csv_path: Path, row_count: int, sort_by: Optional[str], ascending: bool) -> str:
    order = "ascending" if ascending else "descending"
    if sort_by:
        return f"Source: {csv_path.name} | Rows shown: {row_count} | Sorted by: {sort_by} ({order})"
    return f"Source: {csv_path.name} | Rows shown: {row_count}"


async def render_html_to_png(html_content: str, output_path: Path) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(device_scale_factor=2)
        await page.set_content(html_content, wait_until="networkidle")
        await page.screenshot(path=str(output_path), full_page=True)
        await browser.close()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"Input CSV has no rows: {csv_path}")

    columns = choose_columns(df, args.columns)
    if not columns:
        raise SystemExit("No displayable columns were found in the CSV file.")

    sort_by = choose_sort_column(df, args.sort_by)
    if sort_by:
        sorted_df = df.sort_values(by=sort_by, ascending=args.ascending, na_position="last")
    else:
        sorted_df = df.copy()

    dashboard_df = sorted_df.loc[:, columns].head(args.max_rows).reset_index(drop=True)
    subtitle = args.subtitle or build_subtitle(csv_path, len(dashboard_df), sort_by, args.ascending)
    html_content = build_table_html(dashboard_df, columns, args.title, subtitle)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(render_html_to_png(html_content, output_path))
    print(f"Saved dashboard image to: {output_path}")


if __name__ == "__main__":
    main()
