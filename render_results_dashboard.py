#!/usr/bin/env python3
"""Render a self-contained HTML review dashboard from a CSV results file.

This script reads a CSV file and writes a standalone HTML dashboard with:
- sortable columns
- light and dark theme toggle
- settings modal for filters and column layout
- live row-count summary
- column visibility toggles
- pinned columns that move to the left
- draggable header dividers for column-width resizing

The output is intended for quick interactive review in a browser without
external JavaScript or CSS dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_COLUMN_CANDIDATES: List[str] = [
    "Basename",
    "Grid_Size",
    "Recipe",
    "BG_Type",
    "Mode",
    "Stretch_Ratio",
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

FIXED_BAR_KEYWORDS: Tuple[str, ...] = ("frac",)
LINEAR_BAR_KEYWORDS: Tuple[str, ...] = ("ratio", "time")
LOG_BAR_KEYWORDS: Tuple[str, ...] = ("txx", "tyy", "tzz", "txy", "tyz", "tzx")
COUNT_BAR_KEYWORDS: Tuple[str, ...] = ("voxels", "clusters", "n_")
CATEGORICAL_KEYWORDS: Tuple[str, ...] = ("model", "recipe", "solver", "type", "mode", "grid_size")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>__TITLE__</title>
<style>
:root {
  --bg: #0b1020;
  --panel: #131a2a;
  --panel-2: #172033;
  --panel-3: #1d2940;
  --grid: #25324a;
  --grid-strong: #3a4d71;
  --text: #e5ecf6;
  --muted: #99a7bd;
  --accent: #66b3ff;
  --accent-soft: rgba(102, 179, 255, 0.12);
  --danger-soft: rgba(255, 120, 120, 0.10);
  --shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
  --row-even: rgba(255, 255, 255, 0.00);
  --row-odd: rgba(255, 255, 255, 0.018);
  --row-hover: rgba(102, 179, 255, 0.06);
  --sticky-header: #1a2437;
  --sticky-cell: #141d2e;
  --sticky-cell-odd: #172033;
  --sticky-cell-hover: #1b2940;
  --badge-bg: rgba(255, 255, 255, 0.05);
  --badge-border: rgba(255, 255, 255, 0.08);
  --input-bg: #0f1626;
  --input-border: #2b3a58;
  --overlay: rgba(3, 6, 12, 0.72);
  --bar-fraction: linear-gradient(90deg, rgba(80, 200, 120, 0.55), rgba(46, 139, 87, 0.45));
  --bar-ratio: linear-gradient(90deg, rgba(102, 179, 255, 0.55), rgba(50, 120, 200, 0.45));
  --bar-tensor: linear-gradient(90deg, rgba(199, 126, 224, 0.55), rgba(255, 102, 179, 0.45));
  --bar-count: linear-gradient(90deg, rgba(255, 179, 102, 0.55), rgba(220, 130, 50, 0.45));
  --badge-0-bg: rgba(102, 179, 255, 0.15); --badge-0-fg: #66b3ff; --badge-0-border: rgba(102, 179, 255, 0.28);
  --badge-1-bg: rgba(126, 224, 195, 0.15); --badge-1-fg: #7ee0c3; --badge-1-border: rgba(126, 224, 195, 0.28);
  --badge-2-bg: rgba(255, 179, 102, 0.15); --badge-2-fg: #ffb366; --badge-2-border: rgba(255, 179, 102, 0.28);
  --badge-3-bg: rgba(199, 126, 224, 0.15); --badge-3-fg: #c77ee0; --badge-3-border: rgba(199, 126, 224, 0.28);
  --badge-4-bg: rgba(255, 102, 179, 0.15); --badge-4-fg: #ff66b3; --badge-4-border: rgba(255, 102, 179, 0.28);
}

[data-theme=\"light\"] {
  --bg: #f6f8fb;
  --panel: #ffffff;
  --panel-2: #eef3f8;
  --panel-3: #f6f9fc;
  --grid: #d8e1ec;
  --grid-strong: #aebfd4;
  --text: #1d2735;
  --muted: #6b7a8f;
  --accent: #0067b8;
  --accent-soft: rgba(0, 103, 184, 0.10);
  --danger-soft: rgba(196, 51, 51, 0.08);
  --shadow: 0 10px 24px rgba(34, 52, 84, 0.12);
  --row-even: rgba(255, 255, 255, 0.00);
  --row-odd: rgba(15, 23, 42, 0.018);
  --row-hover: rgba(0, 103, 184, 0.05);
  --sticky-header: #eef3f8;
  --sticky-cell: #ffffff;
  --sticky-cell-odd: #f9fbfd;
  --sticky-cell-hover: #eef6fd;
  --badge-bg: rgba(15, 23, 42, 0.035);
  --badge-border: rgba(15, 23, 42, 0.08);
  --input-bg: #ffffff;
  --input-border: #c6d2e2;
  --overlay: rgba(15, 23, 42, 0.48);
  --bar-fraction: linear-gradient(90deg, rgba(0, 158, 115, 0.5), rgba(0, 110, 80, 0.4));
  --bar-ratio: linear-gradient(90deg, rgba(86, 180, 233, 0.5), rgba(0, 114, 178, 0.4));
  --bar-tensor: linear-gradient(90deg, rgba(204, 121, 167, 0.5), rgba(180, 90, 140, 0.4));
  --bar-count: linear-gradient(90deg, rgba(240, 228, 66, 0.55), rgba(213, 94, 0, 0.38));
  --badge-0-bg: rgba(86, 180, 233, 0.15); --badge-0-fg: #0072b2; --badge-0-border: rgba(86, 180, 233, 0.35);
  --badge-1-bg: rgba(0, 158, 115, 0.14); --badge-1-fg: #009e73; --badge-1-border: rgba(0, 158, 115, 0.30);
  --badge-2-bg: rgba(230, 159, 0, 0.14); --badge-2-fg: #c67e00; --badge-2-border: rgba(230, 159, 0, 0.30);
  --badge-3-bg: rgba(204, 121, 167, 0.14); --badge-3-fg: #b55c93; --badge-3-border: rgba(204, 121, 167, 0.30);
  --badge-4-bg: rgba(213, 94, 0, 0.12); --badge-4-fg: #c45b00; --badge-4-border: rgba(213, 94, 0, 0.28);
}

* {
  box-sizing: border-box;
}

html, body {
  margin: 0;
  min-height: 100%;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
}

button, input {
  font: inherit;
}

.main {
  padding: 20px 24px 28px;
}

.topbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}

.title-wrap {
  min-width: 0;
}

.title {
  margin: 0 0 4px;
  font-size: 24px;
  font-weight: 700;
}

.subtitle {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
}

.summary-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 10px;
}

.summary-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 30px;
  padding: 0 10px;
  border: 1px solid var(--badge-border);
  border-radius: 999px;
  background: var(--badge-bg);
  color: var(--text);
  font-size: 12px;
  font-weight: 600;
}

.summary-chip.muted {
  color: var(--muted);
}

.summary-chip.accent {
  background: var(--accent-soft);
  border-color: rgba(102, 179, 255, 0.26);
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.icon-btn {
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--grid);
  border-radius: 10px;
  background: var(--panel);
  color: var(--text);
  cursor: pointer;
  box-shadow: var(--shadow);
}

.icon-btn:hover {
  border-color: var(--accent);
  background: var(--panel-2);
}

.icon-btn svg {
  width: 18px;
  height: 18px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.table-wrap {
  border: 1px solid var(--grid);
  border-radius: 14px;
  overflow: auto;
  background: var(--panel);
  box-shadow: var(--shadow);
  max-height: calc(100vh - 140px);
}

.dashboard-table {
  width: max-content;
  border-collapse: separate;
  border-spacing: 0;
  table-layout: fixed;
}

.dashboard-table thead th {
  position: sticky;
  top: 0;
  z-index: 20;
  background: var(--sticky-header);
  border-bottom: 1px solid var(--grid);
}

.dashboard-table th,
.dashboard-table td {
  border-right: 1px solid var(--grid);
}

.dashboard-table th:last-child,
.dashboard-table td:last-child {
  border-right: 0;
}

.dashboard-table thead th button {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  text-align: left;
  font-size: 12px;
  font-weight: 700;
}

.dashboard-table thead th button:hover {
  background: rgba(127, 127, 127, 0.08);
}

.dashboard-table thead th {
  overflow: visible;
}

.col-resizer {
  position: absolute;
  top: 0;
  right: -4px;
  width: 8px;
  height: 100%;
  cursor: col-resize;
  z-index: 45;
  touch-action: none;
  display: none;
}

.dashboard-table thead th.resizable .col-resizer {
  display: block;
}

.col-resizer::before {
  content: "";
  position: absolute;
  top: 22%;
  bottom: 22%;
  left: 3px;
  width: 2px;
  border-radius: 999px;
  background: var(--grid-strong);
  opacity: 0;
  transition: opacity 0.14s ease;
}

.dashboard-table thead th:hover .col-resizer::before,
.dashboard-table thead th.resizing .col-resizer::before {
  opacity: 0.9;
}

body.col-resize-active,
body.col-resize-active * {
  cursor: col-resize !important;
  user-select: none !important;
}

.header-text {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  line-height: 1.25;
}

.sort-indicator {
  width: 10px;
  flex-shrink: 0;
  color: var(--muted);
}

.dashboard-table thead th.sorted-asc .sort-indicator::after {
  content: \"^\";
}

.dashboard-table thead th.sorted-desc .sort-indicator::after {
  content: \"v\";
}

.dashboard-table tbody td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--grid);
  font-size: 13px;
  vertical-align: middle;
  white-space: nowrap;
  background: var(--row-even);
}

.dashboard-table tbody tr:nth-child(odd) td {
  background: var(--row-odd);
}

.dashboard-table tbody tr:hover td {
  background: var(--row-hover);
}

.dashboard-table tbody tr:last-child td {
  border-bottom: 0;
}

.dashboard-table td.text-cell,
.dashboard-table td.cat-cell {
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dashboard-table td.num-cell,
.dashboard-table td.bar-cell {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.dashboard-table th.num-header .header-text {
  text-align: right;
  width: 100%;
}

.dashboard-table .pinned {
  position: sticky;
}

.dashboard-table thead .pinned {
  z-index: 30;
  background: var(--sticky-header);
}

.dashboard-table tbody td.pinned {
  z-index: 10;
  background: var(--sticky-cell);
}

.dashboard-table tbody tr:nth-child(odd) td.pinned {
  background: var(--sticky-cell-odd);
}

.dashboard-table tbody tr:hover td.pinned {
  background: var(--sticky-cell-hover);
}

.dashboard-table .pinned.edge {
  box-shadow: inset -1px 0 0 var(--grid-strong);
}

.bar-shell {
  position: relative;
  height: 24px;
  border-radius: 6px;
  overflow: hidden;
  background: rgba(128, 128, 128, 0.08);
  border: 1px solid rgba(128, 128, 128, 0.12);
}

.bar-fill {
  position: absolute;
  inset: 0 auto 0 0;
}

.bar-fraction {
  background: var(--bar-fraction);
}

.bar-ratio {
  background: var(--bar-ratio);
}

.bar-tensor {
  background: var(--bar-tensor);
}

.bar-count {
  background: var(--bar-count);
}

.bar-label {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  width: 100%;
  height: 100%;
  padding: 0 8px;
  font-size: 12px;
}

.badge {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid;
  font-size: 11px;
  font-weight: 700;
  overflow: hidden;
  text-overflow: ellipsis;
}

.badge-0 { background: var(--badge-0-bg); color: var(--badge-0-fg); border-color: var(--badge-0-border); }
.badge-1 { background: var(--badge-1-bg); color: var(--badge-1-fg); border-color: var(--badge-1-border); }
.badge-2 { background: var(--badge-2-bg); color: var(--badge-2-fg); border-color: var(--badge-2-border); }
.badge-3 { background: var(--badge-3-bg); color: var(--badge-3-fg); border-color: var(--badge-3-border); }
.badge-4 { background: var(--badge-4-bg); color: var(--badge-4-fg); border-color: var(--badge-4-border); }

.no-data {
  text-align: center;
  color: var(--muted);
  padding: 24px 12px;
}

.cell-text {
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}

.modal-shell {
  position: fixed;
  inset: 0;
  display: none;
  align-items: center;
  justify-content: center;
  background: var(--overlay);
  z-index: 100;
  padding: 24px;
}

.modal-shell.open {
  display: flex;
}

.modal {
  width: min(1180px, 100%);
  max-height: min(88vh, 980px);
  display: flex;
  flex-direction: column;
  border: 1px solid var(--grid);
  border-radius: 16px;
  background: var(--panel);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 18px;
  border-bottom: 1px solid var(--grid);
  background: var(--panel-2);
}

.modal-title-wrap h2 {
  margin: 0 0 4px;
  font-size: 18px;
}

.modal-title-wrap p {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
}

.modal-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.secondary-btn,
.primary-btn {
  min-height: 34px;
  padding: 0 12px;
  border-radius: 10px;
  cursor: pointer;
}

.secondary-btn {
  border: 1px solid var(--grid);
  background: var(--panel);
  color: var(--text);
}

.secondary-btn:hover {
  border-color: var(--accent);
  background: var(--panel-2);
}

.primary-btn {
  border: 1px solid rgba(102, 179, 255, 0.28);
  background: var(--accent-soft);
  color: var(--text);
}

.primary-btn:hover {
  border-color: var(--accent);
}

.modal-body {
  padding: 18px;
  overflow: auto;
}

.settings-grid {
  display: grid;
  grid-template-columns: minmax(360px, 420px) minmax(520px, 1fr);
  gap: 18px;
}

.panel {
  border: 1px solid var(--grid);
  border-radius: 14px;
  background: var(--panel-3);
  overflow: hidden;
}

.panel-header {
  padding: 12px 14px;
  border-bottom: 1px solid var(--grid);
  background: rgba(127, 127, 127, 0.05);
}

.panel-header h3 {
  margin: 0 0 4px;
  font-size: 14px;
}

.panel-header p {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
}

.panel-body {
  padding: 12px 14px;
}

.column-list,
.filter-list {
  display: grid;
  gap: 10px;
}

.column-row,
.filter-row {
  display: grid;
  gap: 10px;
  align-items: center;
  border: 1px solid var(--grid);
  border-radius: 12px;
  padding: 10px 12px;
  background: var(--panel);
}

.column-row {
  grid-template-columns: minmax(0, 1fr) auto auto;
}

.filter-row {
  grid-template-columns: minmax(180px, 240px) minmax(0, 1fr);
}

.column-name,
.filter-name {
  min-width: 0;
}

.meta-note {
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-size: 11px;
}

.toggle-wrap {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--muted);
}

.toggle-wrap input {
  width: 16px;
  height: 16px;
}

.filter-controls {
  display: grid;
  gap: 8px;
}

.filter-controls.text-filter {
  grid-template-columns: 1fr;
}

.filter-controls.number-filter {
  grid-template-columns: 1fr 1fr;
}

.text-input,
.number-input {
  width: 100%;
  min-height: 34px;
  padding: 0 10px;
  border: 1px solid var(--input-border);
  border-radius: 10px;
  background: var(--input-bg);
  color: var(--text);
}

.text-input::placeholder,
.number-input::placeholder {
  color: var(--muted);
}

.section-stack {
  display: grid;
  gap: 18px;
}

.setting-group {
  display: grid;
  gap: 8px;
}

.setting-label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.setting-label {
  font-size: 13px;
  font-weight: 700;
}

.setting-value {
  color: var(--muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.slider-input {
  width: 100%;
  accent-color: var(--accent);
}

.legend-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.legend-row.spaced {
  margin-bottom: 12px;
}

.legend-chip {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  border: 1px solid var(--badge-border);
  background: var(--badge-bg);
  color: var(--muted);
  font-size: 12px;
}

@media (max-width: 980px) {
  .main {
    padding: 16px;
  }

  .topbar {
    flex-direction: column;
    align-items: stretch;
  }

  .toolbar {
    justify-content: flex-end;
  }

  .settings-grid {
    grid-template-columns: 1fr;
  }

  .filter-row {
    grid-template-columns: 1fr;
  }
}
</style>
</head>
<body data-theme=\"dark\">
<div class=\"main\">
  <div class=\"topbar\">
    <div class=\"title-wrap\">
      <h1 class=\"title\">__TITLE__</h1>
      <p class=\"subtitle\">__SUBTITLE__</p>
      <div class=\"summary-row\" id=\"summary-row\"></div>
    </div>
    <div class=\"toolbar\">
      <button type=\"button\" id=\"settings-btn\" class=\"icon-btn\" title=\"Settings\" aria-label=\"Settings\"></button>
      <button type=\"button\" id=\"theme-btn\" class=\"icon-btn\" title=\"Toggle theme\" aria-label=\"Toggle theme\"></button>
    </div>
  </div>
  <div class=\"table-wrap\">
    <table id=\"dashboard-table\" class=\"dashboard-table\">
      <thead id=\"dashboard-head\"></thead>
      <tbody id=\"dashboard-body\"></tbody>
    </table>
  </div>
</div>

<div id=\"settings-modal\" class=\"modal-shell\" aria-hidden=\"true\">
  <div class=\"modal\" role=\"dialog\" aria-modal=\"true\" aria-labelledby=\"settings-title\">
    <div class=\"modal-header\">
      <div class=\"modal-title-wrap\">
        <h2 id=\"settings-title\">Dashboard Settings</h2>
        <p>Manage filters, column visibility, and pinned columns without cluttering the review table.</p>
      </div>
      <div class=\"modal-actions\">
        <button type=\"button\" id=\"clear-filters-btn\" class=\"secondary-btn\">Clear Filters</button>
        <button type=\"button\" id=\"hide-all-btn\" class=\"secondary-btn\">Hide All</button>
        <button type=\"button\" id=\"reset-layout-btn\" class=\"secondary-btn\">Reset Layout</button>
        <button type=\"button\" id=\"close-settings-btn\" class=\"primary-btn\">Close</button>
      </div>
    </div>
    <div class=\"modal-body\">
      <div class=\"settings-grid\">
        <div class=\"section-stack\">
          <section class=\"panel\">
            <div class=\"panel-header\">
              <h3>Columns</h3>
              <p>Hide columns or pin them to the left side of the table.</p>
            </div>
            <div class=\"panel-body\">
              <div class=\"legend-row spaced\">
                <span class=\"legend-chip\" id=\"columns-summary-chip\"></span>
                <span class=\"legend-chip\" id=\"pin-summary-chip\"></span>
              </div>
              <div id=\"column-list\" class=\"column-list\"></div>
            </div>
          </section>
        </div>
        <section class=\"panel\">
          <div class=\"panel-header\">
            <h3>Filters</h3>
            <p>Apply text contains filters or numeric min and max ranges by column.</p>
          </div>
          <div class=\"panel-body\">
            <div class=\"legend-row spaced\">
              <span class=\"legend-chip\" id=\"rows-summary-chip\"></span>
              <span class=\"legend-chip\" id=\"filters-summary-chip\"></span>
            </div>
            <div id=\"filter-list\" class=\"filter-list\"></div>
          </div>
        </section>
      </div>
    </div>
  </div>
</div>

<script id=\"dashboard-data\" type=\"application/json\">__DASHBOARD_JSON__</script>
<script>
(function () {
  const icons = {
    settings: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.1 2.4 2.6.4-1.9 1.8.5 2.6L12 9.2 9.7 10.2l.5-2.6-1.9-1.8 2.6-.4L12 3z"></path><path d="M4 13.5h6"></path><path d="M14 13.5h6"></path><path d="M7 17.5h10"></path></svg>',
    moon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"></path></svg>',
    sun: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2.5v2.5"></path><path d="M12 19v2.5"></path><path d="M2.5 12H5"></path><path d="M19 12h2.5"></path><path d="M5 5l1.8 1.8"></path><path d="M17.2 17.2L19 19"></path><path d="M19 5l-1.8 1.8"></path><path d="M5 19l1.8-1.8"></path></svg>'
  };

  const payload = JSON.parse(document.getElementById('dashboard-data').textContent);
  const columns = payload.columns;
  const rows = payload.rows;
  const totalRows = rows.length;

  const settingsBtn = document.getElementById('settings-btn');
  const themeBtn = document.getElementById('theme-btn');
  const modal = document.getElementById('settings-modal');
  const closeSettingsBtn = document.getElementById('close-settings-btn');
  const clearFiltersBtn = document.getElementById('clear-filters-btn');
  const hideAllBtn = document.getElementById('hide-all-btn');
  const resetLayoutBtn = document.getElementById('reset-layout-btn');
  const summaryRow = document.getElementById('summary-row');
  const columnList = document.getElementById('column-list');
  const filterList = document.getElementById('filter-list');
  const head = document.getElementById('dashboard-head');
  const body = document.getElementById('dashboard-body');
  const columnsSummaryChip = document.getElementById('columns-summary-chip');
  const pinSummaryChip = document.getElementById('pin-summary-chip');
  const rowsSummaryChip = document.getElementById('rows-summary-chip');
  const filtersSummaryChip = document.getElementById('filters-summary-chip');

  settingsBtn.innerHTML = icons.settings;

  const state = {
    theme: payload.initial_theme || 'dark',
    sortBy: payload.initial_sort_by || null,
    sortDesc: !!payload.initial_descending,
    visible: {},
    pinned: {},
    filters: {},
    columnWidths: {}
  };

  let activeResize = null;

  function initializeState() {
    columns.forEach((col) => {
      state.visible[col.name] = col.default_visible !== false;
      state.pinned[col.name] = !!col.default_pinned;
      state.filters[col.name] = { text: '', min: '', max: '' };
      state.columnWidths[col.name] = col.default_width || 140;
    });
    document.body.setAttribute('data-theme', state.theme);
    updateThemeButton();
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function deterministicColorIndex(text) {
    const raw = String(text || '');
    let sum = 0;
    for (let i = 0; i < raw.length; i += 1) {
      sum += raw.charCodeAt(i);
    }
    return sum % 5;
  }

  function toNumber(value) {
    if (value === null || value === undefined || value === '') {
      return null;
    }
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function formatValue(raw, column) {
    if (raw === null || raw === undefined || raw === '') {
      return '';
    }
    if (column.is_numeric) {
      const value = toNumber(raw);
      if (value === null) {
        return String(raw);
      }
      const lower = column.name.toLowerCase();
      if (lower.includes('time')) {
        return value.toFixed(2) + 's';
      }
      if (column.bar && column.bar.theme === 'count') {
        return Math.round(value).toLocaleString();
      }
      if (column.is_ratio_like) {
        return value.toFixed(4);
      }
      const absValue = Math.abs(value);
      if (absValue === 0) {
        return '0';
      }
      if (absValue >= 10000 || absValue < 1e-4) {
        return value.toExponential(3);
      }
      if (absValue >= 100) {
        return value.toFixed(2);
      }
      if (absValue >= 1) {
        return value.toFixed(3);
      }
      return value.toFixed(4);
    }
    return String(raw);
  }

  function normalizeBar(raw, column) {
    if (!column.bar) {
      return null;
    }
    const value = toNumber(raw);
    if (value === null) {
      return null;
    }
    const meta = column.bar;
    if (meta.mode === 'fixed') {
      return Math.max(0, Math.min(1, value));
    }
    if (meta.mode === 'log') {
      if (value <= 0 || meta.lo === null || meta.hi === null) {
        return null;
      }
      if (meta.lo === meta.hi) {
        return 1;
      }
      const logValue = Math.log10(value);
      return Math.max(0, Math.min(1, (logValue - meta.lo) / (meta.hi - meta.lo)));
    }
    if (meta.lo === null || meta.hi === null) {
      return null;
    }
    if (meta.lo === meta.hi) {
      return 1;
    }
    return Math.max(0, Math.min(1, (value - meta.lo) / (meta.hi - meta.lo)));
  }

  function getOrderedVisibleColumns() {
    const visible = columns.filter((col) => state.visible[col.name]);
    const pinned = visible.filter((col) => state.pinned[col.name]);
    const normal = visible.filter((col) => !state.pinned[col.name]);
    return pinned.concat(normal);
  }

  function countHiddenColumns() {
    return columns.filter((col) => !state.visible[col.name]).length;
  }

  function countPinnedVisibleColumns() {
    return columns.filter((col) => state.visible[col.name] && state.pinned[col.name]).length;
  }

  function countActiveFilters() {
    let count = 0;
    columns.forEach((col) => {
      const filter = state.filters[col.name];
      if (!filter) {
        return;
      }
      if (col.is_numeric) {
        if (String(filter.min).trim() !== '' || String(filter.max).trim() !== '') {
          count += 1;
        }
      } else if (String(filter.text).trim() !== '') {
        count += 1;
      }
    });
    return count;
  }

  function getFilteredRows() {
    return rows.filter((row) => {
      for (const col of columns) {
        const filter = state.filters[col.name];
        if (!filter) {
          continue;
        }
        const raw = row[col.name];
        if (col.is_numeric) {
          const hasMin = String(filter.min).trim() !== '';
          const hasMax = String(filter.max).trim() !== '';
          if (!hasMin && !hasMax) {
            continue;
          }
          const numericValue = toNumber(raw);
          if (numericValue === null) {
            return false;
          }
          if (hasMin && numericValue < Number(filter.min)) {
            return false;
          }
          if (hasMax && numericValue > Number(filter.max)) {
            return false;
          }
        } else {
          const needle = String(filter.text || '').trim().toLowerCase();
          if (!needle) {
            continue;
          }
          const haystack = String(raw === null || raw === undefined ? '' : raw).toLowerCase();
          if (!haystack.includes(needle)) {
            return false;
          }
        }
      }
      return true;
    });
  }

  function compareRows(a, b, column) {
    const av = a[column.name];
    const bv = b[column.name];
    if (column.is_numeric) {
      const an = toNumber(av);
      const bn = toNumber(bv);
      if (an === null && bn === null) {
        return 0;
      }
      if (an === null) {
        return 1;
      }
      if (bn === null) {
        return -1;
      }
      return an - bn;
    }
    const as = String(av === null || av === undefined ? '' : av).toLowerCase();
    const bs = String(bv === null || bv === undefined ? '' : bv).toLowerCase();
    if (as < bs) {
      return -1;
    }
    if (as > bs) {
      return 1;
    }
    return 0;
  }

  function getSortedRows(filteredRows) {
    if (!state.sortBy) {
      return filteredRows.slice();
    }
    const column = columns.find((col) => col.name === state.sortBy);
    if (!column) {
      return filteredRows.slice();
    }
    return filteredRows.slice().sort((a, b) => {
      const result = compareRows(a, b, column);
      return state.sortDesc ? -result : result;
    });
  }

  function renderSummary(filteredCount) {
    const hiddenCount = countHiddenColumns();
    const pinnedCount = countPinnedVisibleColumns();
    const activeFilters = countActiveFilters();

    const chips = [];
    chips.push('<span class="summary-chip"><strong>' + filteredCount.toLocaleString() + '</strong> / ' + totalRows.toLocaleString() + ' rows</span>');
    if (activeFilters > 0) {
      chips.push('<span class="summary-chip accent">Filtered: ' + activeFilters + '</span>');
    }
    if (hiddenCount > 0) {
      chips.push('<span class="summary-chip">Hidden columns: ' + hiddenCount + '</span>');
    }
    if (pinnedCount > 0) {
      chips.push('<span class="summary-chip">Pinned columns: ' + pinnedCount + '</span>');
    }
    if (chips.length === 1) {
      chips.push('<span class="summary-chip muted">Click a header to sort. Use settings for filters and layout.</span>');
    }
    summaryRow.innerHTML = chips.join('');

    columnsSummaryChip.textContent = 'Visible columns: ' + (columns.length - hiddenCount) + ' / ' + columns.length;
    pinSummaryChip.textContent = 'Pinned columns: ' + pinnedCount;
    rowsSummaryChip.textContent = 'Rows shown: ' + filteredCount + ' / ' + totalRows;
    filtersSummaryChip.textContent = activeFilters > 0 ? ('Active filters: ' + activeFilters) : 'Active filters: 0';
  }

  function getMinColumnWidth(column) {
    if (column.bar) {
      return 112;
    }
    if (column.is_numeric) {
      return 92;
    }
    return 120;
  }

  function getColumnWidth(column) {
    const fallback = column.default_width || 140;
    const width = state.columnWidths[column.name];
    return Math.max(getMinColumnWidth(column), Math.round(width || fallback));
  }

  function isResizableColumn(column) {
    return !column.is_numeric && !column.bar;
  }

  function renderHeader(orderedColumns) {
    const headerCells = orderedColumns.map((col) => {
      const classes = [];
      if (col.is_numeric || col.bar) {
        classes.push('num-header');
      }
      if (state.pinned[col.name]) {
        classes.push('pinned');
      }
      if (state.sortBy === col.name) {
        classes.push(state.sortDesc ? 'sorted-desc' : 'sorted-asc');
      }
      if (isResizableColumn(col)) {
        classes.push('resizable');
      }
      const width = getColumnWidth(col);
      return '<th class="' + classes.join(' ') + '" data-column="' + escapeHtml(col.name) + '" style="width:' + width + 'px;min-width:' + width + 'px;max-width:' + width + 'px;">'
        + '<button type="button" data-sort-column="' + escapeHtml(col.name) + '">'
        + '<span class="header-text">' + escapeHtml(col.name) + '</span>'
        + '<span class="sort-indicator"></span>'
        + '</button>'
        + (isResizableColumn(col) ? '<span class="col-resizer" data-resize-column="' + escapeHtml(col.name) + '" title="Drag to resize"></span>' : '')
        + '</th>';
    });
    head.innerHTML = '<tr>' + headerCells.join('') + '</tr>';

    head.querySelectorAll('button[data-sort-column]').forEach((button) => {
      button.addEventListener('click', () => {
        if (activeResize && activeResize.columnName) {
          return;
        }
        const columnName = button.getAttribute('data-sort-column');
        if (state.sortBy === columnName) {
          state.sortDesc = !state.sortDesc;
        } else {
          state.sortBy = columnName;
          state.sortDesc = false;
        }
        render();
      });
    });

    head.querySelectorAll('[data-resize-column]').forEach((handle) => {
      handle.addEventListener('pointerdown', (event) => {
        startColumnResize(event, handle.getAttribute('data-resize-column'));
      });
    });
  }

  function applyColumnWidthToDom(columnName, width) {
    const th = head.querySelector('th[data-column="' + cssEscape(columnName) + '"]');
    if (!th) {
      return;
    }
    const normalizedWidth = Math.round(width);
    th.style.width = normalizedWidth + 'px';
    th.style.minWidth = normalizedWidth + 'px';
    th.style.maxWidth = normalizedWidth + 'px';

    const orderedColumns = getOrderedVisibleColumns();
    const colIndex = orderedColumns.findIndex((col) => col.name === columnName);
    if (colIndex < 0) {
      updatePinnedOffsets();
      return;
    }

    body.querySelectorAll('tr').forEach((tr) => {
      const td = tr.children[colIndex];
      if (!td) {
        return;
      }
      td.style.width = normalizedWidth + 'px';
      td.style.minWidth = normalizedWidth + 'px';
      td.style.maxWidth = normalizedWidth + 'px';
    });

    updatePinnedOffsets();
  }

  function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === 'function') {
      return window.CSS.escape(value);
    }
    return String(value).replace(/"/g, '\\"');
  }

  function startColumnResize(event, columnName) {
    event.preventDefault();
    event.stopPropagation();
    const column = columns.find((col) => col.name === columnName);
    if (!column || !isResizableColumn(column)) {
      return;
    }
    const th = head.querySelector('th[data-column="' + cssEscape(columnName) + '"]');
    if (!th) {
      return;
    }

    activeResize = {
      columnName: columnName,
      startX: event.clientX,
      startWidth: th.getBoundingClientRect().width,
      minWidth: getMinColumnWidth(column),
      pointerId: event.pointerId
    };

    th.classList.add('resizing');
    document.body.classList.add('col-resize-active');

    const onPointerMove = function (moveEvent) {
      if (!activeResize || moveEvent.pointerId !== activeResize.pointerId) {
        return;
      }
      const nextWidth = Math.max(activeResize.minWidth, activeResize.startWidth + (moveEvent.clientX - activeResize.startX));
      state.columnWidths[columnName] = Math.round(nextWidth);
      applyColumnWidthToDom(columnName, nextWidth);
    };

    const stopResize = function (endEvent) {
      if (!activeResize || endEvent.pointerId !== activeResize.pointerId) {
        return;
      }
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', stopResize);
      document.removeEventListener('pointercancel', stopResize);
      th.classList.remove('resizing');
      document.body.classList.remove('col-resize-active');
      activeResize = null;
    };

    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', stopResize);
    document.addEventListener('pointercancel', stopResize);
  }

  function renderCell(raw, column) {
    const title = raw === null || raw === undefined ? '' : String(raw);
    const display = formatValue(raw, column);
    const classes = [];
    if (column.bar) {
      classes.push('bar-cell');
    } else if (column.is_numeric) {
      classes.push('num-cell');
    } else if (column.is_categorical) {
      classes.push('cat-cell');
    } else {
      classes.push('text-cell');
    }
    if (state.pinned[column.name]) {
      classes.push('pinned');
    }

    if (column.bar) {
      const ratio = normalizeBar(raw, column);
      const widthPercent = ratio === null ? 0 : Math.max(0, Math.min(100, ratio * 100));
      return '<td class="' + classes.join(' ') + '" title="' + escapeHtml(title) + '">'
        + '<div class="bar-shell">'
        + '<div class="bar-fill bar-' + escapeHtml(column.bar.theme) + '" style="width:' + widthPercent.toFixed(2) + '%"></div>'
        + '<span class="bar-label">' + escapeHtml(display) + '</span>'
        + '</div>'
        + '</td>';
    }

    let content = '<span class="cell-text">' + escapeHtml(display) + '</span>';
    if (column.name.toLowerCase() === 'basename' && title.indexOf('/') !== -1) {
      const shortName = title.split('/').pop();
      content = '<span class="cell-text">' + escapeHtml(shortName) + '</span>';
    } else if (column.is_categorical && display) {
      const badgeIndex = deterministicColorIndex(display);
      content = '<span class="badge badge-' + badgeIndex + '">' + escapeHtml(display) + '</span>';
    }

    return '<td class="' + classes.join(' ') + '" title="' + escapeHtml(title) + '">' + content + '</td>';
  }

  function renderBody(orderedColumns, sortedRows) {
    if (!sortedRows.length) {
      body.innerHTML = '<tr><td class="no-data" colspan="' + Math.max(1, orderedColumns.length) + '">No rows match the current filters.</td></tr>';
      return;
    }
    const rowsHtml = sortedRows.map((row) => {
      const cells = orderedColumns.map((col) => renderCell(row[col.name], col)).join('');
      return '<tr>' + cells + '</tr>';
    });
    body.innerHTML = rowsHtml.join('');
  }

  function updatePinnedOffsets() {
    const orderedColumns = getOrderedVisibleColumns();
    const pinnedNames = orderedColumns.filter((col) => state.pinned[col.name]).map((col) => col.name);
    const thMap = {};
    const bodyRows = Array.from(body.querySelectorAll('tr'));
    head.querySelectorAll('th').forEach((th) => {
      const name = th.getAttribute('data-column');
      thMap[name] = th;
    });

    let leftOffset = 0;
    pinnedNames.forEach((name, idx) => {
      const th = thMap[name];
      if (!th) {
        return;
      }
      th.style.left = leftOffset + 'px';
      th.classList.add('pinned');
      th.classList.toggle('edge', idx === pinnedNames.length - 1);
      const width = th.offsetWidth;
      bodyRows.forEach((tr) => {
        const rowCells = Array.from(tr.children);
        const colIndex = orderedColumns.findIndex((col) => col.name === name);
        const td = rowCells[colIndex];
        if (!td) {
          return;
        }
        td.style.left = leftOffset + 'px';
        td.classList.add('pinned');
        td.classList.toggle('edge', idx === pinnedNames.length - 1);
      });
      leftOffset += width;
    });
  }

  function buildColumnControls() {
    const columnItems = columns.map((col) => {
      const typeLabel = col.is_numeric ? 'numeric' : 'text';
      return '<div class="column-row">'
        + '<div class="column-name"><strong>' + escapeHtml(col.name) + '</strong><span class="meta-note">' + typeLabel + (col.bar ? ' | data bar' : '') + '</span></div>'
        + '<label class="toggle-wrap"><input type="checkbox" data-visible-column="' + escapeHtml(col.name) + '">Visible</label>'
        + '<label class="toggle-wrap"><input type="checkbox" data-pinned-column="' + escapeHtml(col.name) + '">Pinned</label>'
        + '</div>';
    });
    columnList.innerHTML = columnItems.join('');

    columnList.querySelectorAll('input[data-visible-column]').forEach((input) => {
      input.addEventListener('change', () => {
        const name = input.getAttribute('data-visible-column');
        state.visible[name] = input.checked;
        if (!input.checked) {
          state.pinned[name] = false;
        }
        render();
      });
    });

    columnList.querySelectorAll('input[data-pinned-column]').forEach((input) => {
      input.addEventListener('change', () => {
        const targetName = input.getAttribute('data-pinned-column');
        if (!state.visible[targetName]) {
          input.checked = false;
          return;
        }
        state.pinned[targetName] = input.checked;
        render();
      });
    });
  }

  function buildFilterControls() {
    const filterItems = columns.map((col) => {
      if (col.is_numeric) {
        return '<div class="filter-row">'
          + '<div class="filter-name"><strong>' + escapeHtml(col.name) + '</strong><span class="meta-note">Numeric range filter</span></div>'
          + '<div class="filter-controls number-filter">'
          + '<input class="number-input" type="number" step="any" placeholder="Min" data-filter-min="' + escapeHtml(col.name) + '">'
          + '<input class="number-input" type="number" step="any" placeholder="Max" data-filter-max="' + escapeHtml(col.name) + '">'
          + '</div>'
          + '</div>';
      }
      return '<div class="filter-row">'
        + '<div class="filter-name"><strong>' + escapeHtml(col.name) + '</strong><span class="meta-note">Text contains filter</span></div>'
        + '<div class="filter-controls text-filter">'
        + '<input class="text-input" type="text" placeholder="Contains..." data-filter-text="' + escapeHtml(col.name) + '">'
        + '</div>'
        + '</div>';
    });
    filterList.innerHTML = filterItems.join('');

    filterList.querySelectorAll('input[data-filter-text]').forEach((input) => {
      input.addEventListener('input', () => {
        const name = input.getAttribute('data-filter-text');
        state.filters[name].text = input.value;
        render();
      });
    });

    filterList.querySelectorAll('input[data-filter-min]').forEach((input) => {
      input.addEventListener('input', () => {
        const name = input.getAttribute('data-filter-min');
        state.filters[name].min = input.value;
        render();
      });
    });

    filterList.querySelectorAll('input[data-filter-max]').forEach((input) => {
      input.addEventListener('input', () => {
        const name = input.getAttribute('data-filter-max');
        state.filters[name].max = input.value;
        render();
      });
    });
  }

  function syncControlState() {
    columnList.querySelectorAll('input[data-visible-column]').forEach((input) => {
      const name = input.getAttribute('data-visible-column');
      input.checked = !!state.visible[name];
    });

    columnList.querySelectorAll('input[data-pinned-column]').forEach((input) => {
      const name = input.getAttribute('data-pinned-column');
      input.checked = !!state.pinned[name];
      input.disabled = !state.visible[name];
    });

    filterList.querySelectorAll('input[data-filter-text]').forEach((input) => {
      const name = input.getAttribute('data-filter-text');
      if (document.activeElement !== input) {
        input.value = state.filters[name].text;
      }
    });

    filterList.querySelectorAll('input[data-filter-min]').forEach((input) => {
      const name = input.getAttribute('data-filter-min');
      if (document.activeElement !== input) {
        input.value = state.filters[name].min;
      }
    });

    filterList.querySelectorAll('input[data-filter-max]').forEach((input) => {
      const name = input.getAttribute('data-filter-max');
      if (document.activeElement !== input) {
        input.value = state.filters[name].max;
      }
    });
  }

  function render() {
    const orderedColumns = getOrderedVisibleColumns();
    const filteredRows = getFilteredRows();
    const sortedRows = getSortedRows(filteredRows);
    if (!orderedColumns.length) {
      head.innerHTML = '';
      body.innerHTML = '<tr><td class="no-data" colspan="1">No visible columns selected. Re-enable columns in settings.</td></tr>';
      renderSummary(filteredRows.length);
      syncControlState();
      return;
    }
    renderHeader(orderedColumns);
    renderBody(orderedColumns, sortedRows);
    updatePinnedOffsets();
    renderSummary(filteredRows.length);
    syncControlState();
  }

  function updateThemeButton() {
    themeBtn.innerHTML = state.theme === 'dark' ? icons.sun : icons.moon;
    themeBtn.title = state.theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    themeBtn.setAttribute('aria-label', themeBtn.title);
  }

  function openModal() {
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
  }

  function closeModal() {
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
  }

  settingsBtn.addEventListener('click', openModal);
  closeSettingsBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && modal.classList.contains('open')) {
      closeModal();
    }
  });

  themeBtn.addEventListener('click', () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', state.theme);
    updateThemeButton();
  });

  clearFiltersBtn.addEventListener('click', () => {
    columns.forEach((col) => {
      state.filters[col.name] = { text: '', min: '', max: '' };
    });
    render();
  });

  hideAllBtn.addEventListener('click', () => {
    columns.forEach((col) => {
      state.visible[col.name] = false;
      state.pinned[col.name] = false;
    });
    render();
  });

  resetLayoutBtn.addEventListener('click', () => {
    columns.forEach((col) => {
      state.visible[col.name] = col.default_visible !== false;
      state.pinned[col.name] = !!col.default_pinned;
      state.columnWidths[col.name] = col.default_width || 140;
    });
    render();
  });

  initializeState();
  buildColumnControls();
  buildFilterControls();
  render();
})();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a self-contained HTML review dashboard from a CSV file."
    )
    parser.add_argument("--csv", required=True, help="Input CSV file path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output HTML file path. If omitted, the CSV suffix is replaced with .html.",
    )
    parser.add_argument("--columns", nargs="*", default=None, help="Optional list of columns to display.")
    parser.add_argument("--sort-by", default=None, help="Optional column name used for initial sorting.")
    parser.add_argument("--descending", action="store_true", help="Use descending order for the initial sort.")
    parser.add_argument("--max-rows", type=int, default=2000, help="Maximum number of rows to include.")
    parser.add_argument("--title", default="Simulation Results Dashboard", help="Dashboard title.")
    parser.add_argument(
        "--subtitle",
        default="Interactive summary with filters, column layout, and pinned columns.",
        help="Dashboard subtitle.",
    )
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
    if "Recipe" in df.columns and "Recipe" in columns:
        split_df = df["Recipe"].fillna("").astype(str).str.split(r"\s+", expand=True)
        if split_df.shape[1] > 0:
            recipe_cols = [f"Recipe_{i + 1}" for i in range(split_df.shape[1])]
            split_df.columns = recipe_cols
            df = pd.concat([df.drop(columns=["Recipe"]), split_df], axis=1)
            idx = columns.index("Recipe")
            columns = columns[:idx] + recipe_cols + columns[idx + 1 :]
    return df, columns


def apply_initial_sort(df: pd.DataFrame, sort_by: Optional[str], descending: bool) -> pd.DataFrame:
    if sort_by and sort_by in df.columns:
        return df.sort_values(by=sort_by, ascending=not descending, na_position="last")
    return df


def detect_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, bool]:
    result: Dict[str, bool] = {}
    for col in columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        result[col] = bool(converted.notna().any())
    return result


def classify_bar_attributes(column_name: str) -> Optional[Tuple[str, str]]:
    lower = column_name.lower()
    if any(key in lower for key in COUNT_BAR_KEYWORDS):
        return ("log", "count")
    if any(key in lower for key in LOG_BAR_KEYWORDS):
        return ("log", "tensor")
    if any(key in lower for key in FIXED_BAR_KEYWORDS):
        return ("fixed", "fraction")
    if any(key in lower for key in LINEAR_BAR_KEYWORDS):
        theme = "count" if "time" in lower else "ratio"
        return ("linear", theme)
    return None


def is_categorical(column_name: str) -> bool:
    return any(key in column_name.lower() for key in CATEGORICAL_KEYWORDS)


def is_ratio_like(column_name: str) -> bool:
    lower_name = column_name.lower()
    return (
        "ratio" in lower_name
        or "fraction" in lower_name
        or "frac" in lower_name
        or lower_name.startswith("vf")
    )


def build_bar_meta(series: pd.Series, mode: str, theme: str) -> Dict[str, Optional[float]]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if mode == "fixed":
        return {"mode": mode, "theme": theme, "lo": 0.0, "hi": 1.0}
    if valid.empty:
        return {"mode": mode, "theme": theme, "lo": None, "hi": None}
    if mode == "log":
        positive = valid[valid > 0]
        if positive.empty:
            return {"mode": mode, "theme": theme, "lo": None, "hi": None}
        log_values = positive.map(math.log10)
        return {"mode": mode, "theme": theme, "lo": float(log_values.min()), "hi": float(log_values.max())}
    return {"mode": mode, "theme": theme, "lo": float(valid.min()), "hi": float(valid.max())}


def choose_default_width(column_name: str, is_numeric: bool, has_bar: bool) -> int:
    lower = column_name.lower()
    if has_bar:
        return 150
    if is_numeric:
        return 110
    if lower == "basename":
        return 220
    if lower.startswith("recipe_"):
        return 280
    if lower in {"grid_size", "bg_type", "mode"}:
        return 120
    return 180


def build_columns_meta(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, object]]:
    numeric_map = detect_numeric_columns(df, columns)
    metadata: List[Dict[str, object]] = []
    for idx, col in enumerate(columns):
        attrs = classify_bar_attributes(col) if numeric_map[col] else None
        bar = None
        if attrs is not None:
            mode, theme = attrs
            bar = build_bar_meta(df[col], mode, theme)
        metadata.append(
            {
                "name": col,
                "is_numeric": numeric_map[col],
                "is_categorical": is_categorical(col) and not numeric_map[col],
                "is_ratio_like": is_ratio_like(col),
                "bar": bar,
                "default_width": choose_default_width(col, numeric_map[col], bar is not None),
                "default_visible": True,
                "default_pinned": idx == 0 and col == "Basename",
            }
        )
    return metadata


def json_safe_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return str(value)


def build_rows_payload(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        record: Dict[str, object] = {}
        for col in columns:
            record[col] = json_safe_value(row[col])
        rows.append(record)
    return rows


def drop_empty_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    drop_empty_columns_enabled: bool = True,
) -> List[str]:
    if not drop_empty_columns_enabled:
        return [col for col in columns if col in df.columns]

    filtered: List[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if series.empty:
            continue
        non_empty_mask = series.notna() & (series.astype(str).str.strip() != "")
        if non_empty_mask.any():
            filtered.append(col)
    return filtered


def build_dashboard_html(
    df: pd.DataFrame,
    columns: Sequence[str],
    title: str,
    subtitle: str,
    initial_sort_by: Optional[str],
    initial_descending: bool,
) -> str:
    payload = {
        "columns": build_columns_meta(df, columns),
        "rows": build_rows_payload(df, columns),
        "initial_theme": "dark",
        "initial_sort_by": initial_sort_by if initial_sort_by in columns else None,
        "initial_descending": bool(initial_descending),
    }
    json_payload = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    html_text = HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
    html_text = html_text.replace("__SUBTITLE__", html.escape(subtitle))
    html_text = html_text.replace("__DASHBOARD_JSON__", json_payload)
    return html_text



def render_dashboard_from_csv(
    csv_path: str,
    output_path: str,
    columns: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = None,
    descending: bool = False,
    max_rows: int = 2000,
    title: str = "Simulation Results Dashboard",
    subtitle: str = "Interactive summary with filters, column layout, and pinned columns.",
    drop_empty_columns_enabled: bool = True,
) -> None:
    df = read_csv(Path(csv_path))
    selected_columns = select_columns(df, columns)
    df, selected_columns = split_recipe_column(df, selected_columns)
    selected_columns = drop_empty_columns(df, selected_columns, drop_empty_columns_enabled=drop_empty_columns_enabled)
    df = apply_initial_sort(df, sort_by, descending)
    df = df.loc[:, selected_columns].head(max_rows).copy()

    html_text = build_dashboard_html(
        df=df,
        columns=selected_columns,
        title=title,
        subtitle=subtitle,
        initial_sort_by=sort_by,
        initial_descending=descending,
    )
    Path(output_path).write_text(html_text, encoding="utf-8")



def main() -> None:
    args = parse_args()

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.csv).with_suffix(".html"))
    if Path(output_path).suffix.lower() != ".html":
        raise ValueError(f"Output file must use a .html suffix: {output_path}")

    render_dashboard_from_csv(
        csv_path=args.csv,
        output_path=output_path,
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
