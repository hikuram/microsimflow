import csv
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple


EXP_BASENAME_RE = re.compile(
    r"(?P<exp>exp\d+?)_(?P<profile>.+?)_vf(?P<vf>\d+\.\d+)_seed(?P<seed>\d+)_L(?P<stretch>\d+\.\d+)$"
)


def get_next_result_dir(base_name: str) -> str:
    idx = 1
    while True:
        dir_name = f"{base_name}{idx:02d}"
        if not os.path.exists(dir_name):
            return dir_name
        idx += 1


def get_interface_profiles(bg_prop: str, filler_prop: str) -> Dict[str, Dict[str, str]]:
    return {
        "all_filler": {
            "prop_inter": filler_prop,
            "prop_inter2": filler_prop,
        },
        "primary_filler_secondary_bg": {
            "prop_inter": filler_prop,
            "prop_inter2": bg_prop,
        },
        "all_bg": {
            "prop_inter": bg_prop,
            "prop_inter2": bg_prop,
        },
    }


def build_base_command(
    basename: str,
    csv_log: str,
    size: int,
    seed: int,
    solver: str,
    bg_type: str,
    physics_mode: str,
    prop_a: str,
    prop_b: str,
    prop_inter: str,
    prop_inter2: str,
    voxel_size: float = 1e-8,
    phasea_ratio: Optional[float] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    cmd = [
        "python3", "run_pipeline.py",
        "--size", str(size),
        "--voxel_size", str(voxel_size),
        "--bg_type", bg_type,
        "--physics_mode", physics_mode,
        "--solver", solver,
        "--prop_A", prop_a,
        "--prop_B", prop_b,
        "--prop_inter", prop_inter,
        "--prop_inter2", prop_inter2,
        "--basename", basename,
        "--csv_log", csv_log,
        "--seed", str(seed),
    ]
    if phasea_ratio is not None:
        cmd += ["--phaseA_ratio", str(phasea_ratio)]
    if extra_args:
        cmd += extra_args
    return cmd


def run_command(cmd: List[str], dry_run: bool = False) -> int:
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd)
    return completed.returncode


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def parse_experiment_basename(basename: str) -> Dict[str, object]:
    stem = os.path.basename(basename)
    match = EXP_BASENAME_RE.match(stem)
    if not match:
        return {
            "exp_id": "",
            "interface_profile": "",
            "vf": None,
            "seed": None,
            "stretch_ratio_parsed": None,
        }
    return {
        "exp_id": match.group("exp"),
        "interface_profile": match.group("profile"),
        "vf": float(match.group("vf")),
        "seed": int(match.group("seed")),
        "stretch_ratio_parsed": float(match.group("stretch")),
    }


def compute_diag_mean(row: Dict[str, str], prefix: str) -> Optional[float]:
    vals = [
        safe_float(row.get(f"{prefix}_Txx")),
        safe_float(row.get(f"{prefix}_Tyy")),
        safe_float(row.get(f"{prefix}_Tzz")),
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_anisotropy_xy(row: Dict[str, str], prefix: str) -> Optional[float]:
    xx = safe_float(row.get(f"{prefix}_Txx"))
    yy = safe_float(row.get(f"{prefix}_Tyy"))
    if xx is None or yy is None or yy == 0.0:
        return None
    return xx / yy


def compute_log10_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    return abs(math.log10(a) - math.log10(b))


def _load_csv_rows(csv_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not os.path.exists(csv_path):
        return [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv(path: str, header: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_solver_comparison(
    csv_path: str,
    exp_id: str,
    out_dir: str,
    log10_threshold: float,
    mode: str,
) -> Dict[str, str]:
    _, rows = _load_csv_rows(csv_path)
    target_rows = []
    for row in rows:
        basename = row.get("Basename", "")
        parsed = parse_experiment_basename(basename)
        if parsed["exp_id"] != exp_id:
            continue
        row_copy = dict(row)
        row_copy.update(parsed)
        target_rows.append(row_copy)

    if not target_rows:
        return {
            "detail_csv": "",
            "pair_csv": "",
            "verdict_csv": "",
            "report_md": "",
            "n_rows": "0",
        }

    detail_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for row in target_rows:
        chfem_mean = compute_diag_mean(row, "chfem")
        puma_mean = compute_diag_mean(row, "puma")
        log10_diff = compute_log10_diff(chfem_mean, puma_mean)
        chfem_aniso_xy = compute_anisotropy_xy(row, "chfem")
        puma_aniso_xy = compute_anisotropy_xy(row, "puma")
        has_chfem = safe_float(row.get("chfem_Time_s")) is not None
        has_puma = safe_float(row.get("puma_Time_s")) is not None

        verdict = "SKIP"
        if has_chfem and has_puma:
            if mode == "threshold":
                verdict = "PASS" if log10_diff is not None and log10_diff <= log10_threshold else "FAIL"
            else:
                verdict = "CHECK"

        detail_rows.append({
            "exp_id": row["exp_id"],
            "interface_profile": row["interface_profile"],
            "vf": row["vf"],
            "seed": row["seed"],
            "stretch_ratio": row.get("Stretch_Ratio") or row["stretch_ratio_parsed"],
            "Basename": row.get("Basename", ""),
            "Grid_Size": row.get("Grid_Size", ""),
            "Recipe": row.get("Recipe", ""),
            "Contact_Ratio": row.get("Contact_Ratio", ""),
            "Connectivity_Ratio": row.get("Connectivity_Ratio", ""),
            "chfem_Time_s": row.get("chfem_Time_s", ""),
            "puma_Time_s": row.get("puma_Time_s", ""),
            "chfem_Txx": row.get("chfem_Txx", ""),
            "chfem_Tyy": row.get("chfem_Tyy", ""),
            "chfem_Tzz": row.get("chfem_Tzz", ""),
            "puma_Txx": row.get("puma_Txx", ""),
            "puma_Tyy": row.get("puma_Tyy", ""),
            "puma_Tzz": row.get("puma_Tzz", ""),
            "chfem_diag_mean": format_float(chfem_mean),
            "puma_diag_mean": format_float(puma_mean),
            "log10_diff_diag_mean": format_float(log10_diff),
            "chfem_anisotropy_xy": format_float(chfem_aniso_xy),
            "puma_anisotropy_xy": format_float(puma_aniso_xy),
            "pair_verdict": verdict,
        })

        pair_rows.append({
            "exp_id": row["exp_id"],
            "interface_profile": row["interface_profile"],
            "vf": row["vf"],
            "seed": row["seed"],
            "stretch_ratio": row.get("Stretch_Ratio") or row["stretch_ratio_parsed"],
            "chfem_diag_mean": format_float(chfem_mean),
            "puma_diag_mean": format_float(puma_mean),
            "log10_diff_diag_mean": format_float(log10_diff),
            "chfem_anisotropy_xy": format_float(chfem_aniso_xy),
            "puma_anisotropy_xy": format_float(puma_aniso_xy),
            "pair_verdict": verdict,
        })

    detail_csv = os.path.join(out_dir, f"{exp_id}_detail_with_verdict.csv")
    pair_csv = os.path.join(out_dir, f"{exp_id}_pair_summary.csv")
    _write_csv(detail_csv, list(detail_rows[0].keys()), detail_rows)
    _write_csv(pair_csv, list(pair_rows[0].keys()), pair_rows)

    verdict_rows: List[Dict[str, object]] = []
    report_lines = [
        f"# {exp_id} auto summary",
        "",
        f"- source csv: `{csv_path}`",
        f"- rows analyzed: `{len(detail_rows)}`",
        f"- mode: `{mode}`",
        f"- threshold(log10 diff): `{log10_threshold}`",
        "",
        "## group verdicts",
        "",
    ]

    if mode == "threshold":
        grouped: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
        for row in pair_rows:
            key = (str(row["interface_profile"]), float(row["vf"]))
            grouped.setdefault(key, []).append(row)
        for key in sorted(grouped.keys()):
            g_rows = grouped[key]
            diffs = [safe_float(r["log10_diff_diag_mean"]) for r in g_rows]
            diffs = [d for d in diffs if d is not None]
            pass_count = sum(1 for r in g_rows if r["pair_verdict"] == "PASS")
            fail_count = sum(1 for r in g_rows if r["pair_verdict"] == "FAIL")
            skip_count = sum(1 for r in g_rows if r["pair_verdict"] == "SKIP")
            group_verdict = "PASS" if fail_count == 0 and pass_count > 0 else "FAIL"
            avg_diff = sum(diffs) / len(diffs) if diffs else None
            verdict_rows.append({
                "exp_id": exp_id,
                "interface_profile": key[0],
                "vf": f"{key[1]:.4f}",
                "stretch_ratio": "",
                "n_cases": len(g_rows),
                "n_pass": pass_count,
                "n_fail": fail_count,
                "n_skip": skip_count,
                "avg_log10_diff": format_float(avg_diff),
                "group_verdict": group_verdict,
            })
            report_lines.append(
                f"- {key[0]} / vf={key[1]:.4f}: {group_verdict} "
                f"(pass={pass_count}, fail={fail_count}, skip={skip_count}, avg_log10_diff={format_float(avg_diff)})"
            )
    else:
        grouped = {}
        for row in pair_rows:
            key = (str(row["interface_profile"]), float(row["vf"]))
            grouped.setdefault(key, []).append(row)
        for key in sorted(grouped.keys()):
            g_rows = sorted(grouped[key], key=lambda r: float(r["stretch_ratio"]))
            c_vals = [safe_float(r["chfem_anisotropy_xy"]) for r in g_rows]
            p_vals = [safe_float(r["puma_anisotropy_xy"]) for r in g_rows]
            c_vals = [v for v in c_vals if v is not None]
            p_vals = [v for v in p_vals if v is not None]
            c_monotonic = len(c_vals) >= 2 and all(b >= a for a, b in zip(c_vals, c_vals[1:]))
            p_monotonic = len(p_vals) >= 2 and all(b >= a for a, b in zip(p_vals, p_vals[1:]))
            c_delta = c_vals[-1] - c_vals[0] if len(c_vals) >= 2 else None
            p_delta = p_vals[-1] - p_vals[0] if len(p_vals) >= 2 else None
            same_direction = (
                c_delta is not None and p_delta is not None and c_delta > 0.0 and p_delta > 0.0
            )
            group_verdict = "PASS" if c_monotonic and p_monotonic and same_direction else "FAIL"
            verdict_rows.append({
                "exp_id": exp_id,
                "interface_profile": key[0],
                "vf": f"{key[1]:.4f}",
                "stretch_ratio": "all",
                "n_cases": len(g_rows),
                "n_pass": "",
                "n_fail": "",
                "n_skip": "",
                "avg_log10_diff": "",
                "group_verdict": group_verdict,
                "chfem_monotonic": str(c_monotonic),
                "puma_monotonic": str(p_monotonic),
                "chfem_delta_anisotropy_xy": format_float(c_delta),
                "puma_delta_anisotropy_xy": format_float(p_delta),
            })
            report_lines.append(
                f"- {key[0]} / vf={key[1]:.4f}: {group_verdict} "
                f"(chfem_monotonic={c_monotonic}, puma_monotonic={p_monotonic}, "
                f"chfem_delta={format_float(c_delta)}, puma_delta={format_float(p_delta)})"
            )

    verdict_csv = os.path.join(out_dir, f"{exp_id}_group_verdict.csv")
    _write_csv(verdict_csv, list(verdict_rows[0].keys()), verdict_rows)

    report_md = os.path.join(out_dir, f"{exp_id}_auto_summary.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return {
        "detail_csv": detail_csv,
        "pair_csv": pair_csv,
        "verdict_csv": verdict_csv,
        "report_md": report_md,
        "n_rows": str(len(detail_rows)),
    }
