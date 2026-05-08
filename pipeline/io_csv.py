import csv
import os
import re

def parse_chfem_log(log_path):
    """Extract up to 6 diagonal tensor components (General Txx-Txy notation) and computation time"""
    # Initialize with empty strings for all 6 possible components (xx, yy, zz, yz, zx, xy)
    diag = [""] * 6
    total_time = 0.0
    try:
        if not os.path.exists(log_path):
            return diag, 0.0
            
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract the homogenized constitutive matrix block
        mat_match = re.search(r'Homogenized Constitutive Matrix.*?:[^\n]*\n(.*?)\n-{10,}', content, re.DOTALL)
        if mat_match:
            lines = [line.strip() for line in mat_match.group(1).strip().split('\n') if line.strip()]
            # Extract diagonal elements dynamically based on matrix size (3x3 or 6x6)
            for i in range(min(len(lines), 6)):
                parts = lines[i].split()
                if len(parts) > i:
                    diag[i] = float(parts[i])

        # Sum up elapsed time matches
        time_matches = re.findall(r'Elapsed time(?: \(total\))?:\s*([\d\.eE\+\-]+)', content)
        if time_matches:
            total_time = sum(float(t) for t in time_matches)
    except Exception as e:
        print(f"chfem log parsing error: {e}")

    return diag, total_time

def parse_nf_properties(nf_path):
    """Read and parse property map from an existing .nf file"""
    prop_map = {}
    if not os.path.exists(nf_path):
        return prop_map
        
    with open(nf_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_props = False
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith('%properties_of_materials'):
            in_props = True
            continue
        elif line.startswith('%') and in_props:
            in_props = False
            
        if in_props:
            parts = line.split()
            if len(parts) >= 2:
                phase_id = int(parts[0])
                prop_val = " ".join(parts[1:])
                prop_map[phase_id] = prop_val
                
    return prop_map


STRUCTURE_METRIC_COLUMNS = [
    "Contact_Ratio", "Tunneling_Ratio", "Connectivity_Ratio",
    "N_Contact_Voxels", "N_Tunnel_Voxels", "N_Filler_Voxels",
    "N_Conductive_Candidate_Voxels", "N_Largest_Cluster_Voxels", "N_Conductive_Clusters",
    "tau_X", "tau_Y", "tau_Z", "D_eff_X", "D_eff_Y", "D_eff_Z", "tau_Time_s"
]


def structure_metrics_to_csv_fields(metrics, tau_metrics=None):
    """Format structure metrics for stable CSV output."""
    if tau_metrics is None:
        tau_metrics = {}
        
    fields = {
        "Contact_Ratio": f"{metrics.get('contact_ratio', 0):.4f}",
        "Tunneling_Ratio": f"{metrics.get('tunneling_ratio', 0):.4f}",
        "Connectivity_Ratio": f"{metrics.get('connectivity_ratio', 0):.4f}",
        "N_Contact_Voxels": str(metrics.get('n_contact_voxels', 0)),
        "N_Tunnel_Voxels": str(metrics.get('n_tunnel_voxels', 0)),
        "N_Filler_Voxels": str(metrics.get('n_filler_voxels', 0)),
        "N_Conductive_Candidate_Voxels": str(metrics.get('n_conductive_candidate_voxels', 0)),
        "N_Largest_Cluster_Voxels": str(metrics.get('n_largest_cluster_voxels', 0)),
        "N_Conductive_Clusters": str(metrics.get('n_conductive_clusters', 0)),
        "tau_X": str(tau_metrics.get('tau_X', "")),
        "tau_Y": str(tau_metrics.get('tau_Y', "")),
        "tau_Z": str(tau_metrics.get('tau_Z', "")),
        "D_eff_X": str(tau_metrics.get('D_eff_X', "")),
        "D_eff_Y": str(tau_metrics.get('D_eff_Y', "")),
        "D_eff_Z": str(tau_metrics.get('D_eff_Z', "")),
        "tau_Time_s": str(tau_metrics.get('tau_Time_s', ""))
    }
    
    # Safely format floats if calculation succeeded
    for k in ["tau_X", "tau_Y", "tau_Z", "D_eff_X", "D_eff_Y", "D_eff_Z", "tau_Time_s"]:
        if isinstance(tau_metrics.get(k), float):
            fields[k] = f"{tau_metrics[k]:.6g}"
            
    return fields


def ensure_structure_metric_columns(header, rows):
    """Append missing structure-metric columns and pad existing rows."""
    missing = [col for col in STRUCTURE_METRIC_COLUMNS if col not in header]
    if not missing:
        return header, rows

    header = header + missing
    for row in rows:
        row.extend([""] * len(missing))
    return header, rows


def upgrade_existing_csv_log(csv_path):
    """Upgrade an existing CSV log in place when new structure-metric columns are missing."""
    if not os.path.exists(csv_path):
        return

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        rows = list(reader)

    new_header, new_rows = ensure_structure_metric_columns(header, rows)
    if new_header == header:
        return

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)
