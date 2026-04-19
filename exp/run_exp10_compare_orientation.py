import argparse
import os
from experiment_utils import (
    get_next_result_dir,
    get_interface_profiles,
    build_base_command,
    run_command,
    summarize_solver_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp10: pseudo-orientation benchmark using post-placement stretch with 3 interface profiles"
    )
    parser.add_argument("--solver", default="both", choices=["chfem", "puma", "both", "skip"])
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--vfs", nargs="+", type=float, default=[0.09, 0.12])
    parser.add_argument("--stretch-ratios", nargs="+", type=float, default=[1.0, 1.1, 1.2, 1.4, 1.6])
    parser.add_argument("--fiber-length", type=int, default=32)
    parser.add_argument("--fiber-radius", type=int, default=2)
    parser.add_argument("--poisson-ratio", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run():
    args = parse_args()
    csv_log = "exp10_compare_orientation_results.csv"
    out_dir = get_next_result_dir("result_exp10_")
    os.makedirs(out_dir)

    bg_prop = "1e-4"
    filler_prop = "1e4"
    interface_profiles = get_interface_profiles(bg_prop, filler_prop)

    print(f"Created output directory: {out_dir}")
    print(f"Starting Exp10. Logging to '{csv_log}'")
    print("Note: this script uses post-placement stretch as a pseudo-orientation proxy.")

    for profile_name, props in interface_profiles.items():
        for vf in args.vfs:
            for seed in args.seeds:
                recipe = f"rigidfiber:{vf:.4f}:length={args.fiber_length}:radius={args.fiber_radius}:prop={filler_prop}"
                basename = os.path.join(out_dir, f"exp10_{profile_name}_vf{vf:.2f}_seed{seed}")
                extra_args = [
                    "--stretch_ratios",
                    *[str(x) for x in args.stretch_ratios],
                    "--poisson_ratio", str(args.poisson_ratio),
                    "--recipe",
                    *recipe.split(),
                ]
                cmd = build_base_command(
                    basename=basename,
                    csv_log=csv_log,
                    size=args.size,
                    seed=seed,
                    solver=args.solver,
                    bg_type="single",
                    physics_mode="electrical",
                    prop_a=bg_prop,
                    prop_b=bg_prop,
                    prop_inter=props["prop_inter"],
                    prop_inter2=props["prop_inter2"],
                    extra_args=extra_args,
                )
                rc = run_command(cmd, dry_run=args.dry_run)
                if rc != 0:
                    print(f"Command failed with return code {rc}: {basename}")

    if args.dry_run:
        print("Dry run completed. Aggregation skipped.")
        return

    summary = summarize_solver_comparison(
        csv_path=csv_log,
        exp_id="exp10",
        out_dir=out_dir,
        log10_threshold=0.0,
        mode="orientation",
    )
    print("Auto summary generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run()
