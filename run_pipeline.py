"""
Runs the full pipeline in order, from raw data to dashboard outputs.
Usage: python run_pipeline.py            (all steps)
       python run_pipeline.py --from rank_customers
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    "generate_dataset", "preprocess", "split_and_scale", "train_models", "cross_validate",
    "evaluate_and_segment", "stability_check", "report_model_selection", "tune_model", "check_final_model",
    "shap_analysis", "shap_plots", "rank_customers", "visualize_uplift", "baseline_comparison",
    "roi_simulation", "targeting_comparison",
]


def run(steps, quiet=True):
    start = time.time()
    for i, step in enumerate(steps, 1):
        t0 = time.time()
        print(f"[{i:>2}/{len(steps)}] {step}.py ...", end=" ", flush=True)
        result = subprocess.run([sys.executable, f"{step}.py"], capture_output=quiet, text=True)
        if result.returncode != 0:
            print("FAILED")
            if quiet:
                print(result.stdout[-2000:], result.stderr[-2000:], sep="\n")
            raise SystemExit(f"Pipeline stopped at {step}.py")
        print(f"done ({time.time() - t0:.0f}s)")
    print(f"\nPipeline finished in {(time.time() - start) / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=STEPS, help="start from this step")
    parser.add_argument("--verbose", action="store_true", help="show each script's output")
    args = parser.parse_args()
    steps = STEPS[STEPS.index(args.start):] if args.start else STEPS
    run(steps, quiet=not args.verbose)


if __name__ == "__main__":
    main()