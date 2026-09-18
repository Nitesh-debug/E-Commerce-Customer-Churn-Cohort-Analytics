"""
run_pipeline.py
Master End-to-End Pipeline Orchestrator

Executes all pipeline stages sequentially:
  1. clean_data.py       -> Data cleaning and SQLite ingestion
  2. run_sql_analytics.py -> SQL RFM segmentation & cohort matrix generation
  3. train_churn_model.py -> Feature engineering & Random Forest churn modeling
  4. generate_dashboard.py -> Interactive HTML Executive Analytics Dashboard
"""

import os
import sys
import time
import subprocess


def run_stage(stage_num, stage_name, script_file):
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num}: {stage_name.upper()}")
    print(f"Executing: {script_file}")
    print("=" * 70)
    start_time = time.time()

    process = subprocess.run(
        [sys.executable, script_file],
        capture_output=False,
        text=True,
    )

    elapsed = time.time() - start_time
    if process.returncode != 0:
        print(f"\n[ERROR] Stage {stage_num} ({script_file}) failed with exit code {process.returncode}.")
        sys.exit(process.returncode)

    print(f"\n[OK] Stage {stage_num} ({stage_name}) completed successfully in {elapsed:.2f}s.")
    return elapsed


def main():
    total_start = time.time()
    print("*" * 70)
    print("END-TO-END E-COMMERCE CUSTOMER CHURN & COHORT ANALYTICS PIPELINE")
    print("*" * 70)

    stages = [
        (1, "Data Cleaning & SQLite Database Ingestion", "clean_data.py"),
        (2, "SQL RFM Segmentation & Cohort Retention Analysis", "run_sql_analytics.py"),
        (3, "Machine Learning Churn Prediction Pipeline", "train_churn_model.py"),
    ]

    # Include dashboard generator if it exists
    if os.path.exists("generate_dashboard.py"):
        stages.append((4, "Interactive Executive Analytics Dashboard", "generate_dashboard.py"))

    stage_timings = {}
    for num, name, script in stages:
        stage_timings[name] = run_stage(num, name, script)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    for name, duration in stage_timings.items():
        print(f"  * {name:<50s}: {duration:.2f}s")
    print("-" * 70)
    print(f"  TOTAL EXECUTION TIME: {total_elapsed:.2f}s")
    print("=" * 70)
    print("[SUCCESS] All pipeline stages executed successfully!")


if __name__ == "__main__":
    main()
