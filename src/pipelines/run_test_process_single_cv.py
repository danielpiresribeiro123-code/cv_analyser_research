"""Simple test runner for the CV processing pipeline.

Usage:
    python run_test_pipeline.py <pdf_path> [output_csv]

If <output_csv> is provided, pipeline will save CSV there.
Exit code:
    0 - success with at least one CV page processed
    2 - pipeline completed but no CV pages found
    1 - error
"""
import sys
from pathlib import Path

# Ensure src is on path for imports
try:
    from utils.path_loader import load_paths
except Exception:
    # Try to add project src to sys.path
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / 'src'))
    from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = Path(paths["project_root"]) if isinstance(paths, dict) else Path(paths)
# Make sure src is importable
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pipelines.process_single_cv import process_single_cv


def main(argv):
    if len(argv) < 2:
        print("Usage: python run_test_pipeline.py <pdf_path> [output_csv]")
        return 1

    pdf_path = Path(argv[1])
    out_csv = Path(argv[2]) if len(argv) > 2 else None

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 1

    try:
        df = process_single_cv(pdf_path, save_csv=out_csv, verbose=True)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1

    if df is None or df.empty:
        print("Pipeline finished but no CV pages were detected or extracted.")
        return 2

    print("\nPipeline returned DataFrame with the following columns:")
    
    print(df.columns.tolist())
    print('\nFirst rows:')
    print(df.head().to_string())

    return 0


if __name__ == '__main__':
    exit_code = main(sys.argv)
    sys.exit(exit_code)
