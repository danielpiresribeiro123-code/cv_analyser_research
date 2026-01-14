from pathlib import Path
import sys

# --------------------------------------------------
# Garantir acesso à raiz do projeto
# --------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.path_loader import Paths
from src.reconstruction.colmap_steps import (
    run_feature_extractor,
    run_exhaustive_matcher,
    run_mapper,
)


def run_colmap_pipeline(
    images_dir: Path,
    work_dir: Path,
    colmap_exe: Path | None= None,
) -> None:
    paths = Paths(Path("configs/paths.yaml"))
    if type(colmap_exe) != Path:
        colmap_exe = paths.colmap_executable_cuda
    
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"

    work_dir.mkdir(parents=True, exist_ok=True)

    print("▶ COLMAP | Feature extraction")
    run_feature_extractor(
        colmap_exe=colmap_exe,
        images_dir=images_dir,
        database_path=database_path,
    )

    print("▶ COLMAP | Matching")
    run_exhaustive_matcher(
        colmap_exe=colmap_exe,
        database_path=database_path,
    )

    print("▶ COLMAP | Mapping")
    run_mapper(
        colmap_exe=colmap_exe,
        database_path=database_path,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
    )
