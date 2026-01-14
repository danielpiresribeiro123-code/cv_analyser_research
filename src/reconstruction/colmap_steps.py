from __future__ import annotations

"""
Módulo: colmap_steps.py

Responsabilidade:
    - Encapsular chamadas ao COLMAP (3.13.0+)
    - NÃO usar flags obsoletas (GPU é automática)
"""

from pathlib import Path
import subprocess


# --------------------------------------------------
# Runner interno
# --------------------------------------------------
def _run_colmap(cmd: list[str]) -> None:
    """
    Executa um comando COLMAP.
    Falha explicitamente se o processo retornar erro.
    """
    print(" ".join(cmd))  # útil para debug
    subprocess.run(cmd, check=True)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
def run_feature_extractor(
    colmap_exe: Path,
    images_dir: Path,
    database_path: Path,
    mask_path: Path | None = None,
) -> None:
    """
    Executa o feature_extractor do COLMAP.

    Compatível com COLMAP >= 3.13.0
    (sem flags GPU manuais).
    """

    cmd = [
        str(colmap_exe),
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
    ]

    # Máscaras (opcional)
    if mask_path is not None:
        cmd += ["--ImageReader.mask_path", str(mask_path)]

    _run_colmap(cmd)


# --------------------------------------------------
# Matching
# --------------------------------------------------
def run_exhaustive_matcher(
    colmap_exe: Path,
    database_path: Path,
) -> None:
    """
    Faz matching exaustivo entre todas as imagens.
    """

    _run_colmap([
        str(colmap_exe),
        "exhaustive_matcher",
        "--database_path", str(database_path),
    ])


# --------------------------------------------------
# Mapping (Sparse reconstruction)
# --------------------------------------------------
def run_mapper(
    colmap_exe: Path,
    database_path: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> None:
    """
    Executa o mapeamento sparse do COLMAP.
    """

    sparse_dir.mkdir(parents=True, exist_ok=True)

    _run_colmap([
        str(colmap_exe),
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ])
