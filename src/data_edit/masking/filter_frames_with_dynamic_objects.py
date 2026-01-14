from pathlib import Path
from typing import List
from .dynamic_presence import DynamicPresenceDetector


def filter_frames_with_dynamic_objects(
    frames_dir: Path,
    detector: DynamicPresenceDetector,
    delete: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """
    Filtra frames que contêm objetos dinâmicos.

    Args:
        frames_dir:
            Diretório com frames.

        detector:
            Instância de DynamicPresenceDetector.

        delete:
            Se True, apaga frames SEM objetos dinâmicos.
            Se False, apenas ignora.

        verbose:
            Mostra informação no terminal.

    Returns:
        Lista de frames que contêm objetos dinâmicos.
    """
    valid_frames = []

    for img_path in sorted(frames_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        has_dynamic = detector.has_dynamic_objects(img_path)

        if has_dynamic:
            valid_frames.append(img_path)
        else:
            if verbose:
                print(f"⏭ Sem objetos dinâmicos: {img_path.name}")

            if delete:
                img_path.unlink()

    if verbose:
        print(f"\n✔ Frames com objetos dinâmicos: {len(valid_frames)}")

    return valid_frames
