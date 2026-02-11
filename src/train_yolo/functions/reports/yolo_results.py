from pathlib import Path
import pandas as pd
import re

from utils.path_loader import load_paths

paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]
MODELS_DATA_PATH = PROJECT_ROOT / "src" / "train_yolo"


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def to_relative(path: Path) -> str:
    """
    Converte path absoluto para relativo ao PROJECT_ROOT.
    Se não for possível, devolve string normal.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------
# Helpers de parsing
# ---------------------------------------------------------

def parse_model_folder(name: str):
    """
    Ex: 'yolo11m-text-photo'
    -> version='yolo11m', task='text-photo'
    """
    if "-" not in name:
        return name, None

    version, task = name.split("-", 1)
    return version, task


def parse_train_folder(name: str):
    """
    Ex: 'train-150e4b'
    -> epochs=150, batch=4
    """
    match = re.search(r"train-(\d+)e(\d+)b", name)
    if not match:
        return None, None

    epochs = int(match.group(1))
    batch = int(match.group(2))
    return epochs, batch


# ---------------------------------------------------------
# Extração de métricas
# ---------------------------------------------------------

def extract_yolo_results(run_path: Path):
    """
    Extrai métricas principais de um treino YOLO.
    Usa sempre a estrutura baseada em MODELS_DATA_PATH.
    """

    results_file = run_path / "results.csv"
    best_weights = run_path / "weights" / "best.pt"

    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)

    # Melhor epoch baseado no mAP50-95
    best_row = df.loc[df["metrics/mAP50-95(B)"].idxmax()]

    # parsing de pastas
    model_folder = run_path.parent.name
    train_folder = run_path.name

    yolo_version, task = parse_model_folder(model_folder)
    epochs, batch = parse_train_folder(train_folder)

    return {
        "yolo_version": yolo_version,
        "task": task,
        "train_run": train_folder,
        "epochs_config": epochs,
        "batch_config": batch,
        "best_epoch": int(best_row["epoch"].item()),
        "mAP50": best_row["metrics/mAP50(B)"],
        "mAP50-95": best_row["metrics/mAP50-95(B)"],
        "precision": best_row["metrics/precision(B)"],
        "recall": best_row["metrics/recall(B)"],
        "val_box_loss": best_row["val/box_loss"],
        "val_cls_loss": best_row["val/cls_loss"],
        "val_dfl_loss": best_row["val/dfl_loss"],
        "has_best_weights": best_weights.exists(),
        "run_path": to_relative(run_path),  # 🔹 AGORA RELATIVO
    }


# ---------------------------------------------------------
# Recolha global baseada em MODELS_DATA_PATH
# ---------------------------------------------------------

def collect_models_results(models_root: Path = MODELS_DATA_PATH,verbose: bool = True) -> pd.DataFrame:
    """
    Percorre MODELS_DATA_PATH e recolhe todos os treinos YOLO.
    Se verbose=True, imprime um resumo compacto tipo benchmark.
    """

    if verbose:
        print("\n=== YOLO RESULTS SUMMARY ===")
        print(f"Models root: {models_root}\n")


    all_results = []
    info = []

    if not models_root.exists():
        raise FileNotFoundError(f"MODELS_DATA_PATH não existe: {models_root}")

    # Estrutura esperada:
    # MODELS_DATA_PATH /
    #   yolo11m-text-photo /
    #       train-150e4b /
    #           results.csv

    # descobrir info (só para contar + progresso)
    for model_dir in models_root.iterdir():
        if not model_dir.is_dir():
            continue
        for train_dir in model_dir.iterdir():
            if not train_dir.is_dir():
                continue
            if (train_dir / "results.csv").exists():
                info.append(train_dir)

    if verbose:
        print(f"Número de runs com results.csv: {len(info)}\n")

    for i, train_dir in enumerate(info, 1):
        info = extract_yolo_results(train_dir)
        if not info:
            continue

        all_results.append(info)

        if verbose:
            print(
                f"[{i:02d}/{len(info)}] "
                f"{info['yolo_version'].upper():>7} | "
                f"TASK={str(info['task']):<10} | "
                f"TRAIN={info['train_run']:<12} | "
                f"mAP50-95={info['mAP50-95']:.4f} | "
                f"best_epoch={info['best_epoch']}"
            )

    if not all_results:
        if verbose:
            print("\nNenhum results.csv válido encontrado.\n")
        return pd.DataFrame()

    df = pd.DataFrame(all_results).sort_values("mAP50-95", ascending=False).reset_index(drop=True)

    if verbose:
        best = df.iloc[0]
        print("\n=== BEST (by mAP50-95) ===")
        print(
            f"{best['yolo_version'].upper()} | TASK={best['task']} | TRAIN={best['train_run']} | "
            f"mAP50-95={best['mAP50-95']:.4f} | run_path={best['run_path']}"
        )
        print("=========================\n")

    return df
