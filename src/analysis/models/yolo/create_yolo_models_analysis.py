import pandas as pd
from pathlib import Path
from utils.path_loader import load_paths
paths = load_paths()

PROJECT_ROOT = paths["project_root"]

TEXT_PHOTO_TEST = paths["data"]["datasets_root"] / "labels" / "test"
SKILL_TABLE_TEST = paths["data"]["skill_tables"] / "labels" / "test"


from analysis.models.yolo.support.yolo_results import collect_models_results
from analysis.models.yolo.support.yolo_speed import collect_models_speed


def merge_yolo_results_and_speed(df_yolo_results: pd.DataFrame,
                                 df_yolo_speed: pd.DataFrame) -> pd.DataFrame:
    """
    Junta métricas de treino (results) com métricas de velocidade (speed)
    usando as chaves:
        - yolo_version
        - task
        - train_run

    Retorna DataFrame combinado.
    """

    if df_yolo_results.empty:
        raise ValueError("df_yolo_results está vazio")

    if df_yolo_speed.empty:
        raise ValueError("df_yolo_speed está vazio")

    required_cols = {"yolo_version", "task", "train_run"}

    if not required_cols.issubset(df_yolo_results.columns):
        raise ValueError("df_yolo_results não contém as colunas necessárias")

    if not required_cols.issubset(df_yolo_speed.columns):
        raise ValueError("df_yolo_speed não contém as colunas necessárias")

    # Merge interno apenas nas runs que existem nos dois
    df_merged = pd.merge(
        df_yolo_results,
        df_yolo_speed,
        on=["yolo_version", "task", "train_run"],
        how="inner",
        suffixes=("", "_speed"),
    )

    df_merged["model_name"] = (
        df_merged["yolo_version"].astype(str)+ "-"+ df_merged["task"].astype(str)+ "-"
        + df_merged["train_run"].str.replace("train-", "", regex=False).astype(str))

    # Ordenação útil para análise científica
    if "mAP50-95" in df_merged.columns:
        df_merged = df_merged.sort_values(
            by=["task", "mAP50-95", "fps"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    if not df_merged["model_name"].is_unique:
        raise ValueError("model_name não é único — verificar duplicados nos runs.")

    df_merged = df_merged.set_index("model_name", drop=False)
    return df_merged

from pathlib import Path
import pandas as pd


def save_dataframe(
    df: pd.DataFrame,
    output_path: Path | str,
    filename: str = "df_yolo_final",
    format: str = "csv",
) -> Path:
    """
    Guarda um DataFrame no caminho escolhido.

    Args:
        df: DataFrame a guardar
        output_path: pasta de destino
        filename: nome do ficheiro (sem extensão)
        format: 'csv' ou 'parquet'

    Returns:
        Caminho completo do ficheiro guardado.
    """

    if df.empty:
        raise ValueError("O DataFrame está vazio, nada para guardar.")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    format = format.lower()

    if format == "csv":
        file_path = output_path / f"{filename}.csv"
        df.to_csv(file_path, index=False)

    elif format == "parquet":
        file_path = output_path / f"{filename}.parquet"
        df.to_parquet(file_path, index=False)

    else:
        raise ValueError("Formato inválido. Usa 'csv' ou 'parquet'.")

    print(f"[SAVE] DataFrame guardado em: {file_path}")

    return file_path

def create_yolo_models_benchmarks(
    device: str | int = "cpu",
    num_analysed_images: int = 150,
    output_dir: Path = PROJECT_ROOT / "src"/"analysis" / "models"/"yolo",
    filename: str = "yolo_models_analysis",
    format: str = "csv",
    text_photo_imgs_path: str | Path= TEXT_PHOTO_TEST,
    skill_table_imgs_path: str | Path= SKILL_TABLE_TEST
) -> pd.DataFrame:
    """
    Pipeline completo de benchmarking YOLO:

    1. Lê métricas de treino (results.csv)
    2. Mede velocidade de inferência
    3. Faz merge dos resultados
    4. Guarda DataFrame final
    5. Retorna DataFrame final
    """

    # métricas de treino
    df_yolo_results = collect_models_results()

    # velocidade
    df_yolo_speed = collect_models_speed(
        device=device,
        num_images=num_analysed_images,
        text_photo_imgs_path= text_photo_imgs_path,
        skill_table_imgs_path= skill_table_imgs_path)

    # merge 
    df_yolo_final = merge_yolo_results_and_speed(df_yolo_results,df_yolo_speed,)

    # guardar
    save_dataframe(
        df_yolo_final,
        output_path=output_dir,
        filename=filename,
        format=format,)

    return df_yolo_final


if __name__ == "__main__":
    create_yolo_models_benchmarks(device=0) #mudar para "cpu" caso não tenhas gpu ou a versão torch não esteja instalada