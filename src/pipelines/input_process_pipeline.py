
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys
import shutil

from utils.path_loader import load_paths
paths = load_paths()

# ==================================================
# Paths
# ==================================================
# Path do projeto
PROJECT_ROOT = paths["project_root"]

# Inputs video/imagens
INPUT_VIDEO_ROOT= paths["data"]["input"]["video"]
INPUT_IMAGES_ROOT= paths["data"]["input"]["images"]

# Frames caracterizadas
OUTPUT_FRAMES = paths["data"]["frames"]


# ==================================================
# Extras Functions
# ==================================================
from src.capture_video.video_recorder import record_video
from data_edit.video_to_frames import video_to_frames
from utils.pipeline_manifest import PipelineManifest
# from segmentation.matanyone_runner import run_matanyone

def _confirm_step(msg: str, auto_run: bool) -> bool:
    return auto_run or input(f"{msg} [y/N]: ").lower() == "y"



# ==================================================
# Função/pipeline
# ==================================================

def run_process_pipeline(
    # ==================================================
    # MODO
    # ==================================================
    input_data_type: str = "video",
    auto_run: bool = False,
    
    # ==================================================
    # INPUT
    # ==================================================
    path_input: Path | str | None = r"C:\Users\danie\Desktop\My_Things\Hobbies\VSCODE\Virtual Envorinments\Ai-3d-Reconstruction-0.1\data\input\videos",
    name_input: str | None = None,

    # ==================================================
    # OUTPUT
    # ==================================================
    path_output: Path | str = OUTPUT_FRAMES,
    folder_name_output: str | None = None,

    # ==================================================
    # ETAPAS
    # ==================================================
    record_film_video: bool = False,
    cut_black_sides: bool = False,
    matanyone: bool = False,
    extract_frames: bool = True,

    # ==================================================
    # GRAVAÇÃO
    # ==================================================
    record_duration_seconds: float | None = None,
    camera_type: int | str = "1",
    usb_source: int = 0,
    ip_source: str = "",
) -> None:
    """
    Executa um pipeline completo de processamento de dados visuais (vídeos e/ou imagens).

    Esta função orquestra várias etapas opcionais, incluindo:
    - seleção e validação de inputs (vídeos e/ou imagens);
    - gravação de vídeo a partir de câmara;
    - extração de frames de vídeos;
    - cópia de imagens para a pasta de output;
    - execução de um pipeline de segmentação (MatAnyone);
    - geração de um manifest (relatório) com todos os paths e ações executadas.

    O comportamento do pipeline é controlado através do parâmetro `input_data_type`
    e das flags das etapas, podendo funcionar de forma interativa ou automática.

    --------------------
    Modos de funcionamento
    --------------------
    O parâmetro `input_data_type` define o tipo principal de dados a processar:

    - "video", "v", "videos", "vs":
        Processa um ou mais vídeos.
    - "images", "image", "imagens", "imagem", "i", "is":
        Processa um conjunto de imagens.
    - "both", "ambos", "b":
        Processa vídeos e imagens.
    - "record", "film", "filmar":
        Força o pipeline a entrar em modo vídeo (útil quando combinado com gravação).

    --------------------
    Inputs
    --------------------
    - `path_input`:
        Pasta base onde se encontram os vídeos ou imagens.
        Se None, é definido automaticamente com base em `input_data_type`
        (`INPUT_VIDEO_ROOT` ou `INPUT_IMAGES_ROOT`).

    - `name_input`:
        Opcional. Permite especificar um ficheiro ou subpasta dentro de `path_input`,
        funcionando como um filtro adicional.
        Pode apontar para:
            - um ficheiro de vídeo,
            - uma imagem,
            - uma pasta.

    --------------------
    Outputs
    --------------------
    - `path_output`:
        Pasta base onde os resultados são guardados.

    - `folder_name_output`:
        Nome da subpasta dentro de `path_output` onde serão colocados:
            - imagens copiadas,
            - frames extraídos,
            - manifest do pipeline.

    --------------------
    Etapas opcionais
    --------------------
    - `record_film_video`:
        Se True, grava um novo vídeo usando a câmara configurada.

    - `extract_frames`:
        Se True, extrai frames dos vídeos encontrados ou gravados.

    - `cut_black_sides`:
        Se True, remove automaticamente barras pretas durante a extração de frames.

    - `matanyone`:
        Se True, executa o pipeline de segmentação MatAnyone sobre os outputs.

    --------------------
    Modo interativo / automático
    --------------------
    - `auto_run=False`:
        O pipeline pede confirmação antes de cada etapa.
    - `auto_run=True`:
        Todas as etapas ativas são executadas automaticamente, sem perguntas.

    --------------------
    Manifest do pipeline
    --------------------
    No final da execução é criado um ficheiro `pipeline_manifest.json` na pasta
    de output, contendo:
        - inputs utilizados,
        - vídeos processados e/ou gravados,
        - imagens copiadas,
        - pastas de frames geradas,
        - opções e modos usados no pipeline.

    --------------------
    Exceções
    --------------------
    - Levanta erro se:
        - os paths configurados não existirem,
        - não forem encontrados vídeos nem imagens,
        - o tipo de input não for suportado.

    Returns:
        None
        (Todos os resultados são escritos em disco.)
    """

    # ==================================================
    # Normalização do tipo de input
    # ==================================================
    input_data_type = input_data_type.lower()

    is_video_mode = input_data_type in {"video", "v", "videos", "vs"}
    is_image_mode = input_data_type in {"images", "image", "imagens", "imagem", "i", "is"}
    is_both_mode = input_data_type in {"both", "ambos", "b"}
    is_record_mode = input_data_type in {"record", "film", "filmar"}

    # 👉 record implica vídeo
    if is_record_mode:
        is_video_mode = True

    # ==================================================
    # Definir path_input automaticamente (com validação)
    # ==================================================
    if path_input is None:
        if is_image_mode:
            path_input = INPUT_IMAGES_ROOT
        elif is_video_mode or is_both_mode:
            path_input = INPUT_VIDEO_ROOT
        else:
            raise ValueError(
                f"❌ input_data_type inválido ou não suportado: '{input_data_type}'"
            )

    # Converter para Path
    try:
        path_input = Path(path_input) # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"❌ Erro ao converter path_input para Path: {path_input}"
        ) from e


    # ==================================================
    # Validação do path_input (path_loader)
    # ==================================================
    if not path_input.exists():
        raise FileNotFoundError(
            f"❌ O path definido não existe.\n"
            f"   path_input: {path_input}\n"
            f"   Verifica o utils.path_loader e o ficheiro de configuração."
        )

    if not path_input.is_dir():
        raise NotADirectoryError(
            f"❌ O path_input não é uma pasta válida: {path_input}"
        )


    # ==================================================
    # Normalização do output
    # ==================================================
    path_output = Path(path_output)


    if folder_name_output is None:
        folder_name_output = "frames_output"

    path_output = path_output / folder_name_output
    path_output.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # Criar Manifest do Pipeline
    # ==================================================
    manifest = PipelineManifest(
        pipeline_name="run_process_pipeline",
        input_data_type=input_data_type,
        path_input=path_input,
        path_output=path_output,
        name_input=name_input,
        modes={
            "video": is_video_mode,
            "images": is_image_mode,
            "both": is_both_mode,
            "record": is_record_mode,
        },
        options={
            "record_film_video": record_film_video,
            "extract_frames": extract_frames,
            "cut_black_sides": cut_black_sides,
            "matanyone": matanyone,
        },
    )

    # ==================================================
    # Recolha de inputs
    # ==================================================
    input_videos: list[Path] = []
    input_images: list[Path] = []

    if name_input is not None:
        candidate = path_input / name_input
        if candidate.exists():
            if candidate.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                input_videos.append(candidate)
            else:
                input_images.append(candidate)
    else:
        if path_input.exists():
            for p in path_input.iterdir():
                if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    input_videos.append(p)
                elif p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    input_images.append(p)

    # Registar inputs no manifest
    for v in input_videos:
        manifest.add_input_video(v)

    for i in input_images:
        manifest.add_input_image(i)

    # ==================================================
    # Gravação de vídeo (APENAS se record_film_video=True)
    # ==================================================
    if record_film_video:
        if _confirm_step("🎥 Iniciar gravação de vídeo?", auto_run):
            recorded_video_path = record_video(
                output_name=None,
                duration_seconds=record_duration_seconds,
                preview=True,
                auto_camera=True,
                camera_type=camera_type,
                usb_source=usb_source,
                ip_source=ip_source,
                videos_dir=INPUT_VIDEO_ROOT,
            )

            input_videos.append(recorded_video_path)
            manifest.add_recorded_video(recorded_video_path)
            manifest.data["modes"]["video"] = True

    # ==================================================
    # Validação final
    # ==================================================
    if not input_videos and not input_images:
        raise RuntimeError("❌ Nenhum vídeo ou imagem disponível para processamento.")

    # ==================================================
    # Processamento de imagens
    # ==================================================
    if is_image_mode or is_both_mode:
        if _confirm_step("🖼️ Copiar imagens para output?", auto_run):
            for img in input_images:
                copied_path = path_output / img.name
                shutil.copy(img, copied_path)
                manifest.add_copied_image(copied_path)


    # ==================================================
    # Processamento de vídeos
    # ==================================================
    if is_video_mode or is_both_mode:
        if extract_frames and _confirm_step("🎞️ Extrair frames dos vídeos?", auto_run):
            for video in input_videos:
                print(f"🎞️ A processar vídeo: {video.name}")
                frames_dir = video_to_frames(
                    video_path=video,
                    output_dir=path_output,
                    cut_black_sides=cut_black_sides,
                )

                manifest.add_frames_output(video.stem, frames_dir)

    # ==================================================
    # Etapa MatAnyone
    # ==================================================
    """if matanyone:
        if _confirm_step("🧠 Executar MatAnyone?", auto_run):
            run_matanyone(path_output)

    manifest_path = manifest.save(path_output)
    print(f"🧾 Manifest guardado → {manifest_path}")

    print("✅ Pipeline concluído com sucesso.")"""



run_process_pipeline()
