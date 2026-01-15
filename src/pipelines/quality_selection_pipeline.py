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
# Função/pipeline
# ==================================================

def run_quality_pipeline(
    # ==================================================
    # MODO
    # ==================================================
    copy_all_frames: bool = False, # Se for "True", copia todos os frames/images para a pasta "all_frames".

    csv_frames: bool = False, # Se for "True", faz um ficheiro csv das caractesrísticas de todos os frames.
    # Frames considerados "bad_frames" vão apenas ter as características analisadas até aquela que é a razão de ter sido considerado um "bad_frame".

    auto_run: bool = False, # Pergunta se quer passar para a proxima etapa ou não

    dynamic_protocol: str = "remove", # Pode ser "remove" ou "mask" ou "ignore".
    # "remove" (mais rápido) -> retira os frames com objetos dinamicos da análise.
    # "mask" (mais lento) -> faz masks (guardadas no path "masks") para os frames com objetos dinamicos (separados no path "masked_frames").
    # "ignore" (intermédio) -> considera os frames com objetos dinamicos como um frame normal.
    
    # ==================================================
    # INPUT
    # ==================================================
    path_input: Path | str = INPUT_VIDEO_ROOT,

    # ==================================================
    # OUTPUT
    # ==================================================
    path_output: Path | str = OUTPUT_FRAMES,
    path_csv_frames: Path | str = OUTPUT_FRAMES,

    # ==================================================
    # ETAPAS
    # ==================================================
    do_record_video: bool = False,
    do_extract_frames: bool = False,
    do_gen_masks: bool = False,
    do_apply_masks: bool = False,
    do_copy_for_reconstruction: bool = True,

    # ==================================================
    # GRAVAÇÃO
    # ==================================================
    record_duration_seconds: Optional[float] = None,

    # ==================================================
    # FRAMES (extração)
    # ==================================================
    every_n_frames: int = 1,

    # ==================================================
    # QUALIDADE DE IMAGEM (DESCARTE)
    # ==================================================
    dark_ratio_threshold: float = 0.97,
    blurry_variance_threshold: float = 120.0,

    #corrupted
    block_size= 16, #Tamanho do bloco (NxN) usado para analisar a imagem. Normal->16 
    blockiness_threshold= 1.55, #Mede diferença média entre blocos adjacentes (vertical e horizontal). Normal-> 1.45
    flat_ratio_threshold = 0.35, #Percentagem de blocos quase uniformes (pouca variação interna). Normal-> 0.30
    flat_block_var_threshold = 22.0, #Define o quão “flat” é flat. É o limiar da variância interna do bloco. Normal-> 18.0

    # ==================================================
    # DETEÇÃO DINÂMICA (YOLO – DETEÇÃO)
    # ==================================================
    presence_conf: float = 0.25,
    presence_imgsz: int = 640,
    presence_device: str | None = None,

    # ==================================================
    # SEGMENTAÇÃO (YOLO – SEG)
    # ==================================================
    yolo_conf: float = 0.35,
    yolo_imgsz: int = 960,
    yolo_device: str | None = None,

    # ==================================================
    # SELEÇÃO PARA RECONSTRUÇÃO
    # ==================================================
    selection: Optional[Dict[str, Tuple[int, Optional[int]]]] = None,
) -> None: