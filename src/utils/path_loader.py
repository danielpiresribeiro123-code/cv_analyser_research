import os
from pathlib import Path
from dotenv import load_dotenv


def load_paths():
    """
    Lê todos os paths a partir do .env e devolve um dicionário estruturado.

    Exemplo:
    from config.path_loader import load_paths

    paths = load_paths()

    good_frames = paths["data"]["frames"]["good"]
    masked_frames = paths["data"]["frames"]["masked"]

    colmap_exe = paths["models"]["colmap"]["exe_cuda"]
    yolo_seg = paths["models"]["yolo"]["dynamic_seg"]

    """

    load_dotenv()

    def get_env(key):
        value = os.getenv(key)
        if value is None:
            raise RuntimeError(f"Variável de ambiente não definida: {key}")
        return Path(value)

    paths = {
        # -------------------------
        # Raiz
        # -------------------------
        "project_root": get_env("PROJECT_ROOT"),

        # -------------------------
        # Dados
        # -------------------------
        "data": {
            "root": get_env("DATA_ROOT"),

            "frames": get_env("FRAMES_ROOT"),

            "datasets": get_env("DATASETS_ROOT"),

            "graphics": get_env("GRAPHICS_ROOT"),

            "input": {
                "video": get_env("INPUT_VIDEO_ROOT"),
                "images": get_env("INPUT_IMAGES_ROOT"),
            },
        },

        # -------------------------
        # Modelos
        # -------------------------
        "models": {
            "root": get_env("MODELS_ROOT"),

            "colmap": {
                "bat_cuda": get_env("COLMAP_BAT_CUDA"),
                "bin_cuda": get_env("COLMAP_BIN_CUDA"),
                "exe_cuda": get_env("COLMAP_EXECUTABLE_CUDA"),

                "bat_nocuda": get_env("COLMAP_BAT_NOCUDA"),
                "bin_nocuda": get_env("COLMAP_BIN_NOCUDA"),
                "exe_nocuda": get_env("COLMAP_EXECUTABLE_NOCUDA"),
            },

            "yolo": {
                "root": get_env("YOLO_ROOT"),
                "dynamic_det": get_env("YOLO_DYNAMIC_DET"),
                "dynamic_seg": get_env("YOLO_DYNAMIC_SEG"),
            },
        },
    }

    return paths
