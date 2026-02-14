import os
from pathlib import Path
from dotenv import load_dotenv

def load_paths() -> dict:
    load_dotenv()

    def get_env_path(key: str, must_exist: bool = True) -> Path:
        value = os.getenv(key)
        if value is None:
            raise RuntimeError(f"Variável de ambiente não definida: {key}")

        path = Path(value).resolve()

        if must_exist and not path.exists():
            raise RuntimeError(f"Caminho não encontrado para {key}: {path}")

        return path

    def get_env_str(key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise RuntimeError(f"Variável de ambiente não definida: {key}")
        return value

    try:
        return {
            "project_root": get_env_path("PROJECT_ROOT"),
            "data": {
                "datasets_root": get_env_path("DATASETS_ROOT"),
                "skill_tables": get_env_path("DATASETS_SKILL_TABLE")
            },
            "models": {
                "root": get_env_path("MODELS_IN_ROOT", must_exist=False),
                "out": get_env_path("MODELS_OUT_ROOT", must_exist=False)
            },
            "env": get_env_str("ENV"),
        }
    except RuntimeError as e:
        print(f"Erro ao carregar configurações: {e}")
        raise