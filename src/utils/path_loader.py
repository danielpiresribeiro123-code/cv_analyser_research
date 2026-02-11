import os
from pathlib import Path
from dotenv import load_dotenv


def load_paths() -> dict:
    """
    Load all project paths from the .env file and return
    a structured dictionary.

    Usage:
        from config.path_loader import load_paths

        paths = load_paths()

        project_root = paths["project_root"]
        datasets_root = paths["data"]["datasets_root"]
        skill_tables = paths["data"]["skill_tables"]

        models_root = paths["models"]["root"]
        env_name = paths["env"]
    """

    # Load .env variables
    load_dotenv()

    def get_env_path(key: str, must_exist: bool = True) -> Path:
        """
        Retrieve a path from environment variables and optionally
        validate that it exists.
        """
        value = os.getenv(key)

        if value is None:
            raise RuntimeError(f"Environment variable not defined: {key}")

        path = Path(value)

        if must_exist and not path.exists():
            raise RuntimeError(f"Path does not exist for {key}: {path}")

        return path

    def get_env_str(key: str) -> str:
        """Retrieve a string environment variable."""
        value = os.getenv(key)

        if value is None:
            raise RuntimeError(f"Environment variable not defined: {key}")

        return value

    paths = {
        # --------------------------------------------------
        # Project root
        # --------------------------------------------------
        "project_root": get_env_path("PROJECT_ROOT"),

        # --------------------------------------------------
        # Data
        # --------------------------------------------------
        "data": {
            "datasets_root": get_env_path("DATASETS_ROOT"),
            "skill_tables": get_env_path("DATASETS_SKILL_TABLE")},

        # --------------------------------------------------
        # Models
        # --------------------------------------------------
        "models": {"root": get_env_path("MODELS_IN_ROOT", must_exist=False),
                   "out": get_env_path("MODELS_OUT_ROOT", must_exist=False)},

        # --------------------------------------------------
        # Environment
        # --------------------------------------------------
        "env": get_env_str("ENV"),
    }

    return paths
