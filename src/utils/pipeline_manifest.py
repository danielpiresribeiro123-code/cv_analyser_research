from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineManifest:
    """
    Manifest reutilizável para pipelines de processamento.
    """

    def __init__(
        self,
        *,
        pipeline_name: str,
        input_data_type: str,
        path_input: Path,
        path_output: Path,
        name_input: Optional[str] = None,
        options: Optional[dict] = None,
        modes: Optional[dict] = None,
    ) -> None:
        self._data: dict = {
            "pipeline": pipeline_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input": {
                "input_data_type": input_data_type,
                "path_input": str(path_input),
                "name_input": name_input,
            },
            "modes": modes or {},
            "options": options or {},
            "videos": {
                "input_videos": [],
                "recorded_videos": [],
                "frames_output": {},
            },
            "images": {
                "input_images": [],
                "copied_images": [],
            },
            "output": {
                "path_output": str(path_output),
            },
        }

    # --------------------------------------------------
    # Registos
    # --------------------------------------------------
    def add_input_video(self, path: Path) -> None:
        self._data["videos"]["input_videos"].append(str(path))

    def add_recorded_video(self, path: Path) -> None:
        self._data["videos"]["recorded_videos"].append(str(path))

    def add_frames_output(self, video_stem: str, frames_dir: Path) -> None:
        self._data["videos"]["frames_output"][video_stem] = str(frames_dir)

    def add_input_image(self, path: Path) -> None:
        self._data["images"]["input_images"].append(str(path))

    def add_copied_image(self, path: Path) -> None:
        self._data["images"]["copied_images"].append(str(path))

    # --------------------------------------------------
    # Guardar
    # --------------------------------------------------
    def save(self, output_dir: Path, filename: str = "pipeline_manifest.json") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / filename
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)

        return manifest_path

    # --------------------------------------------------
    # Acesso (opcional)
    # --------------------------------------------------
    @property
    def data(self) -> dict:
        return self._data
