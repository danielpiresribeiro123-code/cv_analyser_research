from pathlib import Path
import sys

from utils.path_loader import load_paths
paths = load_paths()

PROJECT_ROOT = paths["project_root"]
TRAIN_IMAGES_RANDOM_ROOT = paths["train_random_images"]

from data_edit.image_degradations.defocused_blur import process_folder_defocus_blur
from data_edit.image_degradations.motion_blur import process_folder_motion_blur
from data_edit.image_degradations.noise import process_folder_noise
from data_edit.image_degradations.pixelation import process_folder_pixelation
from data_edit.image_degradations.jpeg_compression import process_folder_compression
from data_edit.image_degradations.glitchs import process_folder_channel

NOISE_TYPES = [
    "color_gaussian",
    "color_poisson",
    "color_speckle",
    "color_salt_pepper",
    "color_uniform",
    "gray_gaussian",
    "gray_poisson",
    "gray_speckle",
    "gray_salt_pepper",
    "gray_uniform"
]

PIXELATION_TYPES = [
    "mosaic_small",
    "mosaic_big",
    "isolated_small",
    "isolated_medium",
    "isolated_big",
    "global_small",
    "global_big",
    "jpeg_compress_small",
    "jpeg_compress_big"
]

COMPRESSION_TYPES = [
    "jpeg_dct",
    "jpeg2000",
    "jpeg_xl"
]


# ==================================================
# Runner genérico
# ==================================================

def run_single_transform(
    process_func,
    input_dir,
    output_root,
    transform_name,
    strength_mode="medium",   # low | medium | strong | all | random
    **kwargs
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if strength_mode in ["low", "medium", "strong"]:
        out_dir = output_root / f"{transform_name}_{strength_mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        process_func(
            input_dir=input_dir,
            output_dir=out_dir,
            use_random=False,
            strength=strength_mode,
            **kwargs
        )

    elif strength_mode == "all":
        for s in ["low", "medium", "strong"]:
            out_dir = output_root / f"{transform_name}_{s}"
            out_dir.mkdir(parents=True, exist_ok=True)
            process_func(
                input_dir=input_dir,
                output_dir=out_dir,
                use_random=False,
                strength=s,
                **kwargs
            )

    elif strength_mode == "random":
        out_dir = output_root / f"{transform_name}_random"
        out_dir.mkdir(parents=True, exist_ok=True)
        process_func(
            input_dir=input_dir,
            output_dir=out_dir,
            use_random=True,
            **kwargs
        )

    else:
        raise ValueError("strength_mode deve ser: low | medium | strong | all | random")

def run_transform_with_types(
    process_func,
    input_dir,
    output_root,
    base_name,
    available_types,
    type_mode="global",        # tipo específico | all | random
    strength_mode="medium",    # low | medium | strong | all | random
    type_kw_name="blur_type"   # nome do argumento a passar (blur_type, noise_type, etc.)
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if type_mode in available_types:
        # Tipo único
        run_single_transform(
            process_func=process_func,
            input_dir=input_dir,
            output_root=output_root,
            transform_name=f"{base_name}_{type_mode}",
            strength_mode=strength_mode,
            **{type_kw_name: type_mode}
        )

    elif type_mode == "all":
        # Todos os tipos → uma pasta por tipo
        for t in available_types:
            run_single_transform(
                process_func=process_func,
                input_dir=input_dir,
                output_root=output_root,
                transform_name=f"{base_name}_{t}",
                strength_mode=strength_mode,
                **{type_kw_name: t}
            )

    elif type_mode == "random":
        # Um único folder, tipo aleatório por imagem
        out_dir = output_root / f"{base_name}_random_types"
        out_dir.mkdir(parents=True, exist_ok=True)
        process_func(
            input_dir=input_dir,
            output_dir=out_dir,
            use_random=True
        )

    else:
        raise ValueError(
            f"type_mode deve ser um dos seguintes: {available_types} | all | random"
        )


# ==================================================
# PIPELINE INTERATIVA
# ==================================================

def run_degradations_pipeline(
    path_input: Path | str,
    path_output: Path | str
):
    path_input = Path(path_input)
    path_output = Path(path_output)
    path_output.mkdir(parents=True, exist_ok=True)

    print("\n=== PIPELINE DE DEGRADAÇÕES ===\n")
    print("Escolhe o tipo de transformação:")
    print("[1] Defocused Blur")
    print("[2] Motion Blur")
    print("[3] Noise")
    print("[4] Pixelation")
    print("[5] JPEG Compression")
    print("[6] Glitches (Channel)")
    print("[7] Todas")
    
    choice = input("Opção: ").strip()

    print("\nEscolhe a força:")
    print("[1] low")
    print("[2] medium")
    print("[3] strong")
    print("[4] all (cria 3 pastas)")
    print("[5] random (uma pasta com forças aleatórias)")
    
    strength_map = {
        "1": "low",
        "2": "medium",
        "3": "strong",
        "4": "all",
        "5": "random"
    }
    strength_choice = input("Força: ").strip()
    strength_mode = strength_map.get(strength_choice, "medium")

    # --------------------------------------------------
    # 1. Defocused Blur
    # --------------------------------------------------
    if choice in ["1", "7"]:
        blur_types = ["global", "partial_random", "partial_radial"]

        print("\nTipos de Defocused Blur:")
        for t in blur_types:
            print(f"  {t}")
        print("  all")
        print("  random")

        blur_type_mode = input("Escolhe: ").strip().lower()

        run_transform_with_types(
            process_func=process_folder_defocus_blur,
            input_dir=path_input,
            output_root=path_output / "defocused_blur",
            base_name="defocus",
            available_types=blur_types,
            type_mode=blur_type_mode,
            strength_mode=strength_mode,
            type_kw_name="blur_type"
        )


    # --------------------------------------------------
    # 2. Motion Blur
    # --------------------------------------------------
    if choice in ["2", "7"]:
        blur_types = ["global", "partial_random", "partial_radial"]

        print("\nTipos de Motion Blur:")
        for t in blur_types:
            print(f"  {t}")
        print("  all")
        print("  random")

        blur_type_mode = input("Escolhe: ").strip().lower()

        run_transform_with_types(
            process_func=process_folder_motion_blur,
            input_dir=path_input,
            output_root=path_output / "motion_blur",
            base_name="motion",
            available_types=blur_types,
            type_mode=blur_type_mode,
            strength_mode=strength_mode,
            type_kw_name="blur_type"
        )


    # --------------------------------------------------
    # 3. Noise
    # --------------------------------------------------
    if choice in ["3", "7"]:
        print("\nTipos de Noise:")
        for t in NOISE_TYPES:
            print(f"  {t}")
        print("  all")
        print("  random")

        noise_type_mode = input("Escolhe: ").strip().lower()

        run_transform_with_types(
            process_func=process_folder_noise,
            input_dir=path_input,
            output_root=path_output / "noise",
            base_name="noise",
            available_types=NOISE_TYPES,
            type_mode=noise_type_mode,
            strength_mode=strength_mode,
            type_kw_name="noise_type"
        )

    # --------------------------------------------------
    # 4. Pixelation
    # --------------------------------------------------
    if choice in ["4", "7"]:
        print("\nTipos de Pixelation:")
        for t in PIXELATION_TYPES:
            print(f"  {t}")
        print("  all")
        print("  random")

        pixel_type_mode = input("Escolhe: ").strip().lower()

        run_transform_with_types(
            process_func=process_folder_pixelation,
            input_dir=path_input,
            output_root=path_output / "pixelation",
            base_name="pixelation",
            available_types=PIXELATION_TYPES,
            type_mode=pixel_type_mode,
            strength_mode=strength_mode,
            type_kw_name="pixelation_type"
        )

    # --------------------------------------------------
    # 5. JPEG Compression
    # --------------------------------------------------
    if choice in ["5", "7"]:
        print("\nTipos de Compressão:")
        for t in COMPRESSION_TYPES:
            print(f"  {t}")
        print("  all")
        print("  random")

        comp_type_mode = input("Escolhe: ").strip().lower()

        run_transform_with_types(
            process_func=process_folder_compression,
            input_dir=path_input,
            output_root=path_output / "compression",
            base_name="compression",
            available_types=COMPRESSION_TYPES,
            type_mode=comp_type_mode,
            strength_mode=strength_mode,
            type_kw_name="compression_type"
        )

    # --------------------------------------------------
    # 6. Glitches
    # --------------------------------------------------
    if choice in ["6", "7"]:
        glitch_types = [
            "fast_fading_rayleigh",
            "block_drop",
            "block_shuffle",
            "color_channel_swap",
            "posterize_glitch",
            "random_bit_flip"
        ]

        print("\nTipos de Glitch:")
        for t in glitch_types:
            print(f"  {t}")
        print("  all")
        print("  random")

        glitch_type_mode = input("Escolhe: ").strip().lower()

        run_transform_with_types(
            process_func=process_folder_channel,
            input_dir=path_input,
            output_root=path_output / "glitches",
            base_name="glitch",
            available_types=glitch_types,
            type_mode=glitch_type_mode,
            strength_mode=strength_mode,
            type_kw_name="channel_type"
        )


    print("\nPipeline terminada com sucesso.")


# ==================================================
# MAIN
# ==================================================
"""
if __name__ == "__main__":
    run_degradations_pipeline(
        path_input=TRAIN_IMAGES_RANDOM_ROOT / "Good" / "kwentar-sharp",
        path_output=PROJECT_ROOT / "data" / "images_degraded"
    )
"""
if __name__ == "__main__":
    run_degradations_pipeline(
        path_input=r"C:\Users\danie\Desktop\My_Things\Hobbies\VSCODE\Data\Personal_images",
        path_output=PROJECT_ROOT / "data" / "images_degraded"
    )
