from pathlib import Path
import sys
print(sys.executable)

from utils.path_loader import load_paths
paths = load_paths()

PROJECT_ROOT = paths["project_root"]
TRAIN_IMAGES_RANDOM_ROOT = paths["train_random_images"]

from data_edit.image_degradations.defocused_blur import process_folder_defocus_blur
from data_edit.image_degradations.noise import process_folder_noise
from data_edit.image_degradations.pixelation import process_folder_pixelation
from data_edit.image_degradations.jpeg_compression import process_folder_compression
from data_edit.image_degradations.fast_fading_rayleigh import process_folder_channel




def run_degradations_pipeline(
    # ==================================================
    # INPUT
    # ==================================================
    path_input: Path | str = PROJECT_ROOT /"data"/"images_clean",

    # ==================================================
    # OUTPUT ROOT
    # ==================================================
    path_output: Path | str = PROJECT_ROOT /"data"/"images_degraded",

) -> None:

    path_input = Path(path_input)
    path_output = Path(path_output)
    path_output.mkdir(parents=True, exist_ok=True)

    print(f"Input: {path_input}")
    print(f"Output: {path_output}")

    # Opções
    print("\nEscolhe o tipo de degradação:")
    print("[1] Defocused Blur")
    print("[2] Noisy")
    print("[3] Very Noisy")
    print("[4] Pixelated")
    print("[5] Very Pixelated")
    print("[6] JPEG / JPEG2000 (Compression)")
    print("[7] JPEG2000 + Fast Fading Rayleigh (Channel Errors)")
    print("[8] Todas")

    escolha = input("Opção: ").strip().lower()
    ALL = escolha in ["8", "all", "todos"]

    # -----------------------------
    # DEFOCUSED BLUR
    # -----------------------------
    if escolha in ["1", "blur", "defocused", "defocused_blur", "d", "db","d_b"] or ALL:
        (path_output / "defocused").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar Defocused Blur...")
        process_folder_defocus_blur(
            input_dir=path_input,
            output_dir=path_output / "defocused",
            strength="random_strong"
        )

    # -----------------------------
    # NOISY
    # -----------------------------
    if escolha in ["2", "noisy", "noise", "n", "no"] or ALL:
        (path_output / "noisy").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar Noisy...")

        num_noises_distribution = {
            1: 0.6,
            2: 0.3,
            3: 0.09,
            4: 0.01
        }

        strengths_noisy = {
            "gaussian": {"std_range": (10, 35)},
            "salt_pepper": {"amount_range": (0.005, 0.07)},
            "speckle": {"std_range": (0.08, 0.4)},
            "poisson": {"scale_range": (0.9, 3.5)}
        }

        process_folder_noise(
            input_dir=path_input,
            output_dir=path_output / "noisy",
            noise_types="all",
            num_noises_distribution=num_noises_distribution,
            strength_ranges=strengths_noisy
        )

    # -----------------------------
    # VERY NOISY
    # -----------------------------
    if escolha in ["3", "very_noisy", "vn", "v_n", "v_noisy", "vnoise"] or ALL:
        (path_output / "very_noisy").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar Very Noisy...")

        num_noises_distribution = {
            1: 0.6,
            2: 0.3,
            3: 0.09,
            4: 0.01
        }

        strengths_very_noisy = {
            "gaussian": {"std_range": (39, 110)},
            "salt_pepper": {"amount_range": (0.08, 0.3)},
            "speckle": {"std_range": (0.45, 1.1)},
            "poisson": {"scale_range": (3.5, 8.0)}
        }

        process_folder_noise(
            input_dir=path_input,
            output_dir=path_output / "very_noisy",
            noise_types="all",
            num_noises_distribution=num_noises_distribution,
            strength_ranges=strengths_very_noisy
        )

    # -----------------------------
    # PIXELATED
    # -----------------------------
    if escolha in ["4", "pixelation", "pixelated", "pixel", "p"] or ALL:
        (path_output / "pixelated").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar Pixelated...")

        num_pixelations_distribution = {
            1: 0.6,
            2: 0.3,
            3: 0.09,
            4: 0.01,
        }

        strengths_pixelated = {
            "mosaic_small": {"block_range": (10, 55), "num_patches_range": (4, 12)},
            "mosaic_big": {"block_range": (50, 100), "num_patches_range": (1, 6)},

            # Blocos grandes
            "isolated_small": {
                "block_range": (10, 38),            # píxel relativamente fino
                "num_clusters_range": (2, 30),      # quantos aglomerados aparecem
                "blocks_per_cluster_range": (5, 30), # quantos “píxeis grandes” por aglomerado
                "size_range": (10, 50),             # tamanho de cada bloco
                "cluster_radius_range": (50, 800)   # quão compacto é cada aglomerado
            },

            # Blocos médios
            "isolated_medium": {
                "block_range": (45, 100),          
                "num_clusters_range": (5, 10),
                "blocks_per_cluster_range": (2, 10),
                "size_range": (50, 200),
                "cluster_radius_range": (80, 1000)
            },

            # Blocos enormes
            "isolated_big": {
                "block_range": (100, 230),           
                "num_clusters_range": (2, 5),
                "blocks_per_cluster_range": (1, 3),
                "size_range": (120, 400),
                "cluster_radius_range": (150, 1000)
            }
        }



        process_folder_pixelation(
            input_dir=path_input,
            output_dir=path_output / "pixelated",
            pixelation_types="all",
            num_pixelations_distribution=num_pixelations_distribution,
            strength_ranges=strengths_pixelated
        )

    # -----------------------------
    # VERY PIXELATED
    # -----------------------------
    if escolha in ["5", "very_pixelated", "v_pixelated","vpixelated", "very_pixelation", "vpixel", "vp"] or ALL:
        (path_output / "very_pixelated").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar Very Pixelated...")

        num_pixelations_distribution = {
            1: 0.70,
            2: 0.23,
            3: 0.06,
            4: 0.01,
        }

        strengths_pixelated = {
            "global_small": {"block_range": (5, 30)},
            "global_big": {"block_range": (31, 60)},
            "mosaic_big": {"block_range": (50, 120), "num_patches_range": (10, 20)},
            "isolated_small": {
                "block_range": (10, 45),            # píxel relativamente fino
                "num_clusters_range": (30, 60),      # quantos aglomerados aparecem
                "blocks_per_cluster_range": (5, 30), # quantos “píxeis grandes” por aglomerado
                "size_range": (20, 100),             # tamanho de cada bloco
                "cluster_radius_range": (300, 2000)   # quão compacto é cada aglomerado
            },
            "isolated_big": {
                "block_range": (100, 230),           
                "num_clusters_range": (9, 20),
                "blocks_per_cluster_range": (1, 4),
                "size_range": (120, 500),
                "cluster_radius_range": (300, 2500)
            }
        }



        process_folder_pixelation(
            input_dir=path_input,
            output_dir=path_output / "very_pixelated",
            pixelation_types="all",
            num_pixelations_distribution=num_pixelations_distribution,
            strength_ranges=strengths_pixelated
        )
    # -----------------------------
    # JPEG / JPEG2000 COMPRESSION
    # -----------------------------
    if escolha in ["6", "jpeg", "compression", "c", "jpeg_compression","comp"] or ALL:
        (path_output / "jpeg_compression").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar JPEG / JPEG2000 Compression...")

        strengths_compression = {
            "jpeg": {
                "quality_range": (5, 50)      # compressão forte a média
            },
            "jpeg2000": {
                "rate_range": (0.15, 2.5)     # em bits por pixel, como no LIVE IQA
            }
        }

        process_folder_compression(
            input_dir=path_input,
            output_dir=path_output / "jpeg_compression",
            compression_types="all",
            strength_ranges=strengths_compression,
            use_kakadu=False   # muda para True se tiveres Kakadu real instalado
        )

    # -----------------------------
    # FAST FADING RAYLEIGH
    # -----------------------------
    if escolha in ["7", "rayleigh", "fast_fading_rayleigh", "fastfading", "ffr","ff", "channel"] or ALL:
        (path_output / "fast_fading_rayleigh").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar Fast Fading Rayleigh Channel...")

        strengths_channel = {
            "fast_fading_rayleigh": {
                "snr_range": (2, 25)   # SNR baixo → muito erro, alto → pouco erro
            }
        }

        process_folder_channel(
            input_dir=path_input,
            output_dir=path_output / "fast_fading_rayleigh",
            channel_types="all",
            strength_ranges=strengths_channel
        )
    
    print("\nPipeline terminada com sucesso.")




if __name__ == "__main__":
    run_degradations_pipeline(
        path_input=TRAIN_IMAGES_RANDOM_ROOT /"Good"/"kwentar-sharp",
        path_output=PROJECT_ROOT /"data"/"images_degraded"
    )
