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
            1: 0.75,
            2: 0.2,
            3: 0.04,
            4: 0.01
        }

        strengths_noisy = {
            # RGB noise (moderado)
            "gaussian": {"std_range": (8, 35)},
            "salt_pepper": {"amount_range": (0.003, 0.05)},
            "speckle": {"std_range": (0.05, 0.35)},
            "poisson": {"scale_range": (2.5, 7.0)},
            "uniform": {"range_val": (5, 25)},

            # Luminance noise (mais suave)
            "lum_gaussian": {"std_range": (3, 18)},
            "lum_poisson": {"scale_range": (2.5, 7.0)},
            "lum_speckle": {"std_range": (0.03, 0.25)},

            # Estrutural
            #"banding": {"amplitude_range": (8, 25),"frequency_range": (2, 5)}
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
            1: 0.75,
            2: 0.2,
            3: 0.04,
            4: 0.01
        }

        strengths_very_noisy = {
            # RGB noise (forte / destrutivo)
            "gaussian": {"std_range": (45, 220)},
            "salt_pepper": {"amount_range": (0.08, 0.6)},
            "speckle": {"std_range": (0.6, 4.0)},
            "poisson": {"scale_range": (0.2, 2.4)},
            "uniform": {"range_val": (30, 120)},

            # Luminance noise (forte mas ainda fotográfico)
            "lum_gaussian": {"std_range": (20, 80)},
            "lum_poisson": {"scale_range": (0.2, 2.4)},
            "lum_speckle": {"std_range": (0.3, 1.5)},

            # Estrutural
            #"banding": {"amplitude_range": (26, 80),"frequency_range": (5, 12)}
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
            1: 0.70,
            2: 0.25,
            3: 0.04,
            4: 0.01}
        
        strengths_pixelated = {
            "jpeg_compress_small": {"quality_range": (30, 45)},
            "jpeg_compress_big": {"quality_range": (25, 35)},

            "mosaic_small": {"block_range": (10, 50), "num_patches_range": (4, 10)},
            "mosaic_big": {"block_range": (40, 100), "num_patches_range": (2, 6)},

            "isolated_small": {
                "block_range": (10, 35),
                "num_clusters_range": (2, 25),
                "blocks_per_cluster_range": (4, 20),
                "size_range": (10, 50),
                "cluster_radius_range": (50, 700)},

            "isolated_medium": {
                "block_range": (40, 90),
                "num_clusters_range": (4, 10),
                "blocks_per_cluster_range": (2, 10),
                "size_range": (40, 180),
                "cluster_radius_range": (80, 900)},

            "isolated_big": {
                "block_range": (90, 180),
                "num_clusters_range": (1, 4),
                "blocks_per_cluster_range": (1, 3),
                "size_range": (100, 350),
                "cluster_radius_range": (150, 900)}}



        process_folder_pixelation(
            input_dir=path_input,
            output_dir=path_output / "pixelated",
            pixelation_types="p",
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
            1: 0.60,
            2: 0.30,
            3: 0.09,
            4: 0.01,}

        strengths_very_pixelated = {
            "jpeg_compress_small": {"quality_range": (7, 24)},
            "jpeg_compress_big": {"quality_range": (1, 10)},

            "global_small": {"block_range": (10, 40)},
            "global_big": {"block_range": (40, 120)},

            "mosaic_small": {"block_range": (15, 50), "num_patches_range": (20, 60)},
            "mosaic_big": {"block_range": (50, 200), "num_patches_range": (8, 25)},

            "isolated_small": {
                "block_range": (20, 60),
                "num_clusters_range": (20, 50),
                "blocks_per_cluster_range": (5, 25),
                "size_range": (40, 300),
                "cluster_radius_range": (150, 1500)},

            "isolated_big": {
                "block_range": (80, 220),
                "num_clusters_range": (12, 25),
                "blocks_per_cluster_range": (1, 6),
                "size_range": (100, 500),
                "cluster_radius_range": (200, 2000)}}



        process_folder_pixelation(
            input_dir=path_input,
            output_dir=path_output / "very_pixelated",
            pixelation_types="vp",
            num_pixelations_distribution=num_pixelations_distribution,
            strength_ranges=strengths_very_pixelated
        )
    # -----------------------------
    # JPEG / JPEG2000 COMPRESSION
    # -----------------------------
    if escolha in ["6", "jpeg", "compression", "c", "jpeg_compression","comp"] or ALL:
        (path_output / "jpeg_compression").mkdir(parents=True, exist_ok=True)
        print("\nA aplicar JPEG / JPEG2000 Compression...")

        strengths_compression = {
            "jpeg": {"quality_range": (5, 50)},      # compressão forte a média
            #"jpeg2000": {"rate_range": (0.01, 0.05)}     # em bits por pixel, como no LIVE IQA
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
