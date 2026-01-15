from pathlib import Path
import sys
print(sys.executable)

from utils.path_loader import load_paths
paths = load_paths()

PROJECT_ROOT = paths["project_root"]

from data_edit.image_degradations.defocused_blur import process_folder_defocus_blur
from data_edit.image_degradations.noise import process_folder_noise
from data_edit.image_degradations.pixelation import process_folder_pixelation





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

    print("\nEscolhe o tipo de degradação:")
    print("[1] Defocused Blur")
    print("[2] Noisy")
    print("[3] Very Noisy")
    print("[4] Pixelation")
    print("[5] Todas")
    escolha = input("Opção: ").strip().lower()

    # -----------------------------
    # DEFUSED BLUR
    # -----------------------------
    if escolha in ["1", "blur", "defocused", "defocused_blur", "d", "5"]:
        print("\nA aplicar Defocused Blur...")
        process_folder_defocus_blur(
            input_dir=path_input,
            output_dir=path_output / "defocused",
            strength="random"
        )

    # -----------------------------
    # NOISY
    # -----------------------------
    if escolha in ["2", "noisy", "n", "5"]:
        print("\nA aplicar Noisy...")

        num_noises_distribution = {
            1: 0.5,
            2: 0.4,
            3: 0.1
        }

        strengths_noisy = {
            "gaussian": {"std_range": (10, 25)},
            "salt_pepper": {"amount_range": (0.005, 0.02)},
            "speckle": {"std_range": (0.05, 0.25)},
            "poisson": {"scale_range": (0.8, 1.2)}
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
    if escolha in ["3", "very_noisy", "vn", "5"]:
        print("\nA aplicar Very Noisy...")

        num_noises_distribution = {
            1: 0.4,
            2: 0.4,
            3: 0.2
        }

        strengths_very_noisy = {
            "gaussian": {"std_range": (30, 70)},
            "salt_pepper": {"amount_range": (0.02, 0.08)},
            "speckle": {"std_range": (0.3, 0.8)},
            "poisson": {"scale_range": (1.5, 3.0)}
        }

        process_folder_noise(
            input_dir=path_input,
            output_dir=path_output / "very_noisy",
            noise_types="all",
            num_noises_distribution=num_noises_distribution,
            strength_ranges=strengths_very_noisy
        )

    # -----------------------------
    # PIXELATION
    # -----------------------------
    if escolha in ["4", "pixelation", "pixel", "p", "5"]:
        print("\nA aplicar Pixelation...")

        num_pixelations_distribution = {
            1: 0.5,
            2: 0.4,
            3: 0.1
        }

        strengths_pixelated = {
            "global": {"block_range": (12, 40)},
            "local": {"block_range": (15, 60), "area_range": (0.2, 0.5)},
            "mosaic": {"block_range": (20, 80), "num_patches_range": (2, 6)}
        }

        process_folder_pixelation(
            input_dir=path_input,
            output_dir=path_output / "pixelated",
            pixelation_types="all",
            num_pixelations_distribution=num_pixelations_distribution,
            strength_ranges=strengths_pixelated
        )

if __name__ == "__main__":
    run_degradations_pipeline(
        path_input=r"C:\Users\danie\Desktop\My_Things\Hobbies\VSCODE\Data\Random_images\Good\personal-sharp\camera_GalaxyS215G_SM-G991BDS",
        path_output=PROJECT_ROOT /"data"/"images_degraded"
    )
