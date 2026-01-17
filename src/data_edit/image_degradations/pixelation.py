import cv2
import numpy as np
import random
from pathlib import Path
import json

def pixelate(image, block_size):
    h, w = image.shape[:2]

    # garantir que nunca fica 0
    new_w = max(1, w // block_size)
    new_h = max(1, h // block_size)

    temp = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixelated



def global_pixelation(image, block_range=(6, 40)):
    block_size = random.randint(*block_range)
    return pixelate(image, block_size), {"block_size": block_size}



def mosaic_pixelation(image, block_range=(8, 50), num_patches_range=(3, 8)):
    h, w = image.shape[:2]
    output = image.copy()

    num_patches = random.randint(*num_patches_range)
    block_size = random.randint(*block_range)

    params = []

    for _ in range(num_patches):
        pw = random.randint(w//10, w//3)
        ph = random.randint(h//10, h//3)

        x = random.randint(0, w - pw)
        y = random.randint(0, h - ph)

        patch = output[y:y+ph, x:x+pw]
        pixel_patch = pixelate(patch, block_size)
        output[y:y+ph, x:x+pw] = pixel_patch

        params.append({
            "x": x, "y": y,
            "w": pw, "h": ph,
            "block_size": block_size
        })

    return output, {
        "num_patches": num_patches,
        "patches": params
    }

def isolated_blocks_pixelation(
    image,
    block_range=(10, 40),
    num_clusters_range=(1, 5),
    blocks_per_cluster_range=(2, 8),
    size_range=(20, 80),
    cluster_radius_range=(40, 120)
):
    """
    Cria vários aglomerados de quadrados pixelados.

    block_range              : intensidade da pixelização dentro de cada bloco
    num_clusters_range       : número de aglomerados na imagem
    blocks_per_cluster_range : quantos blocos existem dentro de cada aglomerado
    size_range               : tamanho de cada bloco (quadrado)
    cluster_radius_range     : quão espalhados ficam os blocos dentro do aglomerado
    """

    h, w = image.shape[:2]
    output = image.copy()

    num_clusters = random.randint(*num_clusters_range)
    block_size = random.randint(*block_range)

    params = []

    for _ in range(num_clusters):

        # centro do aglomerado
        cx = random.randint(0, w)
        cy = random.randint(0, h)

        num_blocks = random.randint(*blocks_per_cluster_range)
        cluster_radius = random.randint(*cluster_radius_range)

        cluster_blocks = []

        for _ in range(num_blocks):
            sq = random.randint(*size_range)
            sq = min(sq, w, h)

            # posição relativa ao centro do aglomerado
            dx = random.randint(-cluster_radius, cluster_radius)
            dy = random.randint(-cluster_radius, cluster_radius)

            x = int(cx + dx - sq // 2)
            y = int(cy + dy - sq // 2)

            # garantir que fica dentro da imagem
            x = max(0, min(w - sq, x))
            y = max(0, min(h - sq, y))

            patch = output[y:y+sq, x:x+sq]

            # garantir que block_size nunca é maior que o patch
            local_block_size = min(block_size, sq)
            if local_block_size % 2 == 0:
                local_block_size -= 1
            local_block_size = max(1, local_block_size)

            pixel_patch = pixelate(patch, local_block_size)
            output[y:y+sq, x:x+sq] = pixel_patch

            cluster_blocks.append({
                "x": x,
                "y": y,
                "size": sq,
                "block_size": local_block_size
            })

        params.append({
            "center": (cx, cy),
            "num_blocks": num_blocks,
            "radius": cluster_radius,
            "blocks": cluster_blocks
        })

    return output, {
        "num_clusters": num_clusters,
        "clusters": params
    }



PIXELATION_FUNCTIONS = {
    "global_small": global_pixelation,
    "global_big": global_pixelation,
    "mosaic_small": mosaic_pixelation,
    "mosaic_big": mosaic_pixelation,
    "isolated_small": isolated_blocks_pixelation,
    "isolated_medium": isolated_blocks_pixelation,
    "isolated_big": isolated_blocks_pixelation
}

P_PIXELATION_FUNCTIONS = {
    "mosaic_small": mosaic_pixelation,
    "mosaic_big": mosaic_pixelation,
    "isolated_small": isolated_blocks_pixelation,
    "isolated_medium": isolated_blocks_pixelation,
    "isolated_big": isolated_blocks_pixelation
}

VP_PIXELATION_FUNCTIONS = {
    "global_small": global_pixelation,
    "global_big": global_pixelation,
    "mosaic_small": mosaic_pixelation,
    "mosaic_big": mosaic_pixelation,
    "isolated_small": isolated_blocks_pixelation,
    "isolated_big": isolated_blocks_pixelation
}


def sample_num_pixelations(distribution: dict):
    """
    distribution: {num_pixelations: probability}
    Ex: {1: 0.5, 2: 0.4, 3: 0.1}
    """
    nums = list(distribution.keys())
    probs = list(distribution.values())
    return random.choices(nums, weights=probs, k=1)[0]
def apply_random_pixelation(
    image,
    pixelation_types="all",
    num_pixelations=1,
    num_pixelations_distribution=None,
    strength_ranges=None
):  
    if pixelation_types == "all" or pixelation_types == "p" or pixelation_types == "vp":
        if pixelation_types == "p":
            PIXELATION_FUNCTIONS = P_PIXELATION_FUNCTIONS
        else:
            PIXELATION_FUNCTIONS = VP_PIXELATION_FUNCTIONS
        available = list(PIXELATION_FUNCTIONS.keys())
    elif isinstance(pixelation_types, list):
        available = pixelation_types
    else:
        raise ValueError("pixelation_types must be 'all', 'p', 'vp' or list")

    # Escolher quantas pixelations aplicar
    if num_pixelations_distribution is not None:
        num_pixelations = sample_num_pixelations(num_pixelations_distribution)

    num_pixelations = min(num_pixelations, len(available))

    chosen = random.sample(available, num_pixelations)

    out = image.copy()
    applied = []
    params_used = []

    for ptype in chosen:
        func = PIXELATION_FUNCTIONS[ptype]
        params = strength_ranges.get(ptype, {}) if strength_ranges else {}
        out, p = func(out, **params)
        applied.append(ptype)
        params_used.append(p)

    return out, applied, params_used

def process_folder_pixelation(
    input_dir,
    output_dir,
    num_pixelations=1,
    num_pixelations_distribution=None,
    pixelation_types="all",
    strength_ranges=None,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="pixelation_manifest.json"
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        pixelated, applied, params = apply_random_pixelation(
            image,
            pixelation_types=pixelation_types,
            num_pixelations=num_pixelations,
            num_pixelations_distribution=num_pixelations_distribution,
            strength_ranges=strength_ranges
        )

        tag = "_".join(applied) if applied else "none"
        out_name = f"{img_path.stem}_pixel_{tag}{img_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), pixelated)

        if save_manifest:
            manifest.append({
                "input": img_path.name,
                "output": out_name,
                "applied_pixelations": applied,
                "params": params
            })

        print(f"{img_path.name} -> {out_name}")

    if save_manifest:
        with open(output_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
