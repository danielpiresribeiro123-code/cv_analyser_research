import cv2
import numpy as np
import random
from pathlib import Path
import json




def pixelate(image, block_size):
    block_size = max(1, int(block_size))
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



def partial_pixelation(image, block_range=(8, 50), num_patches_range=(3, 8)):
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



PIXELATION_STRENGTHS = {
    "global": {
        "low":    {"block_range": (6, 12)},
        "medium": {"block_range": (12, 25)},
        "strong": {"block_range": (25, 50)},
    },

    "partial": {
        "low": {
            "block_range": (8, 15),
            "num_patches_range": (1, 3)
        },
        "medium": {
            "block_range": (15, 30),
            "num_patches_range": (3, 6)
        },
        "strong": {
            "block_range": (30, 60),
            "num_patches_range": (6, 10)
        }
    },

    "isolated": {
        "low": {
            "block_range": (10, 20),
            "num_clusters_range": (1, 2),
            "blocks_per_cluster_range": (1, 3),
            "size_range": (20, 50),
            "cluster_radius_range": (40, 80)
        },
        "medium": {
            "block_range": (20, 35),
            "num_clusters_range": (2, 4),
            "blocks_per_cluster_range": (3, 6),
            "size_range": (40, 100),
            "cluster_radius_range": (80, 150)
        },
        "strong": {
            "block_range": (35, 70),
            "num_clusters_range": (4, 8),
            "blocks_per_cluster_range": (5, 12),
            "size_range": (80, 200),
            "cluster_radius_range": (120, 300)
        }
    }
}

PIXELATION_TYPES = {
    "global": global_pixelation,
    "partial": partial_pixelation,
    "isolated": isolated_blocks_pixelation
}


def apply_pixelation(
    image,
    pixelation_type: str,
    strength: str
):
    """
    pixelation_type : 'global', 'partial' ou 'isolated'
    strength        : 'low', 'medium' ou 'strong'
    """

    if pixelation_type not in PIXELATION_TYPES:
        raise ValueError(f"Tipo inválido: {pixelation_type}")

    if strength not in PIXELATION_STRENGTHS[pixelation_type]:
        raise ValueError(f"Força inválida para {pixelation_type}: {strength}")

    func = PIXELATION_TYPES[pixelation_type]
    params = PIXELATION_STRENGTHS[pixelation_type][strength]

    out, p = func(image, **params)

    applied = f"{pixelation_type}_{strength}"
    return out, applied, p




def apply_random_pixelation(image):
    pixelation_type = random.choice(["global", "partial", "isolated"])
    strength = random.choice(["low", "medium", "strong"])
    return apply_pixelation(image, pixelation_type, strength)



def process_folder_pixelation(
    input_dir,
    output_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="pixelation_manifest.json",
    use_random: bool = False,
    pixelation_type ="isolated", #isolated, partial, global
    strength ="strong" #low, medium, strong
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

        # agora é simples e consistente
        if use_random:
            pixelated, applied, params = apply_random_pixelation(image)
        else:
            pixelated, applied, params = apply_pixelation(image, pixelation_type,strength)


        tag = applied if applied else "none"
        out_name = f"{img_path.stem}_pixel_{tag}{img_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), pixelated)

        if save_manifest:
            manifest.append({
                "input": img_path.name,
                "output": out_name,
                "applied_pixelation": applied,
                "params": params
            })

        print(f"{img_path.name} -> {out_name}")

    if save_manifest:
        with open(output_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
