## Instalação

Recomendado: usar `venv` no VSCode

Ou com conda:

```bash
conda create -n 3d_reconstruction python=3.11.9
conda init
conda activate 3d_reconstruction
```

---

## Dependências

```bash
pip install ultralytics
pip install opencv-python pillow onvif-zeep pandas
pip install scikit-learn seaborn ipython tqdm scikit-image
```

---

## Models

São usados:

* **YOLO** para deteção e criação de masks
* **COLMAP** (com ou sem CUDA) para reconstrução 3D

Documentação útil para YOLO segmentation:

* [https://docs.ultralytics.com/pt/tasks/segment/#train](https://docs.ultralytics.com/pt/tasks/segment/#train)
* [https://docs.ultralytics.com/pt/datasets/segment/coco/#dataset-yaml](https://docs.ultralytics.com/pt/datasets/segment/coco/#dataset-yaml)

---

## Filtragem simples (antes de ML/DL)

1. Exposição

* Overexposed: > 40% dos pixels > 240
* Underexposed: > 40% dos pixels < 15

2. Baixa textura

* Variância global muito baixa
* Ou > 85% da imagem quase da mesma cor

3. Objetos dinâmicos (opcional, fase futura)

* Pessoas, carros, animais, etc.
* YOLO11l.pt → deteção
* YOLO11l-seg.pt → masks
* Escolher: remover, manter ou mascarar

4. Imagens obviamente inadequadas

* Muito noisy ou muito blurry
* Features muito fora dos valores normais

---

## Classificação de Imagens

* **Defocused Blur** → imagem desfocada
* **Motion Blur** → imagem com movimento
* **Noisy** → granulação aleatória
* **Very Noisy** → ruído extremo
* **Pixelated** → blocos visíveis de cor
* **Good** → imagem adequada

---

## Clustering

Entre imagens semelhantes, escolher a mais adequada e eliminar redundâncias.

---

## Representação 3D

Reconstrução do espaço 3D usando **COLMAP** a partir das imagens filtradas.
