# Projeto 2D → 3D

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

## Models / AIs

São usados:

* **YOLO** → deteção e criação de masks (opcional)
* **COLMAP** → reconstrução 3D (com ou sem CUDA)
* **MatAnyOne** → remoção de background em vídeo (opcional)

Links:

* YOLO: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* COLMAP: [https://colmap.github.io/](https://colmap.github.io/)
* MatAnyOne: [https://github.com/pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone)

Documentação útil para YOLO segmentation:

* [https://docs.ultralytics.com/pt/tasks/segment/#train](https://docs.ultralytics.com/pt/tasks/segment/#train)
* [https://docs.ultralytics.com/pt/datasets/segment/coco/#dataset-yaml](https://docs.ultralytics.com/pt/datasets/segment/coco/#dataset-yaml)

---

## Dataset para Classificação de Imagens

Para cada tipo de classificação serão usadas **500 imagens**:

* **350 imagens** do dataset:

  * [https://www.kaggle.com/datasets/kwentar/blur-dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset)
* **150 imagens próprias**, obtidas por nós

Isto garante:

* Diversidade real de dados
* Mistura entre dados públicos e dados controlados
* Melhor generalização do modelo

---

## Input / Filmagem

Aceita:

* Vídeos
* Imagens
* Ambos
* Ou gravação direta de vídeo em tempo real

---

## Segmentação do objeto principal com MatAnyOne (opcional / futuro)

Usado apenas quando o input é um **vídeo** (não é adequado para imagens soltas).

O MatAnyOne permite:

* Criar **masks temporais** consistentes ao longo do vídeo
* Isolar automaticamente o objeto principal em cada frame

Vantagens:

* Melhor separação entre objeto e fundo
* Ajuda em cenas com movimento de câmara ou do objeto

Nota:

> Este passo é opcional e só faz sentido quando se trabalha diretamente com vídeo.

Pipeline com MatAnyOne:

```
Vídeo → MatAnyOne → Frames + Masks → Filtragem → Classificação → Clustering → 3D
```

Pipeline sem MatAnyOne:

```
Vídeo → Filtragem → Classificação → Clustering → 3D
```

---

## Processamento Inicial

Transforma vídeos em imagens/frames (caso ainda não sejam) e guarda-os.

---

## Filtragem Simples (antes de ML/DL)

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

O modelo DL vai classificar em:

* **Defocused Blur** → imagem desfocada
* **Motion Blur** → imagem com movimento
* **Noisy** → granulação aleatória
* **Very Noisy** → ruído extremo
* **Pixelated** → blocos visíveis de cor
* **Good** → imagem adequada

Apenas imagens classificadas como **Good** são usadas para reconstrução 3D.
Todas as restantes são descartadas.

---

## Clustering

Entre imagens semelhantes, escolher a mais adequada e eliminar redundâncias.

---

## Representação 3D

Reconstrução do espaço 3D usando **COLMAP** a partir das imagens filtradas e selecionadas.
