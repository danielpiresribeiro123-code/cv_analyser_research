# Projeto 2D → 3D
Este projeto tem como objetivo transformar imagens 2D ou frames extraídos de vídeo em uma reconstrução 3D do espaço, garantindo primeiro a qualidade dos dados utilizados.
Para isso, é criado um pipeline que combina filtragem automática, classificação por Deep Learning, remoção opcional de objetos dinâmicos, clustering de imagens semelhantes e, por fim, reconstrução 3D com o COLMAP.

A ideia principal é:

> só usar imagens realmente adequadas para reconstrução 3D, evitando ruído, blur, baixa textura, redundâncias e interferências de objetos em movimento.

O projeto junta:

* Regras simples de visão computacional (filtros iniciais)
* Modelos de Deep Learning para classificação e segmentação
* Técnicas de seleção e redução de dados
* Reconstrução 3D baseada em fotogrametria

O pipeline é modular, permitindo ativar ou desativar componentes como MatAnyOne ou masking com YOLO, conforme o tipo de input (vídeo ou imagens) e a complexidade desejada.


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

Além disso, a partir das imagens consideradas adequadas (**Good**), foram criadas versões artificiais para gerar mais classes:

* **Noisy**
* **Very Noisy**
* **Pixelated**

Estas transformações são feitas usando **scikit-image**, aplicando:

* Adição controlada de ruído
* Aumento da intensidade do ruído
* Pixelização por redução e reamostragem da resolução

Isto permite:

* Ter dados realistas para classes que são difíceis de recolher manualmente
* Controlar o nível de degradação da imagem
* Garantir consistência entre classes

No conjunto, isto garante:

* Diversidade real de dados
* Mistura entre dados públicos, dados próprios e dados sintéticos controlados
* Melhor generalização do modelo

---
# Pipeline Input Process
### Input / Filmagem

Aceita:

* Vídeos
* Imagens
* Ambos
* Ou gravação direta de vídeo em tempo real

---
### Segmentação do objeto principal com MatAnyOne (opcional / futuro)

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
Vídeo → Frames → Filtragem → Classificação → Clustering → 3D
```

---
### Processamento Inicial

Transforma vídeos em imagens/frames (caso ainda não sejam) e guarda-os.

---
# Pipeline Quality Selection
### Filtragem Simples (antes de ML/DL)

1. Exposição

* Overexposed: > 40% dos pixels > 240
* Underexposed: > 40% dos pixels < 15

2. Baixa textura

* Variância global muito baixa
* Ou > 85% da imagem quase da mesma cor

3. Objetos dinâmicos (opcional, fase futura)

* Pessoas, carros, animais, etc.
* YOLO11l.pt → deteção
* Escolher: remover ou manter imagens

4. Imagens obviamente inadequadas

* Muito noisy ou muito blurry
* Features muito fora dos valores normais

Este filtro considerado no ponto 4 é baseado nos **valores mínimo e máximo das features** extraídas das imagens consideradas adequadas (**Good**) no dataset de treino.
Qualquer imagem cujas features fiquem fora destes limites é automaticamente descartada, por ser considerada fora da distribuição normal de imagens válidas para reconstrução 3D.


---
### Classificação de Imagens

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
### Masking de Objetos Dinâmicos (opcional)

Aplicado apenas às imagens classificadas como **Good**, antes do clustering.
Não possivel de aplicar caso se tenha em **Filtragem Simples (antes de ML/DL)** no ponto 3 escolhido remover as imagens com objetos dinâmicos.

Objetivo:

* Remover ou reduzir a influência de objetos dinâmicos como:

  * Pessoas
  * Carros
  * Animais
  * Outros elementos em movimento

É usado o modelo:
* **YOLO11l-seg.pt** → deteção + criação de masks por segmentação

Processo:

* Detetar objetos dinâmicos na imagem
* Criar máscaras para essas regiões

Pipeline atualizado:

```
Filtragem → Classificação → Masking Dynamic (opcional) → Clustering → 3D
```

---
### Clustering

Entre imagens semelhantes, escolher a mais adequada e eliminar redundâncias.

---
## Representação 3D

Reconstrução do espaço 3D usando **COLMAP** a partir das imagens filtradas e selecionadas.
