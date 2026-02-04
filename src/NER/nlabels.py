import json

with open("C:\\Users\\duart\\OneDrive\\Ambiente de Trabalho\\Faculdade\\Samsung\\CapStone\\NerData\\ResumesJsonAnnotated\\ResumesJsonAnnotated\\cv (1)_annotated.json", 'r') as f:
    data = json.load(f)

# Extrair nomes únicos de etiquetas (ex: 'SKILL: Building', 'SKILL: Python')
# Se limpares para ficar apenas 'SKILL', terás 1 entidade única.
unique_entities = set([ann[2].split(':')[0] for ann in data['annotations']])

# Se usares BIO: (O + Begin + Inside) por cada entidade
num_labels = 1 + (len(unique_entities) * 2) 
print(f"O num_labels ideal para o teu caso é: {num_labels}")