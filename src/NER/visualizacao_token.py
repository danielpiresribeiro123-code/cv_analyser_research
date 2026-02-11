from transformers import pipeline
import json

# 1. Carregar o teu ficheiro de exemplo
with open("data/raw/Json/cv (1)_annotated.json", "r", encoding="utf-8") as f:
    exemplo = json.load(f)

texto_real = exemplo["text"]
anotacoes_reais = [ann[2].replace("SKILL: ", "") for ann in exemplo["annotations"]]

# 2. Usar o teu modelo treinado
nlp = pipeline("ner", model="./modelo_final_ner", aggregation_strategy="simple")
previsoes = nlp(texto_real)

# 3. Comparação Direta
print(f"--- SKILLS QUE TU ANOTASTE ---")
print(set(anotacoes_reais))

print(f"\n--- SKILLS QUE O MODELO DETETOU ---")
skills_detetadas = [p['word'] for p in previsoes]
print(set(skills_detetadas))

# 4. Verificação de erros
falta = set(anotacoes_reais) - set(skills_detetadas)
extra = set(skills_detetadas) - set(anotacoes_reais)

if falta:
    print(f"\n⚠️ O modelo não apanhou estas: {falta}")
if extra:
    print(f"🔍 O modelo encontrou estas extra (podem ser skills novas): {extra}")