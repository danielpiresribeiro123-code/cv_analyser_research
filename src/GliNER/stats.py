import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURAÇÃO
# ==========================================
NAME = "Comparativo_GliNER_Skills"
PASTA_ORIGINAL = Path("data/raw/GliNER_test")          # Onde estão os JSONs anotados à mão
PASTA_GLINER = Path("src/GliNER/output_test")   # Onde estão os JSONs gerados pelo GliNER
PASTA_RELATORIO = Path("data/comparar_NER/GliNER")
# ==========================================

def calcular_metricas_set(true_set, pred_set):
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def consolidar_avaliacao_gliner():
    PASTA_RELATORIO.mkdir(parents=True, exist_ok=True)
    
    resultados_por_ficheiro = []

    # 1. Iterar pelos ficheiros originais (Ground Truth)
    for ficheiro_original in PASTA_ORIGINAL.glob("*.json"):
        # Tentar encontrar o correspondente no GliNER (ajusta o prefixo se necessário)
        nome_extraido = f"extracted_{ficheiro_original.name}"
        caminho_gliner = PASTA_GLINER / nome_extraido

        if not caminho_gliner.exists():
            print(f"⚠️ Aviso: Resultado do GliNER não encontrado para {ficheiro_original.name}")
            continue

        # Carregar dados originais
        with open(ficheiro_original, 'r', encoding='utf-8') as f:
            data_true = json.load(f)
            text = data_true.get("text", "")
            # Extrair skills anotadas (Ground Truth)
            true_skills = set()
            for ann in data_true.get("annotations", []):
                if "SKILL" in str(ann[2]).upper():
                    true_skills.add(text[ann[0]:ann[1]].strip().lower())

        # Carregar dados extraídos pelo GliNER
        with open(caminho_gliner, 'r', encoding='utf-8') as f:
            data_pred = json.load(f)
            # No teu código anterior, o GliNER guarda numa chave "skills"
            pred_skills = set([s.strip().lower() for s in data_pred.get("skills", [])])

        # Calcular métricas para este ficheiro
        p, r, f1 = calcular_metricas_set(true_skills, pred_skills)
        
        resultados_por_ficheiro.append({
            "ficheiro": ficheiro_original.name,
            "precision": p,
            "recall": r,
            "f1": f1
        })

    if not resultados_por_ficheiro:
        print("❌ Nenhum ficheiro correspondente encontrado para comparar.")
        return

    # 2. Criar DataFrame e calcular médias
    df = pd.DataFrame(resultados_por_ficheiro)
    df_resumo = pd.DataFrame({
        "Métrica": ["Precision", "Recall", "F1-Score"],
        "Valor Médio": [df["precision"].mean(), df["recall"].mean(), df["f1"].mean()]
    })

    # 3. Guardar Ficheiros
    df.to_csv(PASTA_RELATORIO / f"{NAME}_detalhado.csv", index=False)
    df_resumo.to_csv(PASTA_RELATORIO / f"{NAME}_resumo.csv", index=False)

    # 4. Gerar Gráfico
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Métrica", y="Valor Médio", data=df_resumo, palette="magma")
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.title(f"Métricas Médias GliNER (Skills)\nBaseado em {len(df)} ficheiros")
    plt.ylim(0, 1.1)
    plt.savefig(PASTA_RELATORIO / f"{NAME}_grafico.png", dpi=300)
    plt.close()

    print(f"✅ Relatório concluído!")
    print(f"📍 Resultados em: {PASTA_RELATORIO.absolute()}")

if __name__ == "__main__":
    consolidar_avaliacao_gliner()