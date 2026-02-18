import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# ==========================================
# CONFIGURAÇÃO
# ==========================================
NAME = "Avaliacao_GliNER_Final_Capstone"
PASTA_ORIGINAL = Path("data/raw/GliNER_test")
PASTA_GLINER = Path("src/GliNER/output_test")
PASTA_RELATORIO = Path("data/comparar_NER/GliNER")

# Blacklist baseada nos teus logs de erro
LIXO_GT = {
    "com", "gmail", "work", "information", "technical", "technologies", 
    "instructions", "salary", "material", "location", "use", "ensure", 
    "can", "stream", "requirements", "activities", "success", "profile", 
    "page", "date", "cv", "curriculum", "gender", "male", "female", 
    "single", "married", "nationality", "academic", "training", "knowledge", 
    "handling", "senior", "ltd", "parts", "experience", "m", "c", "scientist",
    "engineer", "analyst", "science", "engineering", "technology", "object"
}

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\W+', '', str(text)).lower()
    return text if len(text) > 1 or text in ["r", "c"] else ""

def filtrar_redundancias_e_curtas(lista_skills):
    """
    Remove skills redundantes (Data dentro de Data Science) 
    e palavras soltas que não fazem sentido sozinhas (Scientist, Engineer).
    """
    skills = sorted(list(set(lista_skills)), key=len, reverse=True)
    filtradas = []
    
    for skill in skills:
        s_lower = skill.lower().strip()
        # Ignora palavras que sabemos serem lixo ou demasiado curtas/genéricas
        if s_lower in LIXO_GT or len(s_lower) < 3:
            if s_lower not in ["r", "c", "sql", "aws"]: # Whitelist de exceções
                continue
                
        if not any(s_lower in f.lower() for f in filtradas):
            filtradas.append(skill)
            
    return set(filtradas)

def calcular_metricas_capstone(true_set, pred_set):
    # 1. Filtramos o Ground Truth para manter apenas conceitos longos/reais
    true_clean_raw = filtrar_redundancias_e_curtas(true_set)
    true_clean = {clean_text(s) for s in true_clean_raw if clean_text(s)}
    
    # 2. Limpamos as previsões do GliNER
    pred_clean = {clean_text(s) for s in pred_set if clean_text(s) and clean_text(s) not in LIXO_GT}
    
    if not true_clean:
        return None, None, None, set(), set()

    tp = len(true_clean.intersection(pred_clean))
    fp = len(pred_clean - true_clean)
    fn = len(true_clean - pred_clean)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, true_clean, pred_clean

def consolidar_avaliacao():
    PASTA_RELATORIO.mkdir(parents=True, exist_ok=True)
    resultados = []
    
    for f_gliner in PASTA_GLINER.glob("*.json"):
        nome_original = f_gliner.name.replace("extracted_", "")
        caminho_original = PASTA_ORIGINAL / nome_original
        
        if not caminho_original.exists(): continue
            
        try:
            with open(caminho_original, 'r', encoding='utf-8') as f:
                data_true = json.load(f)
                text = data_true.get("text", "")
                true_skills = {text[ann[0]:ann[1]].strip() for ann in data_true.get("annotations", []) if "SKILL" in str(ann[2]).upper()}

            with open(f_gliner, 'r', encoding='utf-8') as f:
                data_pred = json.load(f)
                pred_skills = set(data_pred.get("skills", []))

            p, r, f1, t_clean, p_clean = calcular_metricas_capstone(true_skills, pred_skills)
            
            if p is not None:
                resultados.append({"file": nome_original, "precision": p, "recall": r, "f1": f1})
        except Exception as e:
            print(f"❌ Erro em {nome_original}: {e}")

    df = pd.DataFrame(resultados)
    if df.empty: return

    # Relatório final
    print(f"\n✅ AVALIAÇÃO CAPSTONE FINALIZADA")
    print(f"📍 Precision: {df['precision'].mean():.2%}")
    print(f"📍 Recall:    {df['recall'].mean():.2%}")
    print(f"📍 F1 Médio:  {df['f1'].mean():.2%}")

if __name__ == "__main__":
    consolidar_avaliacao()