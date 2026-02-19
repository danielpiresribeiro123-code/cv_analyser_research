import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURAÇÃO DE CAMINHOS
# ==========================================
# Usamos agora a pasta que criaste com os ficheiros filtrados
PASTA_PROCESSED = Path("data/raw/GliNER_test")
PASTA_GLINER = Path("src/GliNER/output_test")
PASTA_RELATORIO = Path("data/comparar_NER/GliNER")

def clean_text(text):
    if not text: return ""
    # Normalização para garantir que "Python" e "python" coincidam
    return re.sub(r'\W+', '', str(text)).lower()

def calcular_metricas(true_skills, pred_skills):
    true_clean = {clean_text(s) for s in true_skills if clean_text(s)}
    pred_clean = {clean_text(s) for s in pred_skills if clean_text(s)}
    
    if not true_clean:
        return None

    tp_set = true_clean.intersection(pred_clean)
    fp_set = pred_clean - true_clean
    fn_set = true_clean - pred_clean
    
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_list": sorted(list(tp_set)),
        "fp_list": sorted(list(fp_set)),
        "fn_list": sorted(list(fn_set))
    }

def executar_estatisticas():
    PASTA_RELATORIO.mkdir(parents=True, exist_ok=True)
    resultados = []
    
    print(f"🚀 Iniciando Comparação: {PASTA_PROCESSED.name} vs GliNER")

    # Listar ficheiros na pasta processada
    ficheiros_limpos = list(PASTA_PROCESSED.glob("*.json"))
    
    for f_gt in ficheiros_limpos:
        # Procurar o correspondente no GliNER (que tem o prefixo extracted_)
        f_pred = PASTA_GLINER / f"extracted_{f_gt.name}"
        
        if not f_pred.exists():
            continue

        try:
            # 1. Carregar Real (Ground Truth filtrado)
            with open(f_gt, 'r', encoding='utf-8') as f:
                data_gt = json.load(f)
                # Extraímos as skills diretamente das anotações
                # Como usaste coordenadas, o texto no JSON já reflete a remoção
                true_skills = []
                for ann in data_gt.get("annotations", []):
                    if "SKILL" in str(ann[2]).upper():
                        # O texto da skill costuma estar no terceiro elemento da anotação
                        # ou extraído via coordenadas do campo 'text'
                        skill_val = data_gt["text"][ann[0]:ann[1]].strip()
                        true_skills.append(skill_val)

            # 2. Carregar Predição (GliNER)
            with open(f_pred, 'r', encoding='utf-8') as f:
                data_pred = json.load(f)
                pred_skills = data_pred.get("skills", [])

            # 3. Calcular
            m = calcular_metricas(true_skills, pred_skills)
            
            if m:
                # Debug visual para os casos com F1 baixo
                if m["f1"] < 0.4:
                    print("\n" + "─"*50)
                    print(f"📄 FICHEIRO: {f_gt.name}")
                    print(f"🎯 F1-SCORE: {m['f1']:.2%}")
                    print(f"✅ [REAL (Annotations)]: {true_skills[:5]}...")
                    print(f"🤖 [PRED (GliNER IA)]:   {pred_skills[:5]}...")
                    if m['fn_list']: print(f"❌ Faltou (FN): {m['fn_list'][:3]}")
                    if m['fp_list']: print(f"➕ Extra (FP):  {m['fp_list'][:3]}")
                
                resultados.append({
                    "ficheiro": f_gt.name,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"]
                })

        except Exception as e:
            print(f"❌ Erro ao processar {f_gt.name}: {e}")

    # 4. Consolidar e Gerar Gráficos
    df = pd.DataFrame(resultados)
    if df.empty:
        print("❌ Nenhum dado comparável encontrado.")
        return

    resumo = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score"],
        "Average": [df['precision'].mean(), df['recall'].mean(), df['f1'].mean()]
    })

    print("\n" + "="*40)
    print(f"✅ SUCESSO: {len(df)} ficheiros analisados")
    print(f"📍 Precision Média: {df['precision'].mean():.2%}")
    print(f"📍 Recall Médio:    {df['recall'].mean():.2%}")
    print(f"📍 F1-Score Médio:  {df['f1'].mean():.2%}")
    print("="*40)

    # Exportar resultados
    df.to_csv(PASTA_RELATORIO / "metricas_finais_limpas.csv", index=False)
    
    # Gráfico Final
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Metric", y="Average", data=resumo, hue="Metric", palette="magma", legend=False)
    plt.ylim(0, 1.1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.title("Performance GliNER")
    plt.savefig(PASTA_RELATORIO / "grafico_final_limpo.png")

if __name__ == "__main__":
    executar_estatisticas()