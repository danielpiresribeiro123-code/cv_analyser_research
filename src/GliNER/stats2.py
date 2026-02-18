import json
import pandas as pd
from pathlib import Path

PASTA_ORIGINAL = Path("data/raw/Gliner_test")
PASTA_GLINER = Path("src/GliNER/output_test")
PASTA_OUTPUT_CSV = Path("data/comparar_NER/GliNER/csv_comparativos")

FICHEIROS_PARA_TESTAR = []

def gerar_comparativo_visual(num_exemplos=3):
    PASTA_OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    
    paths_gliner = list(PASTA_GLINER.glob("*.json"))
    if not paths_gliner:
        print("❌ Nenhum ficheiro do GliNER encontrado.")
        return

    ficheiros = FICHEIROS_PARA_TESTAR if FICHEIROS_PARA_TESTAR else [f.name.replace("extracted_", "") for f in paths_gliner[:num_exemplos]]

    for nome_arq in ficheiros:
        path_gt = PASTA_ORIGINAL / nome_arq
        path_pred = PASTA_GLINER / f"extracted_{nome_arq}"

        if not path_gt.exists() or not path_pred.exists():
            print(f"⚠️ Ficheiro não encontrado: {nome_arq}")
            continue

        with open(path_gt, 'r', encoding='utf-8') as f:
            data_gt = json.load(f)
            txt = data_gt.get("text", "")
            gt_skills = sorted(list(set([txt[a[0]:a[1]].strip() for a in data_gt.get("annotations", []) if "SKILL" in str(a[2]).upper()])))

        with open(path_pred, 'r', encoding='utf-8') as f:
            data_pred = json.load(f)
            pred_skills = sorted(list(set(data_pred.get("skills", []))))

        max_len = max(len(gt_skills), len(pred_skills))
        gt_list = gt_skills + [""] * (max_len - len(gt_skills))
        pred_list = pred_skills + [""] * (max_len - len(pred_skills))

        df_comp = pd.DataFrame({
            "ANOTAÇÃO HUMANA (GT)": gt_list,
            "EXTRAÇÃO GliNER (IA)": pred_list
        })

        output_file = PASTA_OUTPUT_CSV / f"comparativo_{nome_arq.replace('.json', '.csv')}"
        df_comp.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n" + "="*60)
        print(f"📄 FICHEIRO: {nome_arq}")
        print(f"📍 Guardado em: {output_file}")
        print("="*60)
        print(df_comp.to_string(index=False))

if __name__ == "__main__":
    gerar_comparativo_visual()