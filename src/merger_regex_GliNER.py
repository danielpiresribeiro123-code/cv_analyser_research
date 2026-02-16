import pandas as pd
import json
from pathlib import Path
from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = Path(paths["project_root"])

REGEX_CSV_PATH = PROJECT_ROOT / "data" / "processed_regex" / "regex_output.csv"
GLINER_OUTPUT_DIR = PROJECT_ROOT / "src" / "GliNER" / "output_test"
FINAL_OUTPUT_PATH = PROJECT_ROOT / "data" / "final_output" / "GYRO.csv"

def merge_outputs():
    if not REGEX_CSV_PATH.exists():
        print(f"Erro: Ficheiro não encontrado em {REGEX_CSV_PATH}")
        return

    df_regex = pd.read_csv(REGEX_CSV_PATH, sep=';', encoding='utf-8-sig')
    
    gliner_rows = []
    for json_file in GLINER_OUTPUT_DIR.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            clean_name = json_file.name.replace("extracted_", "")
            
            entry = {"file": clean_name}
            
            for key, value in data.items():
                if isinstance(value, list):
                    entry[key] = ", ".join(value) if value else None
                else:
                    entry[key] = value
            gliner_rows.append(entry)
        except Exception as e:
            print(f"Erro ao ler {json_file.name}: {e}")

    df_gliner = pd.DataFrame(gliner_rows)

    if "address" in df_gliner.columns:
        df_gliner = df_gliner.drop(columns=["address"])

    df_final = pd.merge(df_regex, df_gliner, on="file", how="left")

    desired_order = [
        "file", 
        "person name", 
        "phone", 
        "address", 
        "skills", 
        "degree", 
        "university", 
        "company", 
        "job title"
    ]

    existing_cols = [col for col in desired_order if col in df_final.columns]
    df_final = df_final[existing_cols]

    FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(FINAL_OUTPUT_PATH, index=False, sep=';', encoding='utf-8-sig')
    
    return df_final

if __name__ == "__main__":
    df = merge_outputs()
    if df is not None:
        print("Fusão concluída com sucesso.")
        print(df.head())