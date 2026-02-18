import json
import sys
import importlib.util
from pathlib import Path
from typing import Union, List, Dict, Optional
import pandas as pd
from gliner import GLiNER
from utils.path_loader import load_paths

# ==========================================
# CONFIGURAÇÃO DE LABELS (UNIFICADA)
# ==========================================

SKILL_LABELS = [
    "skill", "technical expertise", "software and tools", "programming language",
    "software framework", "library", "database", "cloud platform", "devops tool",
    "operating system", "web technology", "machine learning algorithm",
    "data analysis tool", "big data tool", "version control system",
    "creative software", "project management tool", "office suite",
    "crm software", "erp software", "methodology", "development practice",
    "soft skill", "leadership skill", "personal attribute", "management skill",
    "competency", "spoken language", "interpersonal strength", 
    "standard or protocol", "certification", "area of specialization"
]

OTHER_LABELS = [
    "person name", "job title", "company", "university", 
    "degree", "years of experience", "address", "email", "phone"
]

ALL_LABELS = SKILL_LABELS + OTHER_LABELS

# ==========================================
# GESTÃO DE PREPROCESSAMENTO (LÓGICA LOCAL)
# ==========================================

def get_preprocess_func():
    """Tenta carregar a função de preprocessamento de forma resiliente."""
    try:
        from src.GliNER.text_Gliner_preprocessing import preprocess_ocr_for_gliner
        return preprocess_ocr_for_gliner
    except Exception:
        try:
            module_path = Path(__file__).parent / "text_Gliner_preprocessing.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location("text_Gliner_preprocessing", module_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["text_Gliner_preprocessing"] = mod
                spec.loader.exec_module(mod)
                return mod.preprocess_ocr_for_gliner
        except Exception:
            pass
    return lambda x: x 

# ==========================================
# FUNÇÕES DE EXTRAÇÃO (PIPELINE)
# ==========================================

def extract_entities_from_text(
    text: str,
    model: GLiNER = None,
    threshold: float = 0.15
) -> Dict[str, List[str]]:
    """Extrai entidades de um texto único usando a lógica local."""
    if model is None:
        model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    # Predict
    entities = model.predict_entities(text, ALL_LABELS, threshold=threshold)
    
    # Organização por labels (Agrupando skills como no pipeline)
    result_labels = OTHER_LABELS + ["skills"]
    entities_by_label = {label: [] for label in result_labels}
    
    for entity in entities:
        label = entity['label']
        val = entity['text'].strip()
        
        if label in SKILL_LABELS:
            entities_by_label["skills"].append(val)
        elif label in OTHER_LABELS:
            entities_by_label[label].append(val)
    
    # Deduplicação preservando a ordem
    for label in entities_by_label:
        entities_by_label[label] = list(dict.fromkeys(entities_by_label[label]))
    
    return entities_by_label


def process_texts_gliner(
    texts: Union[List[str], Dict[str, str]],
    model: GLiNER = None,
    threshold: float = 0.15
) -> pd.DataFrame:
    """Processa múltiplos textos e retorna um DataFrame."""
    if model is None:
        model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    if isinstance(texts, list):
        text_items = [(i, text) for i, text in enumerate(texts)]
    elif isinstance(texts, dict):
        text_items = list(texts.items())
    else:
        raise ValueError("texts must be a list or dict")
    
    rows = []
    for text_id, text in text_items:
        try:
            entities_dict = extract_entities_from_text(text, model=model, threshold=threshold)
            
            row = {"id": text_id}
            for label in OTHER_LABELS + ["skills"]:
                entities = entities_dict.get(label, [])
                row[label] = ", ".join(entities) if entities else None
            rows.append(row)
        except Exception as e:
            rows.append({"id": text_id, "error": str(e)})
    
    return pd.DataFrame(rows)

# ==========================================
# PROCESSO DE DISCO (LEGACY / LOCAL)
# ==========================================

def process_folder_and_save(input_folder_path: str = None, output_folder_name: str = "output_test"):
    paths = load_paths()
    project_root = Path(paths["project_root"])
    
    input_dir = project_root / (input_folder_path if input_folder_path else "data/raw/GliNER_test")
    output_dir = project_root / "src" / "GliNER" / output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"⏳ Carregando modelo GliNER...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    preprocess_func = get_preprocess_func()

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"❌ Nenhum arquivo JSON encontrado em: {input_dir}")
        return

    for file_path in json_files:
        try:
            print(f"  -> Processando: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                cv_data = json.load(f)
            
            cv_text = cv_data.get("text", str(cv_data)) if isinstance(cv_data, dict) else str(cv_data)
            cv_text = preprocess_func(cv_text)

            # Usa a função de extração da pipeline para manter consistência
            entities_by_label = extract_entities_from_text(cv_text, model=model)

            output_file = output_dir / f"extracted_{file_path.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entities_by_label, f, ensure_ascii=False, indent=4)
            
            print(f"    ✅ Sucesso: {file_path.name}")

        except Exception as e:
            print(f"    ❌ Erro ao processar {file_path.name}: {e}")

if __name__ == "__main__":
    # Teste rápido simulando pipeline
    preprocess_func = get_preprocess_func()
    sample = "John Smith. Skills: Python, Docker. Experience: 5 years at TechCorp."
    
    # Processamento em memória
    df = process_texts_gliner([preprocess_func(sample)])
    print(df)
    
    # Processamento em disco
    # process_folder_and_save()