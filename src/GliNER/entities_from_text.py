import json
import sys
import importlib.util
from pathlib import Path
from gliner import GLiNER
from utils.path_loader import load_paths

# ==========================================
# CONFIGURAÇÃO DE LABELS (TAXONOMIA COMPLETA)
# ==========================================

AUTO_SKILL_LABELS = [
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

ALL_LABELS = AUTO_SKILL_LABELS + OTHER_LABELS

# ==========================================
# GESTÃO DE PREPROCESSAMENTO
# ==========================================

def get_preprocess_func():
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
# PROCESSO PRINCIPAL
# ==========================================

def process_folder_and_save(input_folder_path: str = None, output_folder_name: str = "output_test"):
    paths = load_paths()
    project_root = Path(paths["project_root"])
    
    input_dir = project_root / "data"/"raw"/"GliNER_test"
    output_dir = project_root / "src"/"GliNER"/"output_test"
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

            # Inferência com threshold baixo para capturar o máximo de entidades
            entities = model.predict_entities(cv_text, ALL_LABELS, threshold=0.15)

            final_output = {
                "skills": [],
                "professional_info": {label: [] for label in OTHER_LABELS},
                "metadata": {
                    "total_skills": 0,
                    "confidence_scores": {},
                    "filename": file_path.name
                }
            }
            
            seen_entities = set()

            for entity in entities:
                label = entity['label']
                text = entity['text'].strip()
                text_lower = text.lower()
                score = round(float(entity.get('score', 0)), 4)

                # Sem blacklist: apenas removemos duplicados exatos
                unique_key = f"{label}_{text_lower}"
                if unique_key in seen_entities:
                    continue
                
                if label in AUTO_SKILL_LABELS:
                    final_output["skills"].append(text)
                    final_output["metadata"]["confidence_scores"][text] = score
                else:
                    if label in final_output["professional_info"]:
                        final_output["professional_info"][label].append(text)
                
                seen_entities.add(unique_key)

            final_output["metadata"]["total_skills"] = len(final_output["skills"])

            output_file = output_dir / f"extracted_{file_path.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            
            print(f"    ✅ Sucesso: Extraídas {len(entities)} entidades no total.")

        except Exception as e:
            print(f"    ❌ Erro ao processar {file_path.name}: {e}")

if __name__ == "__main__":
    process_folder_and_save()