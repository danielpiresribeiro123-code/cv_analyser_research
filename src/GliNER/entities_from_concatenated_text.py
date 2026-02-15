from pathlib import Path
import json
from gliner import GLiNER
from utils.path_loader import load_paths
import importlib.util
import sys

try:
    from GliNER.text_json_preprocessing import preprocess_ocr_for_gliner
except Exception:
    try:
        from text_json_preprocessing import preprocess_ocr_for_gliner
    except Exception:
        module_path = Path(__file__).parent / "text_json_preprocessing.py"
        spec = importlib.util.spec_from_file_location("text_json_preprocessing", module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["text_json_preprocessing"] = mod
        spec.loader.exec_module(mod)
        preprocess_ocr_for_gliner = mod.preprocess_ocr_for_gliner

def process_folder_and_save(input_folder_path: str, output_folder_name: str = "results"):
    paths = load_paths()
    project_root = Path(paths["project_root"])
    input_dir = project_root / "data"/"raw"/"GliNER_test"
    output_dir = project_root / "src"/"GliNER"/"output_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    skill_labels = [
        "skill", "personal attribute", "management skill", "competency",
        "spoken language", "programming language", "software framework",
        "library", "database", "cloud platform", "devops tool",
        "operating system", "web technology", "machine learning algorithm",
        "data analysis tool", "big data tool", "version control system",
        "creative software", "project management tool", "office suite",
        "crm software", "erp software", "methodology", "development practice",
        "soft skill", "leadership skill",
    ]
    other_labels = [
        "person name", "address", "years of experience", 
        "job title", "company", "university", "degree",
    ]
    all_labels = skill_labels + other_labels

    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"Nenhum arquivo JSON encontrado em: {input_dir}")
        return

    for file_path in json_files:
        print(f"A processar: {file_path.name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cv_data = json.load(f)
            
            cv_text = cv_data.get("text", str(cv_data)) if isinstance(cv_data, dict) else str(cv_data)

            try:
                cv_text = preprocess_ocr_for_gliner(cv_text)
            except:
                pass

            entities = model.predict_entities(cv_text, all_labels, threshold=0.3)

            result_labels = other_labels + ["skills"]
            entities_by_label = {label: [] for label in result_labels}
            
            for entity in entities:
                if entity['label'] in skill_labels:
                    entities_by_label["skills"].append(entity['text'])
                elif entity['label'] in other_labels:
                    entities_by_label[entity['label']].append(entity['text'])

            for label in entities_by_label:
                entities_by_label[label] = list(set(entities_by_label[label]))

            output_file = output_dir / f"extracted_{file_path.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entities_by_label, f, ensure_ascii=False, indent=4)
            
            print(f"Sucesso: Guardado em {output_file}")

        except Exception as e:
            print(f"Erro ao processar {file_path.name}: {e}")

if __name__ == "__main__":
    process_folder_and_save("src/ocr/extracted_text", "src/ocr/results")