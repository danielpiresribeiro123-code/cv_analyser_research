import json
import sys
import importlib.util
from pathlib import Path
from gliner import GLiNER
from utils.path_loader import load_paths

# ==========================================
# CONFIGURAÇÃO DE LABELS E FILTROS
# ==========================================
# Etiquetas genéricas para extração automática (Zero-Shot)
AUTO_SKILL_LABELS = [
    "technical expertise",      # Domínios como Data Science, UI/UX
    "software and tools",       # Ferramentas como Excel, SAP, VS Code
    "programming language",     # Linguagens de programação
    "framework or library",     # React, PyTorch, Django
    "methodology",              # Agile, Scrum, Kanban, Lean
    "soft skill",               # Comunicação, Liderança
    "interpersonal strength",   # Trabalho em equipa, negociação
    "personal attribute",       # Proativo, organizado
    "standard or protocol",     # ISO 9001, HTTP, TCP/IP
    "certification",            # AWS Certified, PMP
    "area of specialization",   # Cybersecurity, Cloud Architecture
    "spoken language"           # Inglês, Espanhol, etc.
]

OTHER_LABELS = [
    "person name", "job title", "company", "university", "degree", "years of experience"
]

ALL_LABELS = AUTO_SKILL_LABELS + OTHER_LABELS

# Palavras que o teu dataset original marca como skill, mas que são ruído
BLACK_LIST = {
    "personal", "curriculum", "senior", "handling", "ltd", "male", 
    "single", "cv", "page", "date", "email", "address", "phone", "profile"
}

# ==========================================
# UTILITÁRIOS DE PRÉ-PROCESSAMENTO
# ==========================================
def get_preprocess_func():
    try:
        module_path = Path(__file__).parent / "text_Gliner_preprocessing.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("text_Gliner_preprocessing", module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.preprocess_ocr_for_gliner
    except Exception:
        pass
    return lambda x: x  # Fallback: retorna o texto sem alteração

def process_folder_and_save():
    # 1. Carregar caminhos do projeto
    paths = load_paths()
    project_root = Path(paths["project_root"])
    input_dir = project_root / "data" / "raw" / "GliNER_test"
    output_dir = project_root / "src" / "GliNER" / "output_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Inicializar o modelo e preprocessamento
    print("⏳ Carregando modelo GliNER...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    preprocess_func = get_preprocess_func()

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"❌ Nenhum arquivo JSON encontrado em: {input_dir}")
        return

    print(f"🚀 Iniciando extração em {len(json_files)} ficheiros...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cv_data = json.load(f)
            
            # Extrair texto original
            cv_text = cv_data.get("text", str(cv_data)) if isinstance(cv_data, dict) else str(cv_data)
            cv_text = preprocess_func(cv_text)

            # 3. Inferência (Baixamos o threshold para aumentar o RECALL)
            entities = model.predict_entities(cv_text, ALL_LABELS, threshold=0.15)

            # 4. Organização e Filtragem Automática
            results = {
                "skills": [],
                "others": [],
                "metadata": {
                    "total_skills": 0,
                    "confidence_scores": {}
                }
            }
            
            seen_skills = set()

            for entity in entities:
                label = entity['label']
                text = entity['text'].strip()
                text_lower = text.lower()
                score = round(float(entity.get('score', 0)), 4)

                # Filtrar ruído (Blacklist e palavras muito curtas)
                if text_lower in BLACK_LIST or len(text) < 2:
                    continue

                if label in AUTO_SKILL_LABELS:
                    if text_lower not in seen_skills:
                        results["skills"].append(text)
                        results["metadata"]["confidence_scores"][text] = score
                        seen_skills.add(text_lower)
                else:
                    results["others"].append({"label": label, "text": text, "score": score})

            results["metadata"]["total_skills"] = len(results["skills"])

            # 5. Guardar Resultados
            output_file = output_dir / f"extracted_{file_path.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            print(f"✅ {file_path.name}: {len(results['skills'])} skills extraídas automaticamente.")

        except Exception as e:
            print(f"❌ Erro ao processar {file_path.name}: {e}")

if __name__ == "__main__":
    process_folder_and_save()