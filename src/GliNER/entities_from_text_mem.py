from pathlib import Path
import json
from gliner import GLiNER
from utils.path_loader import load_paths
import pandas as pd
from typing import Union, List, Dict, Optional

# Define label sets globally for reuse
SKILL_LABELS = [
    "skill", "personal attribute", "management skill", "competency",
    "spoken language", "programming language", "software framework",
    "library", "database", "cloud platform", "devops tool",
    "operating system", "web technology", "machine learning algorithm",
    "data analysis tool", "big data tool", "version control system",
    "creative software", "project management tool", "office suite",
    "crm software", "erp software", "methodology", "development practice",
    "soft skill", "leadership skill",
]

OTHER_LABELS = [
    "person name", "address", "years of experience", 
    "job title", "company", "university", "degree",
]

ALL_LABELS = SKILL_LABELS + OTHER_LABELS

def extract_entities_from_text(
    text: str,
    model: GLiNER = None,
    threshold: float = 0.3
) -> Dict[str, List[str]]:
    """
    Extract entities from a single text using GLiNER.
    
    Args:
        text: The CV text to process (should be preprocessed beforehand if needed)
        model: GLiNER model instance. If None, loads default model.
        threshold: Confidence threshold for entity detection
    
    Returns:
        Dictionary with entity labels as keys and lists of extracted entities as values
    """
    if model is None:
        model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    # Predict entities
    entities = model.predict_entities(text, ALL_LABELS, threshold=threshold)
    
    # Organize entities by label
    result_labels = OTHER_LABELS + ["skills"]
    entities_by_label = {label: [] for label in result_labels}
    
    for entity in entities:
        if entity['label'] in SKILL_LABELS:
            entities_by_label["skills"].append(entity['text'])
        elif entity['label'] in OTHER_LABELS:
            entities_by_label[entity['label']].append(entity['text'])
    
    # Remove duplicates while preserving order
    for label in entities_by_label:
        entities_by_label[label] = list(dict.fromkeys(entities_by_label[label]))
    
    return entities_by_label


def process_texts_gliner(
    texts: Union[List[str], Dict[str, str]],
    model: GLiNER = None,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Process multiple texts and extract entities.
    Returns a pandas DataFrame with one row per text.
    
    Args:
        texts: Either:
            - List of text strings
            - Dict mapping IDs to text strings
        model: GLiNER model. If None, loads default.
        threshold: Entity detection threshold
    
    Returns:
        DataFrame with columns: id, skills, person_name, job_title, company, 
                               university, degree, address, years_of_experience
    
    Example:
        >>> from text_GliNER_preprocessing import preprocess_ocr_for_gliner
        >>> texts = [preprocess_ocr_for_gliner("Email: john@example.com\nSkills: Python, Java"), ...]
        >>> df = process_texts_gliner(texts)
    """
    if model is None:
        model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    
    # Normalize input
    if isinstance(texts, list):
        text_items = [(i, text) for i, text in enumerate(texts)]
    elif isinstance(texts, dict):
        text_items = list(texts.items())
    else:
        raise ValueError("texts must be a list or dict")
    
    rows = []
    
    for text_id, text in text_items:
        try:
            # Extract entities
            entities_dict = extract_entities_from_text(
                text,
                model=model,
                threshold=threshold
            )
            
            # Flatten lists into comma-separated strings for DataFrame
            row = {"id": text_id}
            for label in OTHER_LABELS + ["skills"]:
                entities = entities_dict.get(label, [])
                row[label] = ", ".join(entities) if entities else None
            
            rows.append(row)
        
        except Exception as e:
            rows.append({
                "id": text_id,
                "error": str(e)
            })
    
    df = pd.DataFrame(rows)
    return df


def process_folder_and_save(input_folder_path: str = None, output_folder_name: str = "results"):
    """
    Legacy function: reads from disk and saves to disk.
    For memory-based usage, use process_texts_gliner() instead.
    """
    paths = load_paths()
    project_root = Path(paths["project_root"])
    input_dir = project_root / "data"/"raw"/"GliNER_test"
    output_dir = project_root / "src"/"GliNER"/"output_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

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

            entities_by_label = extract_entities_from_text(cv_text, model=model)

            output_file = output_dir / f"extracted_{file_path.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entities_by_label, f, ensure_ascii=False, indent=4)
            
            print(f"Sucesso: Guardado em {output_file}")

        except Exception as e:
            print(f"Erro ao processar {file_path.name}: {e}")

if __name__ == "__main__":
    # Example: Process texts from OCR pipeline and get DataFrame
    from text_GliNER_preprocessing import preprocess_ocr_for_gliner
    
    sample_texts = [
        "John Smith\nEmail: john@example.com\nSkills: Python, Java, C++\nYears of Experience: 5\nCompany: Tech Corp",
        "Jane Doe\nLinkedIn: linkedin.com/in/janedoe\nSkills: Data Science, Machine Learning\nDegree: Bachelor's in Computer Science",
    ]
    
    # Preprocess first
    preprocessed_texts = [preprocess_ocr_for_gliner(text) for text in sample_texts]
    
    # Then process with GLiNER
    df = process_texts_gliner(preprocessed_texts)
    print(df)
    print("\nDataFrame columns:", df.columns.tolist())