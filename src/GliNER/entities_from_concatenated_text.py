from pathlib import Path
import json
from gliner import GLiNER
from utils.path_loader import load_paths


def extract_entities_from_cv(file_path: str):
    """
    Extract entities from a CV JSON file using GLiNER.
    
    Args:
        file_path: Relative path to the JSON file from project root
                   (e.g., "data/my_cv.json")
    
    Returns:
        entities_by_label: Dictionary with labels as keys and extracted entities as values
    """
    # Load project paths
    paths = load_paths()
    project_root = paths["project_root"]
    
    # Construct full file path
    full_path = Path(project_root) / file_path
    
    # Read the JSON file
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {full_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {full_path}")
    
    # Extract text from JSON (assuming it has a "text" field)
    if isinstance(cv_data, dict):
        cv_text = cv_data.get("text", str(cv_data))
    else:
        cv_text = str(cv_data)
    
    # 1. Load the model
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    # 2. Define semantic labels relevant to CV parsing
    skill_labels = [
        "skill",
        "personal attribute",
        "management skill",
        "competency",
        "spoken language",
        "programming language",
        "software framework",
        "library",
        "database",
        "cloud platform",
        "devops tool",
        "operating system",
        "web technology",
        "machine learning algorithm",
        "data analysis tool",
        "big data tool",
        "version control system",
        "creative software",
        "project management tool",
        "office suite",
        "crm software",
        "erp software",
        "methodology",
        "development practice",
        "soft skill",
        "leadership skill",
    ]

    other_labels = [
        "person name",
        "address",
        "years of experience",  
        "job title",
        "company",
        "university",
        "degree",
    ]

    all_labels = skill_labels + other_labels

    # 3. Predict
    # threshold=0.3 is often good for CVs where formatting can be messy.
    entities = model.predict_entities(cv_text, all_labels, threshold=0.3)

    # Organize entities by label - only keep other_labels + "skills"
    result_labels = other_labels + ["skills"]
    entities_by_label = {label: [] for label in result_labels}
    
    for entity in entities:

        # Map skill labels to "skills" key
        if entity['label'] in skill_labels:
            entities_by_label["skills"].append(entity['text'])
        # Map other labels directly
        elif entity['label'] in other_labels:
            entities_by_label[entity['label']].append(entity['text'])

    # Deduplicate each label's entities
    for label in entities_by_label:
        entities_by_label[label] = list(set(entities_by_label[label]))

    print("\n--- Extracted Entities by Label ---")
    for label, values in entities_by_label.items():
        print(f"{label}: {values}")
    
    return entities_by_label


if __name__ == "__main__":
    extract_entities_from_cv("src/ocr/extracted_text/test_image_text.json")