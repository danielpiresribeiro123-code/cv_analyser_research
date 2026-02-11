from collections import Counter
from typing import List, Dict


def extract_all_skills(dataset: List[Dict]) -> List[str]:
    """Extrai todas as skills do dataset."""
    skills = []

    for sample in dataset:
        for _, _, label in sample["annotations"]:
            # "SKILL: Python" -> "Python"
            if ":" in label:
                _, skill = label.split(":", 1)
                skills.append(skill.strip())
            else:
                skills.append(label.strip())

    return skills


def skill_frequency(skills: List[str]) -> Counter:
    """Conta frequência de cada skill."""
    return Counter(skills)


def dataset_basic_stats(dataset: List[Dict]) -> Dict:
    """Estatísticas básicas do dataset."""
    num_docs = len(dataset)
    num_annotations = sum(len(s["annotations"]) for s in dataset)

    return {
        "documents": num_docs,
        "total_annotations": num_annotations,
        "avg_annotations_per_doc": num_annotations / num_docs if num_docs else 0,
    }
