import pandas as pd
from pathlib import Path

from utils.path_loader import load_paths

# ==========================================================
# Paths
# ==========================================================

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

NER_RAW_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Json"


# ==========================================================
# Imports do support
# ==========================================================

from analysis.train_data.NER_raw.support.io_utils import (load_all_raw_ner,)

from analysis.train_data.NER_raw.support.validation import (validate_offsets,count_overlapping_spans,)

from analysis.train_data.NER_raw.support.statistics import (extract_all_skills,skill_frequency,dataset_basic_stats,)

# ==========================================================
# Função principal de análise
# ==========================================================

def create_ner_raw_train_data_analysis() -> pd.DataFrame:
    """
    Executa EDA básica do dataset NER raw e
    retorna um DataFrame com estatísticas globais.

    Este script dá:
    nº de CVs
    nº total de anotações
    média de skills por CV
    offsets inválidos
    overlaps
    nº de skills únicas
    top-20 skills
    """

    # ------------------------------------------------------
    # 1. Carregar dataset
    # ------------------------------------------------------
    dataset = load_all_raw_ner(NER_RAW_TRAIN_DATA)

    if not dataset:
        raise ValueError("Dataset NER raw vazio.")

    # ------------------------------------------------------
    # 2. Validações
    # ------------------------------------------------------
    invalid_offsets = sum(not validate_offsets(sample) for sample in dataset)

    total_overlaps = sum(count_overlapping_spans(sample) for sample in dataset)

    # ------------------------------------------------------
    # 3. Estatísticas básicas
    # ------------------------------------------------------
    basic_stats = dataset_basic_stats(dataset)

    # ------------------------------------------------------
    # 4. Frequência de skills
    # ------------------------------------------------------
    skills = extract_all_skills(dataset)
    skill_freq = skill_frequency(skills)

    num_unique_skills = len(skill_freq)

    # Top 20 skills
    top_skills = skill_freq.most_common(20)

    # ------------------------------------------------------
    # 5. Construir DataFrame resumo
    # ------------------------------------------------------
    summary = {
        "documents": basic_stats["documents"],
        "total_annotations": basic_stats["total_annotations"],
        "avg_annotations_per_doc": basic_stats["avg_annotations_per_doc"],
        "invalid_offset_docs": invalid_offsets,
        "total_overlapping_spans": total_overlaps,
        "unique_skills": num_unique_skills,
    }

    df_summary = pd.DataFrame([summary])

    # ------------------------------------------------------
    # 6. DataFrame top skills
    # ------------------------------------------------------
    df_top_skills = pd.DataFrame(top_skills, columns=["skill", "frequency"])

    # ------------------------------------------------------
    # 7. Output debug no terminal
    # ------------------------------------------------------
    print("\n=== NER RAW DATASET SUMMARY ===")
    print(df_summary.to_string(index=False))

    print("\n=== TOP 20 SKILLS ===")
    print(df_top_skills.to_string(index=False))

    return df_summary


# ==========================================================
# Execução direta
# ==========================================================

if __name__ == "__main__":
    create_ner_raw_train_data_analysis()
