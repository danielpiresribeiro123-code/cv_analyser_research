import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def perform_eda(raw_data_path):
    raw_path = Path(raw_data_path)
    all_skills = []
    cv_lengths = []
    skills_per_cv = []

    for json_file in raw_path.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            text = data.get("text", "")
            annotations = data.get("annotations", [])
            
            cv_lengths.append(len(text.split()))
            
            skills_in_this_cv = [ann[2].replace("SKILL: ", "").lower() for ann in annotations if "SKILL" in str(ann[2])]
            all_skills.extend(skills_in_this_cv)
            skills_per_cv.append(len(skills_in_this_cv))

    skill_counts = Counter(all_skills)
    df_skills = pd.DataFrame(skill_counts.most_common(20), columns=['Skill', 'Frequência'])

    print(f"📊 Estatísticas Gerais:")
    print(f"- Total de CVs analisados: {len(cv_lengths)}")
    print(f"- Média de palavras por CV: {sum(cv_lengths)/len(cv_lengths):.2f}")
    print(f"- Média de skills por CV: {sum(skills_per_cv)/len(skills_per_cv):.2f}")

    plt.figure(figsize=(12, 6))
    plt.bar(df_skills['Skill'], df_skills['Frequência'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Skills mais frequentes no Dataset')
    plt.tight_layout()
    plt.savefig("top_skills_dist.png")
    
    plt.figure(figsize=(8, 5))
    plt.hist(cv_lengths, bins=10, color='orange', edgecolor='black')
    plt.title('Distribuição do Tamanho dos CVs (Palavras)')
    plt.savefig("dist_tamanho_cvs.png")

if __name__ == '__main__':
    perform_eda("data/raw/Json")