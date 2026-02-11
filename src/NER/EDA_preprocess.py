import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from transformers import AutoTokenizer
from pathlib import Path

def analyze_processed_data():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
    file_path = Path("data/process_Json/dataset.json")
    
    if not file_path.exists():
        print("Ficheiro dataset.json não encontrado.")
        return

    all_tokens_as_skills = []
    seq_lengths = []
    skill_counts_per_doc = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            input_ids = record["input_ids"]
            labels = record["labels"]
            
            seq_lengths.append(len(input_ids))
            
            current_skill = []
            skills_in_doc = 0
            
            for i, label in enumerate(labels):
                if label == 1: 
                    if current_skill:
                        all_tokens_as_skills.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(current_skill)).strip())
                        skills_in_doc += 1
                    current_skill = [tokenizer.convert_ids_to_tokens(input_ids[i])]
                elif label == 2: 
                    current_skill.append(tokenizer.convert_ids_to_tokens(input_ids[i]))
                else:
                    if current_skill:
                        all_tokens_as_skills.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(current_skill)).strip())
                        skills_in_doc += 1
                        current_skill = []
            
            if current_skill:
                all_tokens_as_skills.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(current_skill)).strip())
                skills_in_doc += 1
            
            skill_counts_per_doc.append(skills_in_doc)

    all_skills_cleaned = [s.replace("##", "") for s in all_tokens_as_skills]
    skill_freq = Counter(all_skills_cleaned)
    df_skills = pd.DataFrame(skill_freq.most_common(20), columns=['Skill', 'Count'])

    print(f"Total de sequências: {len(seq_lengths)}")
    print(f"Média de Skills por documento: {sum(skill_counts_per_doc)/len(skill_counts_per_doc):.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.bar(df_skills['Skill'], df_skills['Count'], color='green')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Skills (Após Filtragem NLTK)')
    plt.tight_layout()
    plt.savefig("eda_processado_skills.png")

    plt.figure(figsize=(8, 5))
    plt.hist(seq_lengths, bins=20, color='blue', edgecolor='black')
    plt.title('Distribuição de Tokens por Sequência')
    plt.savefig("eda_processado_tokens.png")
    
    with open("lista_skills_pos_processamento.txt", "w", encoding="utf-8") as f:
        for skill, count in skill_freq.most_common():
            f.write(f"{skill}: {count}\n")

    print("Gráficos gerados: eda_processado_skills.png e eda_processado_tokens.png")

if __name__ == "__main__":
    analyze_processed_data()