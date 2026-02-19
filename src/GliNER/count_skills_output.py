import os
import json
import pandas as pd
import matplotlib.pyplot as plt

input_path = 'src/GliNER/output_test'
output_path = 'data/comparar_NER/GliNER/skills_frequencia'

if not os.path.exists(output_path):
    os.makedirs(output_path)

all_skills = []

for file_name in os.listdir(input_path):
    if file_name.endswith('.json'):
        with open(os.path.join(input_path, file_name), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                all_skills.extend(data.get("skills", []))
            except:
                continue

counts = pd.Series(all_skills).value_counts()
total = len(all_skills)
percentages = (counts / total) * 100

df_top_10 = pd.DataFrame({
    'Skill': counts.index[:10],
    'Frequency': counts.values[:10],
    '(%)': percentages.values[:10].round(2)
})

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

tabela = ax.table(cellText=df_top_10.values, 
                  colLabels=df_top_10.columns, 
                  cellLoc='center', 
                  loc='center',
                  colColours=["#e67e22"]*3)

tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.2, 1.2)

plt.savefig(os.path.join(output_path, 'top_10_skills.png'), dpi=300, bbox_inches='tight')
plt.show()