import json
from pathlib import Path


def analyze_ner_labels_in_folder(folder_path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"O caminho '{folder_path}' não é um diretório válido.")

    results = {}
    max_labels = 0
    files_with_more_labels = []

    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extrair entidades base (antes do :)
            unique_entities = {
                ann[2].split(":")[0].strip()
                for ann in data.get("annotations", [])
            }

            num_labels = 1 + (len(unique_entities) * 2)

            results[json_file.name] = {
                "entities": unique_entities,
                "num_labels": num_labels
            }

            if num_labels > max_labels:
                max_labels = num_labels
                files_with_more_labels = [json_file.name]
            elif num_labels == max_labels:
                files_with_more_labels.append(json_file.name)

        except Exception as e:
            print(f"❌ Erro ao processar {json_file.name}: {e}")

    # ---------- Relatório ----------
    print("\n📊 RELATÓRIO GLOBAL NER\n")

    unique_label_counts = sorted(
        set(r["num_labels"] for r in results.values())
    )

    print(f"✔️ Valores de num_labels encontrados: {unique_label_counts}")

    if len(unique_label_counts) == 1:
        print(f"🎯 Todos os ficheiros usam num_labels = {unique_label_counts[0]}")
    else:
        print("⚠️ Foram encontrados ficheiros com num_labels diferentes:")
        for fname, info in results.items():
            print(
                f" - {fname}: num_labels = {info['num_labels']} | entidades = {info['entities']}"
            )

    print("\n🔝 Valor máximo de num_labels encontrado:", max_labels)
    print("📂 Ficheiros com esse valor:")
    for f in files_with_more_labels:
        print(" -", f)

    return results


if __name__ == "__main__":
    analyze_ner_labels_in_folder(
        folder_path=r"C:\Users\duart\OneDrive\Ambiente de Trabalho\Faculdade\Samsung\CapStone\NerData\ResumesJsonAnnotated\ResumesJsonAnnotated"
    )

import json
from pathlib import Path

#Não Estava a gerar relatório então verifiquei se o caminho estava errado
def analyze_ner_labels_in_folder(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Erro: O caminho não existe: {folder.resolve()}")
        return

    json_files = list(folder.glob("*.json"))
    print(f"🔍 Ficheiros .json encontrados: {len(json_files)}")

    if len(json_files) == 0:
        print("⚠️ Verificação: A pasta está vazia ou os ficheiros não têm extensão .json")
        return

    results = {}
    max_labels = 0
    files_with_more_labels = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            unique_entities = {
                ann[2].split(":")[0].strip()
                for ann in data.get("annotations", [])
            }

            num_labels = 1 + (len(unique_entities) * 2)

            results[json_file.name] = {
                "entities": unique_entities,
                "num_labels": num_labels
            }

            if num_labels > max_labels:
                max_labels = num_labels
                files_with_more_labels = [json_file.name]
            elif num_labels == max_labels:
                files_with_more_labels.append(json_file.name)

        except Exception as e:
            print(f"❌ Erro ao processar {json_file.name}: {e}")

    print("\n📊 RELATÓRIO GLOBAL NER\n")
    unique_label_counts = sorted(set(r["num_labels"] for r in results.values()))
    print(f"✔️ Valores de num_labels encontrados: {unique_label_counts}")

    if len(unique_label_counts) == 1:
        print(f"🎯 Todos os ficheiros usam num_labels = {unique_label_counts[0]}")
    else:
        for fname, info in results.items():
            print(f" - {fname}: num_labels = {info['num_labels']} | entidades = {info['entities']}")

    print("\n🔝 Valor máximo de num_labels encontrado:", max_labels)
    return results

if __name__ == "__main__":
    path = r"C:\Users\duart\OneDrive\Ambiente de Trabalho\Faculdade\Samsung\CapStone\NerData\ResumesJsonAnnotated\ResumesJsonAnnotated"
    analyze_ner_labels_in_folder(path)