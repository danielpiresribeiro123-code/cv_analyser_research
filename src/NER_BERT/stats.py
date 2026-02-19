import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURAÇÃO
# ==========================================
PASTA_DO_MODELO = "data/processed/results_ner_lowercase"
PASTA_DESTINO_RELATORIO = "data/comparar_NER/NER"
NAME = "rel_NER"  # Define aqui o nome do ficheiro
# ==========================================

def load_trainer_state(output_dir):
    state_path = os.path.join(output_dir, "trainer_state.json")
    
    if not os.path.exists(state_path):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            state_path = os.path.join(output_dir, latest_checkpoint, "trainer_state.json")
    
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Não foi possível encontrar trainer_state.json em {output_dir}")
        
    with open(state_path, "r") as f:
        return json.load(f)

def generate_metrics_report(input_dir, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    
    state = load_trainer_state(input_dir)
    log_history = state["log_history"]
    df = pd.DataFrame(log_history)

    df_eval = df[df["eval_loss"].notna()].copy() if "eval_loss" in df.columns else pd.DataFrame()
    df_train = df[df["loss"].notna()].copy() if "loss" in df.columns else pd.DataFrame()

    if df_eval.empty:
        print(f"❌ Erro: Dados de avaliação não encontrados em {input_dir}")
        return

    # 1. Guardar Tabela e Resumo
    summary_table = df_eval.sort_values("epoch")
    summary_table.to_csv(f"{save_dir}/{file_name}_tabela.csv", index=False)
    
    with open(f"{save_dir}/{file_name}_resumo.txt", "w", encoding="utf-8") as f:
        f.write(f"RELATÓRIO NER: {file_name}\n")
        f.write("-" * 30 + "\n")
        f.write(summary_table.to_string(index=False))

    # 2. Gerar Gráficos
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.suptitle(f'Métricas do Modelo: {file_name}', fontsize=18)

    # Loss
    if not df_train.empty:
        sns.lineplot(ax=axes[0, 0], data=df_train, x="epoch", y="loss", label="Train Loss", marker="o")
    if "eval_loss" in df_eval.columns:
        sns.lineplot(ax=axes[0, 0], data=df_eval, x="epoch", y="eval_loss", label="Eval Loss", marker="s")
    axes[0, 0].set_title("Loss")

    # F1-Score
    if "eval_f1" in df_eval.columns:
        sns.lineplot(ax=axes[0, 1], data=df_eval, x="epoch", y="eval_f1", marker="o", color="green")
        axes[0, 1].set_title("F1-Score")

    # Precision vs Recall
    if "eval_precision" in df_eval.columns:
        sns.lineplot(ax=axes[1, 0], data=df_eval, x="epoch", y="eval_precision", label="Precision", marker="^")
        sns.lineplot(ax=axes[1, 0], data=df_eval, x="epoch", y="eval_recall", label="Recall", marker="v")
        axes[1, 0].set_title("Precision vs Recall")

    # Accuracy
    if "eval_accuracy" in df_eval.columns:
        sns.lineplot(ax=axes[1, 1], data=df_eval, x="epoch", y="eval_accuracy", marker="p", color="orange")
        axes[1, 1].set_title("Accuracy")

    # Guardar imagem e fechar
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_dir}/{file_name}_graficos.png", dpi=300)
    plt.close()

    print(f"✅ Sucesso! Ficheiros gerados em '{save_dir}':")
    print(f"   - {file_name}_graficos.png")
    print(f"   - {file_name}_tabela.csv")
    print(f"   - {file_name}_resumo.txt")

if __name__ == "__main__":
    generate_metrics_report(PASTA_DO_MODELO, PASTA_DESTINO_RELATORIO, NAME)