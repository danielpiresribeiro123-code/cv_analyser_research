## Configuração do ambiente (.env)

Este projeto utiliza variáveis de ambiente para definir todos os caminhos importantes (dados, modelos, executáveis, etc.).
Por esse motivo, o ficheiro `.env` é **específico de cada computador** e **não deve ser enviado para o repositório**.

No repositório existe apenas um ficheiro de exemplo:

```
.env.example
```

Quando clonares o projeto, deves fazer o seguinte:

1. Criar uma cópia do ficheiro de exemplo:

```bash
cp .env.example .env
```

(no Windows podes simplesmente copiar e renomear o ficheiro manualmente)

2. Abrir o ficheiro `.env` e preencher a variável `PROJECT_ROOT` com o caminho onde o projeto está no teu computador.

Exemplo:

```env
PROJECT_ROOT=C:/Users/teu_utilizador/Desktop/CapStone
```


O resto do ficheiro funciona como configuração automática.

> O ficheiro `.env` é ignorado pelo Git através do `.gitignore`, garantindo que:
>
> * Caminhos pessoais não são enviados para o repositório
> * Cada utilizador pode usar o seu próprio computador sem conflitos

---

## Instalação

Recomendado: usar `venv` no VSCode

```bash
pip install torch numpy evaluate datasets transformers python-dotenv ultralytics
```

Ou com conda:

```bash
conda create -n 3d_reconstruction python=3.11.9
conda init
conda activate 3d_reconstruction
pip install -e .
```

O comando:

```bash
pip install -e .
```

instala automaticamente:

* o projeto em modo editable
* todas as dependências definidas no `pyproject.toml`

---
## Models / AIs

São usados:

* **YOLO** → deteção e criação de masks 
Links:

* YOLO: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

Documentação útil para YOLO segmentation:

* [https://docs.ultralytics.com/pt/tasks/segment/#train](https://docs.ultralytics.com/pt/tasks/segment/#train)
* [https://docs.ultralytics.com/pt/datasets/segment/coco/#dataset-yaml](https://docs.ultralytics.com/pt/datasets/segment/coco/#dataset-yaml)



## Remover todos os arquivos ignorados de uma vez (Melhor opção)
Se você adicionou várias regras ao .gitignore e quer limpar tudo de uma vez:
1. Commit as alterações do seu .gitignore primeiro.
2. Execute o seguinte comando para remover tudo o que está no cache e agora está ignorado:
```bash
git rm -r --cached .
```
3. Adicione novamente os arquivos (o Git agora vai ignorar os arquivos listados no .gitignore):
```bash
git add .
```
4. Faça o commit dessa limpeza:
```bash
git commit -m "Removendo arquivos rastreados que agora estão no .gitignore"
```
5. Faça o push (se necessário):
```bash
git push
```