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

Recomenda-se utilizar um **ambiente virtual isolado** antes de instalar o projeto.

---

### Opção 1 — `venv` (recomendado para VS Code)

```bash
python -m venv .venv
```

Ativar:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Instalar o projeto:

```bash
pip install -e .
```

Para incluir ferramentas de desenvolvimento (notebooks, formatação, etc.):

```bash
pip install -e .[dev]
```

---

### Opção 2 — Conda

```bash
conda create -n cv_analyzer python=3.11
conda activate cv_analyzer
pip install -e .
```

---

### Sobre o comando `pip install -e .`

Este comando:

* instala o **projeto em modo editable**
* instala automaticamente **todas as dependências definidas no `pyproject.toml`**
* permite que alterações no código sejam refletidas **sem reinstalar o pacote**

---

### Nota sobre PyTorch (CPU vs GPU)

O `torch` pode exigir instalação específica dependendo da GPU.

**CPU:**

```bash
pip install torch
```

**CUDA (exemplo CUDA 12.1):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Consulte: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---
## Models / AIs

São usados:

* **YOLO** → deteção e criação de masks 
Links:

* YOLO: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)



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

### Extra
Para cancelar commit antes de push: 
```bash
git reset --soft HEAD~1
```


Para fazer commit e push apenas do .gitignore, tens de garantir que só esse ficheiro está staged: 
```bash
git reset
```

```bash
git add .gitignore
git commit -m "Atualiza .gitignore"
git push
```