# CV GYRO – Research and Model Development

Research and experimentation repository for **CV GYRO**, developed as a team capstone project for the **Samsung Innovation Campus AI Course at NOVA FCT**.

This repository documents the work behind the CV analysis application: dataset preparation, exploratory data analysis, model training, information extraction, and evaluation.

## Final Application

The final application, including its FastAPI backend and React interface, is maintained separately:

**[CV GYRO – Web Application](https://github.com/danielpiresribeiro123-code/cv_analyser_website)**

Use this repository to explore the development process and experiments. Use the application repository to run the web interface.

## Research Areas

- **Dataset preparation:** annotation conversion, dataset splitting, image processing, and data quality experiments.
- **Exploratory data analysis:** CV layouts, image characteristics, text lengths, and skill distributions.
- **Layout detection:** YOLO training and evaluation for text blocks, photos, and skill tables.
- **Text extraction:** OCR experiments, image preprocessing, and extraction from detected regions.
- **Named entity recognition:** BERT training experiments and GLiNER-based information extraction.
- **Document understanding:** experimental LayoutLMv3 training and dataset preparation.
- **Information processing:** combining model outputs with regular expressions and exploring candidate ranking.
- **Evaluation:** metrics, plots, and analysis of model behaviour.

Experimental approaches included here were not necessarily adopted in the final application.

## Repository Structure

| Path | Contents |
|---|---|
| `src/train_yolo/` | YOLO training scripts and dataset preparation utilities |
| `src/NER_BERT/` | BERT data preparation, training, and analysis |
| `src/GliNER/` | GLiNER extraction, preprocessing, and analysis |
| `src/LayoutLMv3/` | Experimental document understanding workflows |
| `src/ocr/` | OCR and image preprocessing experiments |
| `src/analysis/` | Dataset exploration and model evaluation |
| `src/data_edit_files_related/` | File organisation and dataset utilities |
| `src/data_edit_quality/` | Image degradation and quality experiments |
| `src/input_processing/` | CV filtering and PDF-to-image conversion |
| `src/info_extraction/` | Contact extraction, text processing, and ranking |
| `src/pipelines/` | Experimental CV processing pipeline |
| `src/utils/` | Shared configuration utilities |
| `models/trained/` | Trained YOLO checkpoints |
| `evaluation_results/` | Evaluation summaries and plots |

## Technologies

Python, PyTorch, Ultralytics YOLO, Hugging Face Transformers and Datasets, BERT, GLiNER, LayoutLMv3, PaddleOCR, Tesseract, OpenCV, pandas, scikit-learn, and Jupyter.

## Working with the Experiments

This is a development repository containing notebooks and scripts from different stages of the project. It is not a single, fully packaged application.

Some experiments require external datasets, additional model downloads, or changes to local paths.

### 1. Create a Python environment

The project declares Python 3.10 or later.

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 2. Install the declared dependencies

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

The dependency list does not cover every experimental script. Check the imports and requirements of the experiment you intend to run.

OCR engines and GPU support may require separate installation and configuration.

### 3. Configure local paths

Copy `.env.example` to `.env` and configure:

```env
PROJECT_ROOT=/absolute/path/to/repository
DATASETS_ROOT=/absolute/path/to/datasets
DATASETS_SKILL_TABLE=/absolute/path/to/skill-table-dataset
MODELS_IN_ROOT=${PROJECT_ROOT}/models
MODELS_OUT_ROOT=/absolute/path/to/external/models
ENV=dev
```

The shared path loader expects all these variables. The project and dataset directories must exist.

Some scripts also contain hard-coded paths or dataset YAML references that must be updated separately. Keep your local `.env` out of version control.

### 4. Select an experiment

Start with the relevant notebook or script for the task you want to explore.

Before running it, check:

- Input dataset paths and annotation formats.
- Required model checkpoints.
- Output directories.
- CPU or GPU settings.
- Dependencies specific to that experiment.

## Scope and Limitations

- Training datasets are not fully included in this repository.
- Scripts and notebooks reflect different stages of development.
- Some files are exploratory, superseded, or dependent on a local setup.
- Saved metrics relate to particular experiments and should not be interpreted as performance guarantees for the final application.
- Reproducing an experiment requires its corresponding data, configuration, and model versions.

## Team

- Daniel Ribeiro
- Duarte Esteves
- Tomás Fialho

Developed for the Samsung Innovation Campus AI Course at NOVA FCT.
