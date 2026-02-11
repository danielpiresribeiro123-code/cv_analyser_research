import pandas as pd
from pathlib import Path
from utils.path_loader import load_paths
paths = load_paths()

PROJECT_ROOT = paths["project_root"]
# print(PROJECT_ROOT)

NER_RAW_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Json"