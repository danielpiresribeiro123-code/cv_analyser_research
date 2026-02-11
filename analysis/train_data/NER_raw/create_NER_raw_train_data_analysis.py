import pandas as pd
from pathlib import Path
from utils.path_loader import load_paths
paths = load_paths()

PROJECT_ROOT = paths["project_root"]
# print(PROJECT_ROOT)

NER_RAW_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Json"

from analysis.train_data.NER_raw.support.io_utils import (load_all_raw_ner,)

from analysis.train_data.NER_raw.support.validation import (validate_offsets,count_overlapping_spans,)

from analysis.train_data.NER_raw.support.statistics import (extract_all_skills,skill_frequency,dataset_basic_stats,)
