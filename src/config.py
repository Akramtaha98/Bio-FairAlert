"""
Central configuration for the Bio-FairAlert analysis pipeline.

Every path used by the scripts in this folder is defined here, once, instead
of being hardcoded in each script. All defaults are relative to this
repository so the pipeline runs out of the box after the datasets are placed
in the expected folders (see README, "Getting the data"). Any path can be
overridden with an environment variable if you prefer a different layout.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)

# Raw dataset locations. Datasets are public and not included in this
# repository; download them separately (see README, "Getting the data").
DATA_DIR = os.environ.get("OULAD_DATA_DIR", os.path.join(REPO_ROOT, "data", "oulad"))
MOOC_RAW_DIR = os.environ.get("MOOC_RAW_DIR", os.path.join(REPO_ROOT, "data", "mooc_raw", "act-mooc"))

# Output locations for experiment results (JSON) and generated figures (PNG).
OUT_DIR = os.environ.get("BIOFAIRALERT_OUT_DIR", os.path.join(REPO_ROOT, "results"))
MEDIA_DIR = os.environ.get("BIOFAIRALERT_MEDIA_DIR", os.path.join(REPO_ROOT, "results", "figures"))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)
