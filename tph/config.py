"""Central configuration. Every tunable lives here so the pipeline is reproducible."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
PLOTS = DOCS / "plots"          # served by GitHub Pages
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "vetter.joblib"

SCREENED_CSV = RESULTS / "screened.csv"
CANDIDATES_CSV = RESULTS / "candidates.csv"
VALIDATION_JSON = RESULTS / "validation.json"

# --- Target selection (TIC catalog cuts) -----------------------------------
TMAG_RANGE = (9.0, 13.0)        # fainter than the bright stars NASA prioritized, still good SNR
TEFF_RANGE = (3800.0, 6800.0)   # FGK + early M dwarfs
RADIUS_RANGE = (0.4, 1.6)       # main-sequence radii (solar units)
MAX_CONTAMINATION = 0.2         # TIC contratio: flux from neighbours / target flux
FIELD_RADIUS_DEG = 0.5

# --- Light curve handling ---------------------------------------------------
AUTHOR_PREFERENCE = ["SPOC", "TESS-SPOC", "QLP"]
MAX_SECTORS = 4
BIN_MINUTES = 10.0
DETREND_WINDOW_DAYS = 0.75      # ~3x the longest transit we search for

# --- Transit search ---------------------------------------------------------
PERIOD_MIN = 0.5
PERIOD_MAX = 30.0
DURATIONS = [0.03, 0.05, 0.08, 0.12, 0.17, 0.25]   # days
MAX_PERIODS = 200_000
MAX_SIGNALS_PER_STAR = 2

# --- Detection thresholds ---------------------------------------------------
MIN_SDE = 9.0
MIN_SNR = 7.0
MIN_TRANSITS = 3

# --- Ranking ----------------------------------------------------------------
STRONG_SCORE = 0.8
CANDIDATE_SCORE = 0.5
MAX_PLANET_RADIUS_EARTH = 25.0   # bigger than ~2.2 Jupiter radii is almost surely a star
