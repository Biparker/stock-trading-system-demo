import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # update_cash, drift_check
sys.path.insert(0, str(ROOT / "dashboard"))   # data_loader, charts
