"""Where everything lives, relative to the project folder (so the project can be copied anywhere).

Override with environment variables when the data sits elsewhere:
  SHIPSOUND_DATA       folder with mel.npy + clips.parquet   (default: <project>/data)
  SHIPSOUND_AUDIO_ZIP  last part of the split audio zip      (default: <project>/gdrive_download/audio/allclass_5hours.zip)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SHIPSOUND_DATA", ROOT / "data"))
RUNS_DIR = ROOT / "runs"
CONFIG_DIR = ROOT / "configs"
AUDIO_ZIP = Path(os.environ.get("SHIPSOUND_AUDIO_ZIP", ROOT / "gdrive_download" / "audio" / "allclass_5hours.zip"))
