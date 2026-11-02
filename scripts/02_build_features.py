"""Compatibility entry point for the complete data-preparation notebook.

Feature engineering is included in 01_preprocess.py; run either entry point,
not both. Keeping one implementation preserves the submitted notebook's logic.
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("01_preprocess.py")), run_name="__main__")
