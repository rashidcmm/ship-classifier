import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import build_notebooks  # noqa: E402

EXPECTED = ["00_setup_check", "01_explore_data", "02_sound_to_spectrogram", "03_classical_baseline",
            "05_training_the_real_model", "08_trustworthy_confidence", "11_results_and_five_classes",
            "13_six_classes_and_features", "18_distance_buckets_and_regression", "22_other_class_always",
            "23_feature_cost_and_mel_bands", "24_merged_classes_and_panns_with_other", "25_cnn_with_other",
            "26_unknown_ships_cnn"]


def test_parse_percent_cells():
    cells = build_notebooks.parse_percent("# %% [markdown]\n# # Title\n# text\n\n# %%\nx = 1\nprint(x)\n")
    assert cells == [("markdown", "# Title\ntext"), ("code", "x = 1\nprint(x)")]


def test_build_creates_valid_notebooks(tmp_path):
    paths = build_notebooks.build_all(out_dir=tmp_path)
    names = sorted(p.stem for p in paths)
    assert names == EXPECTED
    for p in paths:
        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        assert nb.cells[0].cell_type == "markdown"
        assert any(c.cell_type == "code" for c in nb.cells)
