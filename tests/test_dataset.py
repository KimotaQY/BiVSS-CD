from pathlib import Path

import pytest
from PIL import Image

from bivss_cd.datasets import PairedChangeDataset


def test_dataset_adapter_discovers_pairs(tmp_path: Path):
    for directory in ("A", "B", "label"):
        (tmp_path / directory).mkdir()
        Image.new("L", (2, 2)).save(tmp_path / directory / "sample.png")
    pairs = PairedChangeDataset(tmp_path, "LEVIR-CD").pairs()
    assert len(pairs) == 1
    assert pairs[0].name == "sample.png"


def test_dataset_adapter_rejects_incomplete_pair(tmp_path: Path):
    for directory in ("A", "B", "label"):
        (tmp_path / directory).mkdir()
    Image.new("L", (2, 2)).save(tmp_path / "A" / "sample.png")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        PairedChangeDataset(tmp_path, "WHU-CD").pairs()
