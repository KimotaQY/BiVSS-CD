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


@pytest.mark.parametrize(
    ("dataset", "label_directory"),
    [
        ("SECOND_building", "building_label"),
        ("SECOND_surface", "ground_label"),
        ("SECOND_low_veg", "low_vegetation_label"),
        ("SECOND_playground", "playground_label"),
        ("SECOND_tree", "tree_label"),
        ("SECOND_water", "water_label"),
    ],
)
def test_second_category_uses_its_label_directory(tmp_path, dataset, label_directory):
    for directory in ("im1", "im2", label_directory):
        (tmp_path / directory).mkdir()
        Image.new("L", (2, 2)).save(tmp_path / directory / "sample.png")
    pair = PairedChangeDataset(tmp_path, dataset).pairs()[0]
    assert pair.label.parent.name == label_directory
