import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from bivss_cd.config import BiVSSConfig
from bivss_cd.metrics import BinaryConfusion, binary_metrics, multiclass_metrics
from bivss_cd.sam3_adapter import _verify_imported_sam3, verify_sam3_checkout


def test_config_validation():
    with pytest.raises(ValueError, match="iou_threshold"):
        BiVSSConfig(iou_threshold=1.1)
    with pytest.raises(ValueError, match="new_det_thresh"):
        BiVSSConfig(score_threshold_detection=0.7, new_det_thresh=0.6)
    with pytest.raises(ValueError, match="CUDA"):
        BiVSSConfig(checkpoint="weights.pt", device="cpu")
    with pytest.raises(ValueError, match="gpus_to_use"):
        BiVSSConfig(gpus_to_use=[])
    with pytest.raises(ValueError, match="instance_iou_threshold"):
        BiVSSConfig(instance_iou_threshold=1.1)
    with pytest.raises(ValueError, match="t12_min_instance_area"):
        BiVSSConfig(t12_min_instance_area=-1)


def test_dataset_config_loads_prompts(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("prompts: [tree, forest]\n", encoding="utf-8")
    config = BiVSSConfig.from_yaml(config_path)
    assert config.prompts == ("tree", "forest")


@pytest.mark.parametrize(
    ("filename", "prompts", "thresholds"),
    [
        ("whu_cd.yaml", ("roof",), (0.30, 0.55, 0.60)),
        ("levir_cd.yaml", ("roof",), (0.10, 0.10, 0.15)),
        ("levir_cd_512.yaml", ("roof",), (0.30, 0.35, 0.40)),
        ("second_building.yaml", ("building",), (0.10, 0.30, 0.40)),
        ("second_tree.yaml", ("tree", "forest"), (0.10, 0.15, 0.20)),
        ("second_water.yaml", ("water,river,pond,sea",), (0.50, 0.10, 0.20)),
        ("second_low_veg.yaml", ("low vegetation", "grass"), (0.10, 0.20, 0.30)),
        ("second_surface.yaml", ("the ground", "bare land"), (0.40, 0.40, 0.45)),
        (
            "second_playground.yaml",
            ("football court", "basketball court", "baseball court"),
            (0.40, 0.35, 0.50),
        ),
    ],
)
def test_paper_dataset_configs(filename, prompts, thresholds):
    root = Path(__file__).resolve().parents[1]
    config = BiVSSConfig.from_yaml(root / "configs" / filename)
    assert config.prompts == prompts
    assert (
        config.iou_threshold,
        config.score_threshold_detection,
        config.new_det_thresh,
    ) == thresholds
    assert config.use_decoupled_selection is False
    assert config.use_instance_level_cd is False
    assert config.instance_iou_threshold == 0.30
    assert config.t12_min_instance_area == 0
    assert config.cd_min_instance_area == 0
    assert config.gpus_to_use == (0,)


def test_binary_metrics_known_values():
    result = binary_metrics([[1, 1], [0, 0]], [[1, 0], [1, 0]])
    assert result.iou == pytest.approx(1 / 3)
    assert result.f1 == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.oa == pytest.approx(0.5)


def test_global_confusion_is_not_mean_of_per_image_metrics():
    confusion = BinaryConfusion()
    confusion.update(np.ones((1, 1)), np.ones((1, 1)))
    confusion.update(np.ones((3, 3)), np.zeros((3, 3)))
    result = confusion.metrics()
    assert result.iou == pytest.approx(0.1)
    assert result.precision == pytest.approx(0.1)
    assert result.recall == pytest.approx(1.0)


def test_multiclass_metrics_perfect_prediction():
    labels = np.array([[0, 1], [2, 2]])
    result = multiclass_metrics(labels, labels, num_classes=3)
    assert result["mean_iou"] == 1.0
    assert result["mean_f1"] == 1.0


def test_pinned_sam3_checkout_exposes_required_interface():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    assert verify_sam3_checkout(root / "third_party" / "sam3").name == "sam3"


def test_imported_sam3_must_come_from_pinned_checkout(tmp_path):
    checkout = tmp_path / "third_party" / "sam3"
    expected = checkout / "sam3"
    expected.mkdir(parents=True)
    module = SimpleNamespace(__file__=str(expected / "__init__.py"))
    assert _verify_imported_sam3(module, checkout) == expected.resolve()

    conflicting = tmp_path / "site-packages" / "sam3" / "__init__.py"
    conflicting.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="does not come from"):
        _verify_imported_sam3(SimpleNamespace(__file__=str(conflicting)), checkout)
