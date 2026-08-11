import numpy as np
import pytest
from pathlib import Path

from bivss_cd.config import BiVSSConfig
from bivss_cd.metrics import binary_metrics, multiclass_metrics
from bivss_cd.sam3_adapter import verify_sam3_checkout


def test_config_validation():
    with pytest.raises(ValueError, match="iou_threshold"):
        BiVSSConfig(iou_threshold=1.1)
    with pytest.raises(ValueError, match="new_det_thresh"):
        BiVSSConfig(score_threshold_detection=0.7, new_det_thresh=0.6)
    with pytest.raises(ValueError, match="CUDA"):
        BiVSSConfig(checkpoint="weights.pt", device="cpu")


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


def test_binary_metrics_known_values():
    result = binary_metrics([[1, 1], [0, 0]], [[1, 0], [1, 0]])
    assert result.iou == pytest.approx(1 / 3)
    assert result.f1 == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.oa == pytest.approx(0.5)


def test_multiclass_metrics_perfect_prediction():
    labels = np.array([[0, 1], [2, 2]])
    result = multiclass_metrics(labels, labels, num_classes=3)
    assert result["mean_iou"] == 1.0
    assert result["mean_f1"] == 1.0


def test_pinned_sam3_checkout_exposes_required_interface():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    assert verify_sam3_checkout(root / "third_party" / "sam3").name == "sam3"
