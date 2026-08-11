from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bivss_cd import BiVSSCD, BiVSSConfig


class FakePredictor:
    def __init__(self):
        self.counter = 0
        self.closed = []

    def handle_request(self, request):
        kind = request["type"]
        if kind == "start_session":
            self.counter += 1
            return {"session_id": f"session-{self.counter}"}
        if kind == "close_session":
            self.closed.append(request["session_id"])
        return {}

    def handle_stream_request(self, request):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        first = {"out_obj_ids": np.array([], dtype=int), "out_binary_masks": np.empty((0, 8, 8))}
        last = {
            "out_obj_ids": np.array([1]),
            "out_binary_masks": np.array([mask]),
            "out_probs": np.array([0.9]),
            "out_boxes_xywh": np.array([[0.25, 0.25, 0.5, 0.5]]),
        }
        yield {"frame_index": 0, "outputs": first}
        yield {"frame_index": 1, "outputs": last}


def write_image(path: Path, size=(8, 8)):
    Image.new("RGB", size, "white").save(path)


def test_predict_public_interface_and_session_cleanup(tmp_path):
    t1, t2 = tmp_path / "t1.png", tmp_path / "t2.png"
    write_image(t1)
    write_image(t2)
    predictor = FakePredictor()
    result = BiVSSCD(BiVSSConfig(device="cpu"), predictor=predictor).predict(t1, t2, "building")
    assert result.binary_mask.shape == (8, 8)
    assert set(result.class_masks) == {"building"}
    assert len(predictor.closed) == 2


def test_predict_rejects_empty_prompts_and_size_mismatch(tmp_path):
    t1, t2 = tmp_path / "t1.png", tmp_path / "t2.png"
    write_image(t1)
    write_image(t2, (9, 8))
    model = BiVSSCD(BiVSSConfig(device="cpu"), predictor=FakePredictor())
    with pytest.raises(ValueError, match="prompt"):
        model.predict(t1, t2, [])
    with pytest.raises(ValueError, match="dimensions"):
        model.predict(t1, t2, "building")
