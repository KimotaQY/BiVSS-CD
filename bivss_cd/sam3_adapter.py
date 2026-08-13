from __future__ import annotations

import inspect
import hashlib
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .config import BiVSSConfig


REQUIRED_PREDICTOR_PARAMETERS = {
    "score_threshold_detection",
    "new_det_thresh",
    "use_decoupled_selection",
}


def runtime_fingerprint(config: BiVSSConfig) -> dict[str, Any]:
    """Return reproducibility-critical runtime and model identifiers."""
    import PIL
    import sam3
    import torch
    from PIL import features

    project_root = Path(__file__).resolve().parents[1]
    checkout = verify_sam3_checkout(project_root / "third_party" / "sam3")
    imported_sam3 = _verify_imported_sam3(sam3, checkout)

    checkpoint = Path(config.checkpoint).expanduser().resolve() if config.checkpoint else None
    digest = None
    if checkpoint and checkpoint.is_file():
        hasher = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "inference_implementation": "verified-test-script",
        "sam3_path": str(imported_sam3),
        "sam3_revision": revision,
        "checkpoint_sha256": digest,
        "pillow": PIL.__version__,
        "libjpeg": features.version_codec("jpg"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _verify_imported_sam3(sam3_module: Any, checkout: Path) -> Path:
    imported_sam3 = Path(sam3_module.__file__).resolve().parent
    expected_sam3 = (checkout / "sam3").resolve()
    if imported_sam3 != expected_sam3:
        raise RuntimeError(
            "the imported SAM3 package does not come from BiVSS-CD's pinned "
            f"submodule: imported {imported_sam3}, expected {expected_sam3}. "
            "Uninstall the conflicting package and run "
            "`pip install -e third_party/sam3`."
        )
    return imported_sam3


def verify_sam3_checkout(root: str | Path) -> Path:
    root = Path(root)
    predictor_file = root / "sam3" / "model" / "sam3_video_predictor.py"
    base_file = root / "sam3" / "model" / "sam3_video_base.py"
    if not predictor_file.is_file() or not base_file.is_file():
        raise RuntimeError(
            "BiVSS-CD's SAM3 submodule is missing. Run "
            "`git submodule update --init --recursive`."
        )
    predictor_source = predictor_file.read_text(encoding="utf-8")
    base_source = base_file.read_text(encoding="utf-8")
    missing = [name for name in REQUIRED_PREDICTOR_PARAMETERS if name not in predictor_source]
    if missing or "Decoupled Selection" not in base_source:
        raise RuntimeError(
            "the checked-out SAM3 revision does not expose the BiVSS-CD interface; "
            f"missing markers: {missing or ['Decoupled Selection']}"
        )
    return root


def build_predictor(config: BiVSSConfig) -> Any:
    if not config.checkpoint:
        raise ValueError("config.checkpoint is required to build the SAM3 predictor")
    checkpoint = Path(config.checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

    project_root = Path(__file__).resolve().parents[1]
    verify_sam3_checkout(project_root / "third_party" / "sam3")

    try:
        import sam3
        import torch
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 is not importable. Initialize the submodule and install it with "
            "`pip install -e third_party/sam3`."
        ) from exc

    _verify_imported_sam3(sam3, project_root / "third_party" / "sam3")

    kwargs = {
        "checkpoint_path": str(checkpoint),
        "score_threshold_detection": config.score_threshold_detection,
        "new_det_thresh": config.new_det_thresh,
        "use_decoupled_selection": config.use_decoupled_selection,
        "gpus_to_use": (
            list(range(torch.cuda.device_count()))
            if config.gpus_to_use == "all"
            else list(config.gpus_to_use)
        ),
    }
    if not kwargs["gpus_to_use"]:
        raise RuntimeError("no CUDA devices are available to build the SAM3 predictor")
    signature = inspect.signature(build_sam3_video_predictor)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        unsupported = set(kwargs) - set(signature.parameters)
        if unsupported:
            raise RuntimeError(f"SAM3 builder does not support required arguments: {sorted(unsupported)}")
    return build_sam3_video_predictor(**kwargs)


class SAM3Session:
    """Small lifecycle-safe adapter around SAM3's request protocol."""

    def __init__(self, predictor: Any, resource_path: str | Path):
        self.predictor = predictor
        self.resource_path = str(resource_path)
        self.session_id: str | None = None

    def __enter__(self) -> "SAM3Session":
        response = self.predictor.handle_request(
            request={"type": "start_session", "resource_path": self.resource_path}
        )
        self.session_id = response["session_id"]
        self.predictor.handle_request(
            request={"type": "reset_session", "session_id": self.session_id}
        )
        return self

    def add_prompt(self, prompt: str) -> None:
        if self.session_id is None:
            raise RuntimeError("session is not open")
        self.predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )

    def propagate(self) -> dict[int, dict[str, Any]]:
        if self.session_id is None:
            raise RuntimeError("session is not open")
        responses: dict[int, dict[str, Any]] = {}
        stream: Iterator[dict[str, Any]] = self.predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "both",
            }
        )
        for response in stream:
            responses[int(response["frame_index"])] = response["outputs"]
        if not responses:
            raise RuntimeError("SAM3 returned no propagated frames")
        return responses

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.session_id is not None:
            self.predictor.handle_request(
                request={"type": "close_session", "session_id": self.session_id}
            )
            self.session_id = None


def frame_objects(frame: dict[str, Any]) -> dict[int, dict[str, Any]]:
    ids = frame.get("out_obj_ids")
    masks = frame.get("out_binary_masks")
    if ids is None or masks is None:
        return {}
    to_numpy = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    ids_array, masks_array = to_numpy(ids), to_numpy(masks)
    probabilities = frame.get("out_probs")
    boxes = frame.get("out_boxes_xywh")
    probabilities = to_numpy(probabilities) if probabilities is not None else [None] * len(ids_array)
    boxes = to_numpy(boxes) if boxes is not None else [None] * len(ids_array)
    return {
        int(object_id): {
            "mask": np.squeeze(to_numpy(mask)),
            "probability": probability,
            "box": box,
        }
        for object_id, mask, probability, box in zip(ids_array, masks_array, probabilities, boxes)
    }
