from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import BiVSSConfig
from .masks import consensus_fusion
from .sam3_adapter import SAM3Session, build_predictor, frame_objects
from .sscce import combine_changes, semantic_spatial_changes
from .types import ChangeResult
from .video import pseudo_video


class BiVSSCD:
    """Training-free bidirectional semantic object change detector."""

    def __init__(self, config: BiVSSConfig | str | Path, predictor: Any | None = None):
        self.config = BiVSSConfig.from_yaml(config) if isinstance(config, (str, Path)) else config
        if not isinstance(self.config, BiVSSConfig):
            raise TypeError("config must be a BiVSSConfig or a YAML path")
        self.predictor = predictor

    @staticmethod
    def _prompts(prompts: str | Sequence[str]) -> list[str]:
        values = [prompts] if isinstance(prompts, str) else list(prompts)
        values = [value.strip() for value in values if isinstance(value, str) and value.strip()]
        if not values:
            raise ValueError("at least one non-empty text prompt is required")
        return values

    def _predict_direction(
        self,
        image_t1: str | Path,
        image_t2: str | Path,
        prompts: list[str],
        reverse: bool,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        class_masks: dict[str, np.ndarray] = {}
        with pseudo_video(image_t1, image_t2, reverse=reverse) as video_path:
            for prompt in prompts:
                with SAM3Session(self.predictor, video_path) as session:
                    session.add_prompt(prompt)
                    frames = session.propagate()
                first, last = frames[min(frames)], frames[max(frames)]
                changes = semantic_spatial_changes(
                    frame_objects(first),
                    frame_objects(last),
                    image_shape=shape,
                    iou_threshold=self.config.iou_threshold,
                )
                class_masks[prompt] = combine_changes(changes, shape)
        return class_masks

    def predict(
        self,
        image_t1: str | Path,
        image_t2: str | Path,
        prompts: str | Sequence[str],
    ) -> ChangeResult:
        prompt_list = self._prompts(prompts)
        with Image.open(image_t1) as image:
            shape = (image.height, image.width)
        with Image.open(image_t2) as image:
            if (image.height, image.width) != shape:
                raise ValueError("bi-temporal images must have the same dimensions")
        if self.predictor is None:
            self.predictor = build_predictor(self.config)

        forward = self._predict_direction(image_t1, image_t2, prompt_list, False, shape)
        backward = self._predict_direction(image_t1, image_t2, prompt_list, True, shape)
        # Preserve per-prompt consensus for semantic inspection.
        class_masks = {
            prompt: consensus_fusion(forward[prompt], backward[prompt], self.config.consensus)
            for prompt in prompt_list
        }
        # Paper behavior (`baseline_bi_ssccev4`): masks from every prompt are
        # accumulated within each direction, then both directional counts are
        # added and pixels with at least two votes are retained. This differs
        # from intersecting each prompt first and then taking their union when
        # more than one prompt is used.
        vote_count = np.zeros(shape, dtype=np.int16)
        for prompt in prompt_list:
            vote_count += forward[prompt].astype(np.int16)
            vote_count += backward[prompt].astype(np.int16)
        threshold = 2 if self.config.consensus == "intersection" else 1
        binary = (vote_count >= threshold).astype(np.uint8)
        intermediates = {"forward": forward, "backward": backward} if self.config.save_intermediates else {}
        if self.config.save_intermediates:
            intermediates["vote_count"] = vote_count
        return ChangeResult(binary_mask=binary, class_masks=class_masks, intermediates=intermediates)
