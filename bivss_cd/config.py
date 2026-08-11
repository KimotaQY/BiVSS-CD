from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BiVSSConfig:
    checkpoint: str | None = None
    device: str = "cuda"
    prompts: tuple[str, ...] = ()
    iou_threshold: float = 0.30
    score_threshold_detection: float = 0.55
    new_det_thresh: float = 0.60
    use_decoupled_selection: bool = True
    consensus: str = "intersection"
    save_intermediates: bool = False

    def __post_init__(self) -> None:
        prompts = tuple(self.prompts)
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must contain only non-empty strings")
        object.__setattr__(self, "prompts", prompts)
        for name in ("iou_threshold", "score_threshold_detection", "new_det_thresh"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.new_det_thresh < self.score_threshold_detection:
            raise ValueError("new_det_thresh must not be lower than score_threshold_detection")
        if self.consensus not in {"intersection", "union"}:
            raise ValueError("consensus must be 'intersection' or 'union'")
        if self.device == "cpu" and self.checkpoint is not None:
            raise ValueError("the SAM3 video predictor requires CUDA; device='cpu' is test-only")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "BiVSSConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        if "prompts" in values:
            values["prompts"] = tuple(values["prompts"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
