from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChangeResult:
    """Result returned by :meth:`BiVSSCD.predict`."""

    binary_mask: np.ndarray
    class_masks: dict[str, np.ndarray] | None = None
    intermediates: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.binary_mask)
        if mask.ndim != 2:
            raise ValueError("binary_mask must be a two-dimensional array")
        object.__setattr__(self, "binary_mask", (mask > 0).astype(np.uint8))
