from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    name: str
    image_t1: Path
    image_t2: Path
    label: Path


class PairedChangeDataset:
    """Adapter for LEVIR-CD, WHU-CD, and SECOND directory layouts."""

    LAYOUTS = {
        "levir-cd": ("A", "B", "label"),
        "levir-cd_512": ("A", "B", "label"),
        "whu-cd": ("A", "B", "label"),
        "second": ("im1", "im2", "label"),
        "second_building": ("im1", "im2", "label"),
        "second_tree": ("im1", "im2", "label"),
        "second_water": ("im1", "im2", "label"),
        "second_low_veg": ("im1", "im2", "label"),
        "second_surface": ("im1", "im2", "label"),
        "second_playground": ("im1", "im2", "label"),
    }

    def __init__(self, root: str | Path, dataset: str):
        key = dataset.lower()
        if key not in self.LAYOUTS:
            raise ValueError(f"unsupported dataset: {dataset}")
        root = Path(root)
        split = root / "test" if (root / "test").is_dir() else root
        first, second, label = self.LAYOUTS[key]
        self.directories = split / first, split / second, split / label
        for directory in self.directories:
            if not directory.is_dir():
                raise FileNotFoundError(f"dataset directory not found: {directory}")

    def pairs(self) -> list[ImagePair]:
        first, second, labels = self.directories
        pairs: list[ImagePair] = []
        for image in sorted(p for p in first.iterdir() if p.suffix.lower() in SUPPORTED_IMAGES):
            counterpart, label = second / image.name, labels / image.name
            if not counterpart.is_file() or not label.is_file():
                raise FileNotFoundError(f"incomplete image pair for {image.name}")
            pairs.append(ImagePair(image.name, image, counterpart, label))
        if not pairs:
            raise RuntimeError(f"no supported images found in {first}")
        return pairs
