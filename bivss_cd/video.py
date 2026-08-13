from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from PIL import Image


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


@contextmanager
def pseudo_video(image_t1: str | Path, image_t2: str | Path, reverse: bool = False) -> Iterator[Path]:
    images = [load_rgb(image_t1), load_rgb(image_t2)]
    if images[0].size != images[1].size:
        raise ValueError(f"input image sizes differ: {images[0].size} and {images[1].size}")
    if reverse:
        images.reverse()
    with TemporaryDirectory(prefix="bivss_cd_") as directory:
        root = Path(directory)
        # Match the verified paper implementation's `gen_frame_v2` output.
        for index, image in enumerate(images, start=1):
            image.save(root / f"{index}.jpg", format="JPEG", quality=100)
        yield root
