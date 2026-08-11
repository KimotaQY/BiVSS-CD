import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BiVSSConfig
from .model import BiVSSCD


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BiVSS-CD on one bi-temporal image pair.")
    parser.add_argument("--image-t1", required=True, type=Path)
    parser.add_argument("--image-t2", required=True, type=Path)
    parser.add_argument("--prompts", nargs="+", help="Override prompts from the YAML config.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint from the YAML file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BiVSSConfig.from_yaml(args.config, checkpoint=args.checkpoint)
    prompts = args.prompts or list(config.prompts)
    if not prompts:
        raise SystemExit("no prompts supplied; set prompts in the config or pass --prompts")
    result = BiVSSCD(config).predict(args.image_t1, args.image_t2, prompts)
    save_mask(result.binary_mask, args.output)
    class_dir = args.output.parent / f"{args.output.stem}_classes"
    for index, (prompt, mask) in enumerate((result.class_masks or {}).items()):
        save_mask(mask, class_dir / f"{index:02d}_{prompt.replace(' ', '_')}.png")
    metadata = {"prompts": prompts, "config": config.to_dict(), "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
