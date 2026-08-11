import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BiVSSConfig
from .datasets import PairedChangeDataset
from .infer import save_mask
from .metrics import binary_metrics, multiclass_metrics
from .model import BiVSSCD


def class_map(class_masks: dict[str, np.ndarray], prompts: list[str]) -> np.ndarray:
    if not class_masks:
        raise ValueError("class masks are required for multiclass evaluation")
    shape = next(iter(class_masks.values())).shape
    result = np.zeros(shape, dtype=np.uint8)
    for class_id, prompt in enumerate(prompts, start=1):
        result[np.asarray(class_masks[prompt]) > 0] = class_id
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BiVSS-CD on a supported dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "WHU-CD", "LEVIR-CD", "LEVIR-CD_512", "SECOND",
            "SECOND_building", "SECOND_tree", "SECOND_water",
            "SECOND_low_veg", "SECOND_surface", "SECOND_playground",
        ],
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--prompts", nargs="+", help="Override prompts from the YAML config.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions"
    predictions.mkdir(exist_ok=True)
    csv_path = args.output_dir / "per_sample.csv"
    completed: set[str] = set()
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            completed = {row["name"] for row in csv.DictReader(stream)}

    dataset = PairedChangeDataset(args.data_root, args.dataset)
    pairs = dataset.pairs()[: args.limit] if args.limit else dataset.pairs()
    config = BiVSSConfig.from_yaml(args.config, checkpoint=args.checkpoint)
    prompts = args.prompts or list(config.prompts)
    if not prompts:
        raise SystemExit("no prompts supplied; set prompts in the config or pass --prompts")
    model = BiVSSCD(config)
    binary_rows: list[dict[str, object]] = []
    fieldnames = ["name", "iou", "f1", "precision", "recall", "oa"]
    append = csv_path.is_file() and csv_path.stat().st_size > 0
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for pair in pairs:
            if pair.name in completed:
                continue
            result = model.predict(pair.image_t1, pair.image_t2, prompts)
            save_mask(result.binary_mask, predictions / pair.name)
            if args.dataset == "SECOND":
                semantic = class_map(result.class_masks or {}, prompts)
                save_mask(semantic, predictions / f"{Path(pair.name).stem}_semantic_preview.png")
                Image.fromarray(semantic).save(predictions / f"{Path(pair.name).stem}_semantic.png")
            target = np.asarray(Image.open(pair.label))
            if target.ndim == 3:
                target = target[..., 0]
            metrics = binary_metrics(result.binary_mask, target).to_dict()
            row: dict[str, object] = {"name": pair.name, **metrics}
            writer.writerow(row)
            stream.flush()
            binary_rows.append(row)

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "samples": len(all_rows),
        "binary": {
            key: float(np.mean([float(row[key]) for row in all_rows]))
            for key in fieldnames[1:]
        },
        "config": config.to_dict(),
        "prompts": prompts,
    }
    if args.dataset == "SECOND":
        predictions_all, targets_all = [], []
        for pair in pairs:
            pred_path = predictions / pair.name
            if not pred_path.is_file():
                continue
            # Binary predictions remain available for compatibility. Full semantic
            # maps are generated in a normal run and evaluated when present.
            semantic_path = predictions / f"{Path(pair.name).stem}_semantic.png"
            if semantic_path.is_file():
                predictions_all.append(np.asarray(Image.open(semantic_path)))
                targets_all.append(np.asarray(Image.open(pair.label)))
        if predictions_all:
            summary["multiclass"] = multiclass_metrics(
                np.stack(predictions_all), np.stack(targets_all), len(prompts) + 1
            )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "evaluation.log").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
