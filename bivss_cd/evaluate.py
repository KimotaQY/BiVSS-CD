import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .config import BiVSSConfig
from .datasets import PairedChangeDataset
from .infer import save_mask
from .metrics import BinaryConfusion, binary_metrics, multiclass_metrics
from .model import BiVSSCD
from .sam3_adapter import runtime_fingerprint


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
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite all selected samples.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions"
    predictions.mkdir(exist_ok=True)
    csv_path = args.output_dir / "per_sample.csv"
    completed: set[str] = set()
    if csv_path.is_file() and not args.force:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            completed = {row["name"] for row in csv.DictReader(stream)}

    dataset = PairedChangeDataset(args.data_root, args.dataset)
    pairs = dataset.pairs()[: args.limit] if args.limit else dataset.pairs()
    config = BiVSSConfig.from_yaml(args.config, checkpoint=args.checkpoint)
    prompts = args.prompts or list(config.prompts)
    if not prompts:
        raise SystemExit("no prompts supplied; set prompts in the config or pass --prompts")
    runtime = runtime_fingerprint(config)
    run_spec = {
        "bivss_cd_version": __version__,
        "dataset": args.dataset,
        "config": config.to_dict(),
        "prompts": prompts,
        "runtime": runtime,
    }
    run_id = hashlib.sha256(
        json.dumps(run_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.is_file() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_id") != run_id:
            raise SystemExit(
                "the output directory contains predictions from a different "
                "BiVSS-CD version or configuration; pass --force or choose a "
                "new --output-dir"
            )
    manifest_path.write_text(
        json.dumps({"run_id": run_id, **run_spec}, indent=2), encoding="utf-8"
    )
    model = BiVSSCD(config)
    binary_rows: list[dict[str, object]] = []
    fieldnames = ["name", "iou", "f1", "precision", "recall", "oa"]
    append = not args.force and csv_path.is_file() and csv_path.stat().st_size > 0
    csv_mode = "a" if append else "w"
    with csv_path.open(csv_mode, encoding="utf-8", newline="") as stream:
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

    global_confusion = BinaryConfusion()
    evaluated_names: list[str] = []
    for pair in pairs:
        prediction_path = predictions / pair.name
        if not prediction_path.is_file():
            continue
        prediction = np.asarray(Image.open(prediction_path))
        target = np.asarray(Image.open(pair.label))
        if prediction.ndim == 3:
            prediction = prediction[..., 0]
        if target.ndim == 3:
            target = target[..., 0]
        global_confusion.update(prediction, target)
        evaluated_names.append(pair.name)

    summary: dict[str, object] = {
        "dataset": args.dataset,
        "samples": len(evaluated_names),
        "binary": global_confusion.metrics().to_dict(),
        "confusion": {
            "tp": global_confusion.tp,
            "fp": global_confusion.fp,
            "fn": global_confusion.fn,
            "tn": global_confusion.tn,
        },
        "config": config.to_dict(),
        "prompts": prompts,
        "runtime": runtime,
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
