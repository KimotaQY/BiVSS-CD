from pathlib import Path

import tomllib


def test_sam3_extra_contains_import_time_dependencies():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    dependencies = {
        requirement.split(";", 1)[0].strip().split(">", 1)[0].split("=", 1)[0].lower()
        for requirement in project["optional-dependencies"]["sam3"]
    }
    assert {"einops", "pycocotools", "psutil"} <= dependencies
