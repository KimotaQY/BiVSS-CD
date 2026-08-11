import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".cff", ".txt"}
FORBIDDEN = [
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"[A-Za-z]:\\(?:Users|CSU_projects|BaiduSyncdisk)\\", re.IGNORECASE),
    re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
]


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines() if line]


def main() -> int:
    errors: list[str] = []
    for path in tracked_files():
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN:
            if pattern.search(content):
                errors.append(f"{path.relative_to(ROOT)} matches {pattern.pattern}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"{path.relative_to(ROOT)} exceeds 5 MiB")
    if errors:
        print("Release check failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Release check passed ({len(tracked_files())} tracked paths checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
