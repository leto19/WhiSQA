from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from whisqa.cli import main

__all__ = ["get_score", "main"]


def get_score(*args, **kwargs):
    from whisqa.api import get_score as package_get_score

    return package_get_score(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
