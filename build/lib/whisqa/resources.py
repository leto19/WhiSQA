from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator


@contextmanager
def package_file(*parts: str) -> Iterator[Path]:
    resource = files("whisqa").joinpath(*parts)
    with as_file(resource) as resolved_path:
        yield resolved_path
