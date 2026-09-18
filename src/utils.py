"""Utilities shared by the Asirra training and inference application."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

LABELS = ("cat", "dog")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def db_connect():
    """Return the optional SQLAlchemy engine configured by ``DATABASE_URL``.

    The classifier does not require a database; keeping this helper lazy means
    training works in a clean environment without database credentials.
    """
    from sqlalchemy import create_engine

    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return create_engine(url)


def label_from_filename(path: str | Path) -> str:
    """Extract ``cat`` or ``dog`` from an Asirra filename."""
    match = re.search(r"(?:^|[^a-z])(cat|dog)(?:[^a-z]|$)", Path(path).stem.lower())
    if not match:
        raise ValueError(f"Could not infer a cat/dog label from {path}")
    return match.group(1)


def image_files(root: str | Path) -> list[Path]:
    """Return supported image files recursively, in deterministic order."""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def ensure_directory_dataset(files: Iterable[Path], output: str | Path) -> Path:
    """Create the directory layout required by ``flow_from_directory``.

    Files are copied (not moved), preserving the raw dataset. Duplicate names
    are made unique so datasets from multiple folders can be combined safely.
    """
    import shutil

    output = Path(output)
    for source in files:
        label = label_from_filename(source)
        target_dir = output / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        index = 1
        while target.exists() and target.resolve() != source.resolve():
            target = target_dir / f"{source.stem}_{index}{source.suffix}"
            index += 1
        if target.resolve() != source.resolve():
            shutil.copy2(source, target)
    return output
