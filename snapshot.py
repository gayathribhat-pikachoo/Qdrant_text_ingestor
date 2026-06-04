from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import requests
from qdrant_client import QdrantClient
from tqdm import tqdm

from qdrant_config import Settings

_CHUNK = 8192


def create_snapshot(client: QdrantClient, collection_name: str) -> Any:
    """Trigger snapshot creation and return the SnapshotDescription."""
    return client.create_snapshot(collection_name=collection_name)


def list_snapshots(client: QdrantClient, collection_name: str) -> list[Any]:
    """Return all SnapshotDescription objects for a collection."""
    return client.list_snapshots(collection_name=collection_name)


def download_snapshot(
    settings: Settings,
    collection_name: str,
    snapshot_name: str,
    output_path: Path,
) -> Path:
    """Download a snapshot file from the Qdrant REST API to *output_path*."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{settings.url}/collections/{collection_name}/snapshots/{snapshot_name}"
    headers: dict[str, str] = {}
    api_key = getattr(settings, "api_key", None)
    if api_key:
        headers["api-key"] = api_key

    with requests.get(url, headers=headers, stream=True, timeout=settings.timeout_seconds) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        with (
            output_path.open("wb") as f,
            tqdm(
                desc=f"Downloading {snapshot_name}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not sys.stdout.isatty(),
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                f.write(chunk)
                bar.update(len(chunk))

    return output_path


def restore_snapshot(
    settings: Settings,
    snapshot_path: Path,
    target_collection: str,
    *,
    priority: str = "snapshot",
) -> None:
    """Upload a local .snapshot file and restore it into *target_collection*."""
    snapshot_path = snapshot_path.resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    url = (
        f"{settings.url}/collections/{target_collection}"
        f"/snapshots/upload?priority={priority}"
    )
    headers: dict[str, str] = {}
    api_key = getattr(settings, "api_key", None)
    if api_key:
        headers["api-key"] = api_key

    file_size = snapshot_path.stat().st_size
    with (
        snapshot_path.open("rb") as f,
        tqdm(
            desc=f"Uploading {snapshot_path.name}",
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not sys.stdout.isatty(),
        ) as bar,
    ):
        class _ProgressReader:
            def __init__(self, fh: Any, bar: tqdm) -> None:  # type: ignore[type-arg]
                self._fh = fh
                self._bar = bar

            def read(self, size: int = -1) -> bytes:
                data = self._fh.read(size)
                self._bar.update(len(data))
                return data

        resp = requests.post(
            url,
            headers=headers,
            files={"snapshot": (snapshot_path.name, _ProgressReader(f, bar))},
            timeout=settings.timeout_seconds,
        )

    resp.raise_for_status()
