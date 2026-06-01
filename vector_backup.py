from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from crud import get_tenants
from ingestion import collection_vector_size
from qdrant_config import Settings

BACKUP_VERSION = 1


@dataclass
class BackupMeta:
    version: int
    collection: str
    vector_size: int
    uses_custom_sharding: bool
    tenant_payload_field: str
    shard_keys_exported: list[str]
    points_total: int
    points_per_shard: dict[str, int]


def _normalize_vector(vec: Any) -> Any:
    """Serialize scroll vector(s) into JSON-safe lists (dense or named vectors)."""
    if vec is None:
        return None
    if isinstance(vec, dict):
        out: dict[str, Any] = {}
        for name, v in vec.items():
            out[str(name)] = _normalize_vector(v)
        return out
    if hasattr(vec, "tolist"):
        return [float(x) for x in vec.tolist()]
    return [float(x) for x in vec]


def _record_to_row(
    record: Any,
    *,
    shard_key: str | None,
) -> dict[str, Any]:
    payload = record.payload or {}
    row: dict[str, Any] = {
        "id": record.id,
        "payload": payload,
        "vector": _normalize_vector(record.vector),
    }
    if shard_key is not None:
        row["shard_key"] = shard_key
    return row


def export_points(
    client: QdrantClient,
    settings: Settings,
    *,
    collection_name: str,
    output_path: Path,
    tenant_name: str | None = None,
    scroll_batch: int = 256,
) -> BackupMeta:
    """
    Stream all points with vectors + payload to a JSONL file (one JSON object per line).
    For custom-sharded collections, scrolls each shard (or only ``tenant_name`` if set).
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dim = collection_vector_size(client, collection_name)
    shard_keys = get_tenants(client, collection_name)
    use_shard = bool(shard_keys)

    if tenant_name is not None:
        if use_shard and tenant_name not in shard_keys:
            raise ValueError(
                f"Shard key {tenant_name!r} not registered on {collection_name!r}. "
                f"Known: {shard_keys[:20]}"
            )
        keys_to_scan: list[str | None] = [tenant_name] if use_shard else [None]
    elif use_shard:
        keys_to_scan = list(shard_keys)
    else:
        keys_to_scan = [None]

    points_total = 0
    points_per_shard: dict[str, int] = {}

    meta_path = output_path.with_suffix(".meta.json")

    with output_path.open("w", encoding="utf-8") as out:
        for sk in keys_to_scan:
            label = sk if sk is not None else "__default__"
            count_shard = 0
            offset = None
            shard_iter = tqdm(
                desc=f"Export {collection_name} [{label}]",
                unit="pt",
                disable=not sys.stdout.isatty(),
            )
            while True:
                kwargs: dict[str, Any] = {
                    "collection_name": collection_name,
                    "limit": scroll_batch,
                    "offset": offset,
                    "with_payload": True,
                    "with_vectors": True,
                }
                if sk is not None:
                    kwargs["shard_key_selector"] = sk

                batch, next_offset = client.scroll(**kwargs)
                if not batch:
                    break
                for rec in batch:
                    row = _record_to_row(rec, shard_key=sk if use_shard else None)
                    out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    count_shard += 1
                    points_total += 1
                    shard_iter.update(1)
                offset = next_offset
                if next_offset is None:
                    break
            shard_iter.close()
            points_per_shard[label] = count_shard

    meta = BackupMeta(
        version=BACKUP_VERSION,
        collection=collection_name,
        vector_size=dim,
        uses_custom_sharding=use_shard,
        tenant_payload_field=settings.tenant_name_field,
        shard_keys_exported=[str(k) for k in keys_to_scan if k is not None],
        points_total=points_total,
        points_per_shard=points_per_shard,
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return meta


def _load_meta(path: Path) -> BackupMeta | None:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return BackupMeta(
        version=int(data["version"]),
        collection=str(data["collection"]),
        vector_size=int(data["vector_size"]),
        uses_custom_sharding=bool(data["uses_custom_sharding"]),
        tenant_payload_field=str(data["tenant_payload_field"]),
        shard_keys_exported=list(data.get("shard_keys_exported", [])),
        points_total=int(data["points_total"]),
        points_per_shard=dict(data.get("points_per_shard", {})),
    )


def import_points(
    client: QdrantClient,
    settings: Settings,
    *,
    input_path: Path,
    collection_name: str | None = None,
    batch_size: int = 200,
    dry_run: bool = False,
) -> int:
    """
    Read JSONL from ``export_points`` and upsert into the collection connected by
    ``client`` (typically a different host after you change ``.env``).

    ``collection_name`` overrides the backup metadata when migrating to a new name.
    """
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Backup file not found: {input_path}")

    meta = _load_meta(input_path)
    target = collection_name or (meta.collection if meta else None)
    if not target:
        raise ValueError("Pass --collection-name or use a backup exported with .meta.json sidecar.")

    expect_dim: int | None = meta.vector_size if meta else None
    if meta:
        coll_dim = collection_vector_size(client, target)
        if coll_dim != meta.vector_size:
            raise ValueError(
                f"Vector dimension mismatch: backup is {meta.vector_size} but collection "
                f"{target!r} expects {coll_dim}. Recreate the collection with matching "
                f"QDRANT_VECTOR_SIZE or use a new empty collection."
            )
    else:
        # Sidecar missing: still require target collection to exist and match first point
        expect_dim = collection_vector_size(client, target)

    shard_keys = get_tenants(client, target)
    use_shard = bool(shard_keys)

    total = 0
    batch: list[models.PointStruct] = []
    line_no = 0
    current_sk: str | None = None

    def flush(pts: list[models.PointStruct], shard_sel: str | None) -> None:
        nonlocal total
        n = len(pts)
        if n == 0:
            return
        if dry_run:
            total += n
            return
        kwargs: dict[str, Any] = {
            "collection_name": target,
            "points": pts,
            "wait": True,
        }
        if shard_sel is not None:
            kwargs["shard_key_selector"] = shard_sel
        client.upsert(**kwargs)
        total += n

    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Import → {target}", unit="line", disable=not sys.stdout.isatty()):
            line_no += 1
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row["id"]
            payload = row.get("payload") or {}
            vector = row["vector"]
            sk: str | None = row.get("shard_key")
            if use_shard:
                if sk is None:
                    sk = payload.get(settings.tenant_name_field)
                if sk is None or not isinstance(sk, str):
                    raise ValueError(
                        f"Line {line_no}: custom sharding requires shard_key in backup line "
                        f"or string {settings.tenant_name_field!r} in payload."
                    )
                if sk not in shard_keys:
                    raise ValueError(
                        f"Line {line_no}: shard key {sk!r} not registered on {target!r}. "
                        f"Run: uv run python cli.py create {target} -t {sk!r}"
                    )

            if isinstance(vector, list) and expect_dim is not None and len(vector) != expect_dim:
                raise ValueError(
                    f"Line {line_no}: vector dim {len(vector)} != expected {expect_dim}"
                )

            pt = models.PointStruct(id=pid, vector=vector, payload=payload)

            if use_shard:
                if batch and sk != current_sk:
                    flush(batch, current_sk)
                    batch = []
                current_sk = sk
                batch.append(pt)
                if len(batch) >= batch_size:
                    flush(batch, current_sk)
                    batch = []
            else:
                batch.append(pt)
                if len(batch) >= batch_size:
                    flush(batch, None)
                    batch = []

    if batch:
        flush(batch, current_sk if use_shard else None)

    return total

