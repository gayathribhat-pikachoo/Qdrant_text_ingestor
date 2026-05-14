from __future__ import annotations
import ast
import uuid
from pathlib import Path
from typing import Any
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm
from crud import get_tenants

# Setup the collection's vector size
def collection_vector_size(client: QdrantClient, collection_name: str) -> int:
    """Configured vector dimension for the collection (single or first named vector)."""
    info = client.get_collection(collection_name=collection_name)
    vectors = info.config.params.vectors
    if vectors is None:
        raise ValueError(f"Collection {collection_name!r} has no vector config.")
    size = getattr(vectors, "size", None)
    if size is not None:
        return int(size)
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()))
        return int(getattr(first, "size"))
    raise ValueError(f"Cannot read vector size for collection {collection_name!r}")


# Default CSV path relative to repo root (same layout as Qdrant_text_ingestor).
DEFAULT_QDRANT_CSV = Path("Qdrant_text_ingestor/data/qdrant.csv")
# Column that matches Qdrant custom shard key / tenant_id for this export.
DEFAULT_TENANT_CSV_FIELD = "org_id"


def parse_embedding(value: Any) -> list[float] | None:
    if pd.isna(value):
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    s = str(value).strip()
    if not s:
        return None
    try:
        return [float(v) for v in ast.literal_eval(s)]
    except (ValueError, SyntaxError):
        return None


def _clean_payload_value(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def qdrant_point_id(uid: Any, idx: int) -> str | int:
    """Qdrant accepts only unsigned int or UUID; map arbitrary CSV uids to a stable UUID."""
    if uid is None or (isinstance(uid, float) and pd.isna(uid)):
        return idx
    s = str(uid).strip()
    if not s or s.lower() == "nan":
        return idx
    if isinstance(uid, int) and not isinstance(uid, bool) and uid >= 0:
        return uid
    if s.isdigit():
        return int(s)
    try:
        uuid.UUID(s)
        return s
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def load_rows_for_tenant(
    csv_path: Path,
    tenant_name: str,
    tenant_field: str = DEFAULT_TENANT_CSV_FIELD,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path, low_memory=False)
    if tenant_field not in df.columns:
        raise ValueError(f"CSV missing column {tenant_field!r}; columns: {list(df.columns)[:30]}...")
    df = df[df[tenant_field] == tenant_name]
    df = df.copy()
    df["embedding"] = df["embedding"].apply(parse_embedding)
    df = df[df["embedding"].notna()]
    return df.to_dict("records")


def _is_pandas_junk_column(name: Any) -> bool:
    """True for empty Excel/CSV headers pandas labels as ``Unnamed: N``."""
    return isinstance(name, str) and name.startswith("Unnamed:")


def records_to_points(records: list[dict[str, Any]]) -> list[PointStruct]:
    points: list[PointStruct] = []
    for idx, record in enumerate(records):
        emb = record.get("embedding")
        if not emb:
            continue
        point_id = qdrant_point_id(record.get("uid"), idx)
        payload = {
            k: _clean_payload_value(v)
            for k, v in record.items()
            if k != "embedding" and not _is_pandas_junk_column(k)
        }
        points.append(
            PointStruct(
                id=point_id,
                vector=list(emb),
                payload=payload,
            )
        )
    return points

# Ingestion function 
def ingest_csv_tenant(
    client: QdrantClient,
    collection_name: str,
    tenant_name: str,
    csv_path: Path,
    batch_size: int = 300,
    tenant_csv_field: str = DEFAULT_TENANT_CSV_FIELD,
) -> int:
    """
    Load rows from qdrant.csv for ``tenant_name`` (CSV column ``tenant_csv_field``),
    build points, upsert into ``collection_name`` with ``shard_key_selector=tenant_name``.

    Returns number of points upserted.
    """
    shard_tenants = get_tenants(client, collection_name)
    use_shard = bool(shard_tenants)
    if use_shard and tenant_name not in shard_tenants:
        raise ValueError(
            f"Tenant {tenant_name!r} is not a shard key on collection {collection_name!r}. "
            f"Known keys (sample): {shard_tenants[:15]}{'...' if len(shard_tenants) > 15 else ''}"
        )

    rows = load_rows_for_tenant(csv_path, tenant_name, tenant_field=tenant_csv_field)
    points = records_to_points(rows)
    if not points:
        return 0

    expected = collection_vector_size(client, collection_name)
    dim = len(points[0].vector)
    if dim != expected:
        raise ValueError(
            f"Vector dimension mismatch: qdrant.csv has {dim}-dim embeddings but collection "
            f"{collection_name!r} is configured for {expected}. "
            f"Recreate the collection with vector size {dim} (set QDRANT_VECTOR_SIZE={dim} in .env, "
            f"then `cli.py create` / `restore --force-recreate`), or use {expected}-dim embeddings in the CSV."
        )

    for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
        batch = points[i : i + batch_size]
        kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "points": batch,
            "wait": True,
        }
        if use_shard:
            kwargs["shard_key_selector"] = tenant_name
        client.upsert(**kwargs)

    return len(points)
