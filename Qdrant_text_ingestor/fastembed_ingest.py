from __future__ import annotations

import csv
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import typer
from qdrant_client import QdrantClient, models
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crud import get_tenants
from ingestion import collection_vector_size
from qdrant_config import load_settings, qdrant_client

CSV_PATH = REPO_ROOT / "data" / "qdrant.csv"
TENANT_NAME = "org_id"
TEXT_DATA = "text"
MODEL_NAME = "intfloat/multilingual-e5-large"
DEFAULT_FASTEMBED_CSV = CSV_PATH
DEFAULT_MODEL_NAME = MODEL_NAME

csv.field_size_limit(sys.maxsize)

app = typer.Typer(
    help="Stream CSV rows into Qdrant with qdrant-client FastEmbed inference."
)

# Create UID for one record(point)
def qdrant_point_id(uid: Any, idx: int) -> str | int:
    if uid is None:
        return idx
    value = str(uid).strip()
    if not value or value.lower() == "nan":
        return idx
    if value.isdigit():
        return int(value)
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))

# Helpers to handle edge cases
def is_empty(value: Any) -> bool:
    return value is None or str(value).strip() == ""

# Helpers to handle edge cases
def is_junk_column(name: Any) -> bool:
    return not isinstance(name, str) or not name.strip() or name.startswith("Unnamed:")

# Helpers to handle edge cases
def clean_payload_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None
    if text[:1] in {"[", "{"}:
        parsed = parse_jsonish(text)
        return value if parsed is None else parsed
    return value

# Helper cleaner
def parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    if '""' not in value:
        return None
    try:
        return json.loads(value.replace('""', '"'))
    except json.JSONDecodeError:
        return None

# definition of value (main data)
def row_text(row: dict[str, Any], *, text_field: str = TEXT_DATA) -> str:
    value = row.get(text_field)
    return "" if is_empty(value) else str(value)

# Store "metdata" in payload feild
def row_payload(
    row: dict[str, Any],
    *,
    tenant_name: str,
    tenant_payload_field: str,
    text_field: str = TEXT_DATA,
) -> dict[str, Any]:
    payload = {
        key: clean_payload_value(value)
        for key, value in row.items()
        if key not in {"embedding", text_field} and not is_junk_column(key)
    }
    payload.pop(text_field, None)
    payload.pop("embedding", None)
    payload[tenant_payload_field] = tenant_name
    return payload

# Read records from csv 
def iter_csv_rows(
    csv_path: Path,
    tenant_name: str,
    *,
    tenant_csv_field: str = TENANT_NAME,
) -> Iterable[tuple[int, dict[str, Any]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {csv_path} is empty or has no header.")
        if tenant_csv_field not in reader.fieldnames:
            raise ValueError(
                f"CSV missing tenant column {tenant_csv_field!r}; "
                f"columns: {reader.fieldnames[:30]}..."
            )

        for idx, row in enumerate(reader):
            if str(row.get(tenant_csv_field, "")).strip() == tenant_name:
                yield idx, row

# Convert records to points on the server
def records_to_points(
    records: Iterable[tuple[int, dict[str, Any]]],
    *,
    model_name: str,
    tenant_name: str,
    tenant_payload_field: str,
    text_field: str = TEXT_DATA,
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for idx, row in records:
        text = row_text(row, text_field=text_field)
        if not text.strip():
            continue
        payload = row_payload(
            row,
            tenant_name=tenant_name,
            tenant_payload_field=tenant_payload_field,
            text_field=text_field,
        )
        points.append(
            models.PointStruct(
                id=qdrant_point_id(row.get("uid"), idx),
                vector=models.Document(text=text, model=model_name),
                payload=payload,
            )
        )
    return points

# when we do batching, call this
def batched(items: Iterable[tuple[int, dict[str, Any]]], size: int) -> Iterable[list[tuple[int, dict[str, Any]]]]:
    batch: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

# Upsert()
def upsert_fastembed_csv(
    client: QdrantClient,
    *,
    collection_name: str,
    tenant_name: str,
    csv_path: Path = CSV_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = 100,
    tenant_csv_field: str = TENANT_NAME,
    tenant_payload_field: str = "tenant_name",
    text_field: str = TEXT_DATA,
) -> int:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    model_size = client.get_embedding_size(model_name)
    vector_size = collection_vector_size(client, collection_name)
    if model_size != vector_size:
        raise ValueError(
            f"FastEmbed model {model_name!r} creates {model_size}-dim vectors, "
            f"but collection {collection_name!r} expects {vector_size}."
        )

    shard_tenants = get_tenants(client, collection_name)
    use_shard = bool(shard_tenants)
    if use_shard and tenant_name not in shard_tenants:
        raise ValueError(
            f"Tenant {tenant_name!r} is not a shard key on collection {collection_name!r}."
        )

    total = 0
    rows = iter_csv_rows(csv_path, tenant_name, tenant_csv_field=tenant_csv_field)
    progress = tqdm(desc="FastEmbed upsert", unit="point")
    try:
        for batch_num, row_batch in enumerate(batched(rows, batch_size), start=1):
            points = records_to_points(
                row_batch,
                model_name=model_name,
                tenant_name=tenant_name,
                tenant_payload_field=tenant_payload_field,
                text_field=text_field,
            )
            if not points:
                continue

            progress.set_description(f"FastEmbed upsert (batch {batch_num})")
            tqdm.write(
                f"Embedding + upserting batch {batch_num} ({len(points)} point(s); "
                f"first batch may take several minutes while the ONNX model warms up)..."
            )

            kwargs: dict[str, Any] = {
                "collection_name": collection_name,
                "points": points,
                "wait": True,
            }
            if use_shard:
                kwargs["shard_key_selector"] = tenant_name
            client.upsert(**kwargs)
            total += len(points)
            progress.update(len(points))
    finally:
        progress.close()

    return total


def build_query_filter(
    *,
    tenant_name: str | None,
    tenant_payload_field: str,
    filename: str | None = None,
) -> models.Filter | None:
    must: list[models.FieldCondition] = []
    if tenant_name is not None:
        must.append(
            models.FieldCondition(
                key=tenant_payload_field,
                match=models.MatchValue(value=tenant_name),
            )
        )
    if filename is not None:
        must.append(
            models.FieldCondition(
                key="filename",
                match=models.MatchValue(value=filename),
            )
        )
    return models.Filter(must=must) if must else None


# Semantic Search
def query_fastembed(
    client: QdrantClient,
    *,
    collection_name: str,
    query_text: str,
    model_name: str = MODEL_NAME,
    top_k: int = 5,
    tenant_name: str | None = None,
    tenant_payload_field: str = "tenant_name",
    payload_keys: list[str] | None = None,
    filename: str | None = None,
) -> list[models.ScoredPoint]:
    """Semantic search via models.Document (embeds query text locally, same as upsert)."""
    with_payload: bool | list[str] = payload_keys if payload_keys else True
    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "query": models.Document(text=query_text, model=model_name),
        "limit": top_k,
        "with_payload": with_payload,
    }
    if tenant_name is not None:
        shard_keys = get_tenants(client, collection_name)
        if shard_keys and tenant_name not in shard_keys:
            raise ValueError(
                f"Tenant {tenant_name!r} is not a shard key on collection {collection_name!r}."
            )
        if shard_keys:
            kwargs["shard_key_selector"] = tenant_name
    query_filter = build_query_filter(
        tenant_name=tenant_name,
        tenant_payload_field=tenant_payload_field,
        filename=filename,
    )
    if query_filter is not None:
        kwargs["query_filter"] = query_filter
    return client.query_points(**kwargs).points

# print search results
def print_search_results(
    points: list[models.ScoredPoint],
    *,
    query_text: str,
    payload_keys: list[str] | None = None,
) -> None:
    typer.echo(f'Query: "{query_text}"')
    if not points:
        typer.echo("No results.")
        return
    for rank, hit in enumerate(points, start=1):
        payload = hit.payload or {}
        typer.echo(f"{rank}. id={hit.id} score={hit.score:.4f}")
        keys = payload_keys or sorted(payload.keys())
        for key in keys:
            if key in payload:
                value = payload[key]
                text = str(value)
                if len(text) > 200:
                    text = text[:200] + "..."
                typer.echo(f"    {key}: {text}")


# CLI
@app.command("query")
def query_command(
    collection_name: str = typer.Option(
        ...,
        "--collection-name",
        "--collection_name",
        "-c",
        help="Qdrant collection name.",
    ),
    query_text: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="Natural-language search text.",
    ),
    tenant_name: str | None = typer.Option(
        None,
        "--tenant-name",
        "--tenant_name",
        "-t",
        help="Optional tenant / shard key to scope search.",
    ),
    model_name: str = typer.Option(
        MODEL_NAME,
        "--model",
        "-m",
        help="FastEmbed model name (must match ingest model).",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "--top_k",
        "-k",
        help="Top-k hits to return.",
    ),
    payload_keys: str | None = typer.Option(
        None,
        "--payload-keys",
        help="Comma-separated payload fields to return and print, e.g. uid,filename,doc_id,org_id.",
    ),
    filename: str | None = typer.Option(
        None,
        "--filename",
        "-n",
        help="Exact payload filename to filter by (must match stored value).",
    ),
) -> None:
    """Semantic search with FastEmbed via models.Document (see Qdrant FastEmbed docs)."""
    settings = load_settings()
    client = qdrant_client(settings)
    keys = [k.strip() for k in payload_keys.split(",") if k.strip()] if payload_keys else None
    results = query_fastembed(
        client,
        collection_name=collection_name,
        query_text=query_text,
        model_name=model_name,
        top_k=top_k,
        tenant_name=tenant_name,
        tenant_payload_field=settings.tenant_name_field,
        payload_keys=keys,
        filename=filename,
    )
    print_search_results(results, query_text=query_text, payload_keys=keys)



@app.command("upsert")
def upsert_command(
    collection_name: str = typer.Option(
        ...,
        "--collection-name",
        "--collection_name",
        "-c",
        help="Qdrant collection name.",
    ),
    tenant_name: str = typer.Option(
        ...,
        "--tenant-name",
        "--tenant_name",
        "-t",
        help="Tenant name / shard key. Rows are filtered by org_id by default.",
    ),
    csv_path: Path = typer.Option(
        CSV_PATH,
        "--csv",
        "-f",
        help="Path to qdrant.csv.",
    ),
    model_name: str = typer.Option(
        MODEL_NAME,
        "--model",
        "-m",
        help="FastEmbed model name used by models.Document().",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        help="Rows per Qdrant upsert call.",
    ),
    tenant_csv_field: str = typer.Option(
        TENANT_NAME,
        "--org_id",
        help="CSV column used to filter rows for --tenant-name.",
    ),
    text_field: str = typer.Option(
        TEXT_DATA,
        "--text-field",
        help="CSV column used as the embedding text.",
    ),
) -> None:
    settings = load_settings()
    client = qdrant_client(settings)
    count = upsert_fastembed_csv(
        client,
        collection_name=collection_name,
        tenant_name=tenant_name,
        csv_path=csv_path,
        model_name=model_name,
        batch_size=batch_size,
        tenant_csv_field=tenant_csv_field,
        tenant_payload_field=settings.tenant_name_field,
        text_field=text_field,
    )
    typer.echo(
        f"Upserted {count} point(s) into collection={collection_name!r}, "
        f"tenant={tenant_name!r} from {csv_path}"
    )


if __name__ == "__main__":
    sys.exit(app())
