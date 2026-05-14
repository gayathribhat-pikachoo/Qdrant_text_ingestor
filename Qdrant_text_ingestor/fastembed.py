from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient

from helper import QdrantProvisioner
from ingestion import DEFAULT_QDRANT_CSV, records_to_points
from parallel_query import query_batch_dense
from qdrant_config import Settings, load_settings, qdrant_client

from Qdrant_text_ingestor.helpers.embeddings import (
    FASTEMBED_MODEL_NAME,
    get_batch_fastembed_embedding,
    get_fastembed_embedding,
)
from Qdrant_text_ingestor.helpers.utils import (
    batch_list,
    chunk_records_for_embedding,
    load_csv,
)


FASTEMBED_VECTOR_SIZE = 1024


def fastembed_settings(
    vector_size: int = FASTEMBED_VECTOR_SIZE,
    force_recreate: bool | None = None,
) -> Settings:
    """Load normal Qdrant settings but override vector size for FastEmbed collections."""
    return replace(
        load_settings(force_recreate=force_recreate),
        vector_size=vector_size,
    )


def create_fastembed_collection(
    collection_name: str,
    tenant_names: Iterable[str],
    *,
    vector_size: int = FASTEMBED_VECTOR_SIZE,
    force_recreate: bool | None = None,
):
    """Create a FastEmbed-compatible collection and tenant shard keys."""
    settings = fastembed_settings(
        vector_size=vector_size,
        force_recreate=force_recreate,
    )
    return QdrantProvisioner(settings).ensure(collection_name, tenant_names)


def embed_csv_records(
    csv_path: Path = DEFAULT_QDRANT_CSV,
    *,
    tenant_name: str | None = None,
    tenant_field: str = "org_id",
    text_field: str = "text",
    model: str = FASTEMBED_MODEL_NAME,
    embed_batch_size: int = 256,
    window_size: int = 400,
    overlap: int = 80,
    min_tokens: int = 100,
) -> list[dict[str, Any]]:
    """Load CSV rows, markdown-chunk them, and attach FastEmbed vectors."""
    records = load_csv(csv_path)
    if tenant_name is not None:
        records = [record for record in records if record.get(tenant_field) == tenant_name]

    chunked_records = chunk_records_for_embedding(
        records,
        text_field=text_field,
        window_size=window_size,
        overlap=overlap,
        min_tokens=min_tokens,
    )

    embedded_records: list[dict[str, Any]] = []
    for record_batch in batch_list(chunked_records, embed_batch_size):
        texts = [record[text_field] for record in record_batch]
        embeddings = get_batch_fastembed_embedding(
            texts,
            model=model,
            batch_size=embed_batch_size,
            input_type="passage",
        )
        for record, embedding in zip(record_batch, embeddings, strict=True):
            item = dict(record)
            item["embedding"] = embedding
            embedded_records.append(item)

    return embedded_records


def upsert_fastembed_csv(
    collection_name: str,
    tenant_name: str,
    csv_path: Path = DEFAULT_QDRANT_CSV,
    *,
    client: QdrantClient | None = None,
    tenant_field: str = "org_id",
    text_field: str = "text",
    model: str = FASTEMBED_MODEL_NAME,
    embed_batch_size: int = 256,
    upsert_batch_size: int = 300,
    window_size: int = 400,
    overlap: int = 80,
    min_tokens: int = 100,
    progress: bool = True,
) -> int:
    """Embed CSV rows and upsert each embedded batch into one tenant shard."""
    settings = fastembed_settings()
    client = client or qdrant_client(settings)

    records = load_csv(csv_path)
    records = [record for record in records if record.get(tenant_field) == tenant_name]
    if progress:
        print(f"Loaded {len(records)} row(s) for tenant {tenant_name!r}", flush=True)

    chunked_records = chunk_records_for_embedding(
        records,
        text_field=text_field,
        window_size=window_size,
        overlap=overlap,
        min_tokens=min_tokens,
    )
    if progress:
        print(f"Created {len(chunked_records)} chunk(s) for embedding", flush=True)

    total_upserted = 0
    embed_batches = batch_list(chunked_records, embed_batch_size)
    for batch_num, record_batch in enumerate(embed_batches, start=1):
        if progress:
            print(
                f"Embedding batch {batch_num}/{len(embed_batches)} "
                f"({len(record_batch)} chunk(s))",
                flush=True,
            )
        texts = [record[text_field] for record in record_batch]
        embeddings = get_batch_fastembed_embedding(
            texts,
            model=model,
            batch_size=embed_batch_size,
            input_type="passage",
        )

        embedded_records: list[dict[str, Any]] = []
        for record, embedding in zip(record_batch, embeddings, strict=True):
            item = dict(record)
            item["embedding"] = embedding
            embedded_records.append(item)

        points = records_to_points(embedded_records)
        for point_batch in batch_list(points, upsert_batch_size):
            client.upsert(
                collection_name=collection_name,
                points=point_batch,
                shard_key_selector=tenant_name,
                wait=True,
            )
            total_upserted += len(point_batch)

        if progress:
            print(f"Upserted {total_upserted} point(s)", flush=True)

    return total_upserted


def query_fastembed(
    collection_name: str,
    query: str,
    *,
    tenant_name: str | None = None,
    limit: int = 10,
    model: str = FASTEMBED_MODEL_NAME,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Embed a query with FastEmbed and search Qdrant."""
    settings = fastembed_settings()
    client = client or qdrant_client(settings)
    vector = get_fastembed_embedding(query, model=model, input_type="query")
    return query_batch_dense(
        client,
        collection_name,
        [vector],
        limit=limit,
        shard_key_selector=tenant_name,
        with_payload=True,
        with_vectors=False,
    )


# Example usage from repo root:
# create_fastembed_collection("diy-pcw-prod-fastembed", ["PCW"])
# upsert_fastembed_csv("diy-pcw-prod-fastembed", "PCW")
# results = query_fastembed("diy-pcw-prod-fastembed", "solar livelihoods", tenant_name="PCW")
