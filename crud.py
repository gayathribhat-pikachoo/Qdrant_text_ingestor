from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient, models

from qdrant_config import Settings

# Show collection and its tenants
def get_collection_tenants(client: QdrantClient, collection: str) -> list[str]:
    response = client.list_shard_keys(collection_name=collection)
    return sorted(
        unwrap_shard_key(entry)
        for entry in (getattr(response, "shard_keys", None) or [])
    )


def get_tenants(client: QdrantClient, collection: str) -> list[str]:
    """Return sorted tenant shard keys for one collection."""
    return get_collection_tenants(client, collection)


def get_collections_with_tenants(client: QdrantClient) -> dict[str, list[str]]:
    return {
        collection.name: get_collection_tenants(client, collection.name)
        for collection in client.get_collections().collections
    }


def save_collections(path: Path, collections: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collections, indent=2, sort_keys=True), encoding="utf-8")


def load_collections(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(collection): [str(tenant) for tenant in tenants] for collection, tenants in data.items()}

# Create collection
def create_collection(client: QdrantClient, collection: str, settings: Settings, custom: bool) -> None:
    kwargs = {
        "collection_name": collection,
        "vectors_config": models.VectorParams(
            size=settings.vector_size,
            distance=models.Distance.COSINE,
        ),
        "replication_factor": settings.replication_factor,
    }
    if custom:
        kwargs["shard_number"] = settings.shards_per_tenant_name
        kwargs["sharding_method"] = models.ShardingMethod.CUSTOM
    client.create_collection(**kwargs)

# Create shard key
def create_shard_key(client: QdrantClient, collection: str, shard_key: str, shards: int) -> bool:
    try:
        client.create_shard_key(
            collection_name=collection,
            shard_key=shard_key,
            shards_number=shards,
        )
        return True
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" in message or "conflict" in message:
            return False
        if "distributed mode disabled" in message or ("cluster" in message and "disabled" in message):
            raise RuntimeError("Qdrant distributed mode is disabled; shard keys cannot be created.") from exc
        raise

# Create tenant payload index
def create_tenant_payload_index(client: QdrantClient, collection: str, field: str) -> None:
    client.create_payload_index(
        collection_name=collection,
        field_name=field,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

# Delete collection
def delete_collection(client: QdrantClient, collection: str) -> bool:
    return client.delete_collection(collection_name=collection)

# Delete tenants
def delete_tenants(client: QdrantClient, collection: str, tenant_names: list[str]) -> tuple[list[str], list[str]]:
    existing = set(get_collection_tenants(client, collection))
    deleted, skipped = [], []
    for tenant_name in tenant_names:
        if tenant_name not in existing:
            skipped.append(tenant_name)
            continue
        if delete_tenant(client, collection, tenant_name):
            deleted.append(tenant_name)
        else:
            skipped.append(tenant_name)
    return deleted, skipped


def delete_tenant(client: QdrantClient, collection: str, tenant_name: str) -> bool:
    return client.delete_shard_key(collection_name=collection, shard_key=tenant_name)


def unwrap_shard_key(entry: object) -> str:
    shard_key = getattr(entry, "shard_key", entry)
    return str(getattr(shard_key, "key", shard_key))
