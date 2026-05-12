from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qdrant_client import QdrantClient

from crud import (
    create_collection,
    create_shard_key,
    create_tenant_payload_index,
    delete_collection,
    get_collection_tenants,
)
from qdrant_config import Settings, qdrant_client


@dataclass(frozen=True)
class ProvisionResult:
    collection: str
    tenant_names: list[str]
    custom_sharding: bool
    collection_created: bool
    created_shard_keys: list[str]
    skipped_shard_keys: list[str]
    tenant_name_index_created: bool


class QdrantProvisioner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = qdrant_client(settings)

    def ensure(self, collection: str, tenant_names: Iterable[str]) -> ProvisionResult:
        tenant_names = dedupe(tenant_names)
        exists = self._ensure_collection(collection, bool(tenant_names))
        index_created = self._ensure_tenant_name_index(collection)
        created, skipped = self._ensure_shard_keys(collection, tenant_names)

        return ProvisionResult(
            collection=collection,
            tenant_names=tenant_names,
            custom_sharding=bool(tenant_names),
            collection_created=not exists,
            created_shard_keys=created,
            skipped_shard_keys=skipped,
            tenant_name_index_created=index_created,
        )

    def _ensure_collection(self, collection: str, custom: bool) -> bool:
        exists = already_collection_exists(self.client, collection)
        if exists and custom and not collection_has_custom_sharding(self.client, collection):
            if not self.settings.force_recreate:
                raise RuntimeError(
                    f"Collection '{collection}' exists without custom sharding. "
                    "Use --force-recreate to drop and create it again."
                )
            delete_collection(self.client, collection)
            exists = False

        if not exists:
            create_collection(self.client, collection, self.settings, custom)
        return exists

    def _ensure_tenant_name_index(self, collection: str) -> bool:
        if already_tenant_name_payload_index_exists(self.client, collection, self.settings.tenant_name_field):
            return False
        create_tenant_payload_index(self.client, collection, self.settings.tenant_name_field)
        return True

    def _ensure_shard_keys(self, collection: str, tenant_names: list[str]) -> tuple[list[str], list[str]]:
        if not tenant_names:
            return [], []

        created, skipped = [], []
        for tenant_name in tenant_names:
            if already_shard_key_exists(self.client, collection, tenant_name):
                skipped.append(tenant_name)
                continue
            if create_shard_key(self.client, collection, tenant_name, self.settings.shards_per_tenant_name):
                created.append(tenant_name)
            else:
                skipped.append(tenant_name)
        return created, skipped


def already_collection_exists(client: QdrantClient, collection: str) -> bool:
    return client.collection_exists(collection_name=collection)


def collection_has_custom_sharding(client: QdrantClient, collection: str) -> bool:
    info = client.get_collection(collection_name=collection)
    params = getattr(getattr(info, "config", None), "params", None)
    return "custom" in str(getattr(params, "sharding_method", "")).lower()


def already_tenant_name_payload_index_exists(client: QdrantClient, collection: str, field: str) -> bool:
    info = client.get_collection(collection_name=collection)
    return field in (info.payload_schema or {})


def already_shard_key_exists(client: QdrantClient, collection: str, shard_key: str) -> bool:
    return shard_key in existing_shard_keys(client, collection)


def existing_shard_keys(client: QdrantClient, collection: str) -> set[str]:
    return set(get_collection_tenants(client, collection))


def dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
