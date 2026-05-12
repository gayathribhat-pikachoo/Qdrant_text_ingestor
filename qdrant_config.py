from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Data Validation happens  here
@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    """REST / HTTP port used in ``url`` (default 6333)."""
    grpc_port: int
    prefer_grpc: bool
    vector_size: int
    shards_per_tenant_name: int
    replication_factor: int
    tenant_name_field: str
    force_recreate: bool
    # HTTP/gRPC client deadline (seconds). qdrant-client defaults to 5s if unset — too low for large upserts.
    timeout_seconds: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_settings(force_recreate: bool | None = None) -> Settings:
    load_dotenv(find_env(), override=True)
    return Settings(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
        prefer_grpc=env_bool("QDRANT_PREFER_GRPC"),
        vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "3072")),
        shards_per_tenant_name=int(os.getenv("QDRANT_SHARDS_PER_TENANT_NAME", "1")),
        replication_factor=int(os.getenv("QDRANT_REPLICATION_FACTOR", "1")),
        tenant_name_field=os.getenv("QDRANT_TENANT_NAME_FIELD", "tenant_name"),
        force_recreate=env_bool("QDRANT_FORCE_RECREATE_COLLECTIONS")
        if force_recreate is None
        else force_recreate,
        timeout_seconds=int(os.getenv("QDRANT_TIMEOUT", "300")),
    )

# find .env file
def find_env() -> Path | None:
    return next((p / ".env" for p in [Path.cwd(), *Path.cwd().parents] if (p / ".env").exists()), None)

# convert string to boolean
def env_bool(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes"}

# create qdrant client
def qdrant_client(settings: Settings) -> QdrantClient:
    probe_qdrant(settings)
    return QdrantClient(
        url=settings.url,
        prefer_grpc=settings.prefer_grpc,
        grpc_port=settings.grpc_port,
        timeout=settings.timeout_seconds,
    )


# probe qdrant
def probe_qdrant(settings: Settings) -> None:
    """Check TCP reachability: gRPC port when prefer_grpc, else REST port."""
    port = settings.grpc_port if settings.prefer_grpc else settings.port
    try:
        with socket.create_connection((settings.host, port), timeout=5):
            return
    except OSError as exc:
        raise RuntimeError(
            f"Cannot reach Qdrant at {settings.host}:{port} "
            f"({'gRPC' if settings.prefer_grpc else 'REST'}): {exc}"
        ) from exc

