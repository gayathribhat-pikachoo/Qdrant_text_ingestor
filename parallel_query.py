from __future__ import annotations
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest

# Embed query
def embed_text_queries(
    texts: list[str],
    *,
    model: str,
    dimensions: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot embed --query text.")

    client = OpenAI(api_key=api_key)
    kwargs: dict[str, Any] = {"model": model, "input": texts}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def load_query_vectors(path: Path) -> list[list[float]]:
    """Load dense query vectors from JSON.

    Supported formats:
    - Single JSON array of arrays: ``[[0.1, ...], [0.2, ...]]``
    - JSON object with key ``vectors``: ``{"vectors": [[...], [...]]}``
    - NDJSON (fallback if whole file is not valid JSON): one JSON array per line
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    def _as_vector(row: Any) -> list[float]:
        if isinstance(row, dict) and "vector" in row:
            row = row["vector"]
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"Each query must be a list of floats, got {type(row)}")
        return [float(x) for x in row]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        out: list[list[float]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(_as_vector(json.loads(line)))
        return out

    if isinstance(data, dict) and "vectors" in data:
        data = data["vectors"]
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array of vectors or an object with 'vectors'")
    return [_as_vector(row) for row in data]


def sanitize_for_json(value: Any) -> Any:
    """Return a structure safe for ``json.dumps`` (replace NaN/Inf with ``None``)."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except (ValueError, AttributeError, TypeError):
            pass
    return value


def prune_unnamed_payload_fields(results: list[dict[str, Any]]) -> None:
    """Drop pandas ``Unnamed:*`` keys from each hit payload (in-place)."""
    for block in results:
        for pt in block.get("points") or []:
            pl = pt.get("payload")
            if not isinstance(pl, dict):
                continue
            pt["payload"] = {
                k: v
                for k, v in pl.items()
                if not (isinstance(k, str) and k.startswith("Unnamed:"))
            }


def _scored_points_to_json(resp: Any) -> list[dict[str, Any]]:
    points = getattr(resp, "points", None) or []
    out: list[dict[str, Any]] = []
    for p in points:
        if hasattr(p, "model_dump"):
            out.append(p.model_dump(mode="json"))
        else:
            out.append(
                {
                    "id": getattr(p, "id", None),
                    "score": getattr(p, "score", None),
                    "payload": getattr(p, "payload", None),
                    "version": getattr(p, "version", None),
                }
            )
    return [sanitize_for_json(row) for row in out]


def query_batch_dense(
    client: QdrantClient,
    collection_name: str,
    vectors: list[list[float]],
    *,
    limit: int = 10,
    shard_key_selector: str | None = None,
    with_payload: bool | list[str] = True,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    """Run multiple dense vector queries in one ``query_batch_points`` RPC (lowest latency)."""
    requests = [
        QueryRequest(
            query=list(vec),
            limit=limit,
            shard_key=shard_key_selector,
            with_payload=with_payload,
            with_vector=with_vectors,
        )
        for vec in vectors
    ]
    responses = client.query_batch_points(
        collection_name=collection_name,
        requests=requests,
    )
    return [
        {"query_index": i, "points": _scored_points_to_json(resp)}
        for i, resp in enumerate(responses)
    ]


def query_parallel_dense(
    client: QdrantClient,
    collection_name: str,
    vectors: list[list[float]],
    *,
    limit: int = 10,
    shard_key_selector: str | None = None,
    max_workers: int | None = None,
    with_payload: bool | list[str] = True,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    """Run one ``query_points`` per vector concurrently (good when batch RPC is not desired)."""
    n = len(vectors)
    if n == 0:
        return []
    workers = max_workers if max_workers is not None else min(32, max(4, n))

    def _one(idx: int, vec: list[float]) -> tuple[int, Any]:
        resp = client.query_points(
            collection_name=collection_name,
            query=vec,
            limit=limit,
            shard_key_selector=shard_key_selector,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return idx, resp

    ordered: list[Any | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, i, vectors[i]): i for i in range(n)}
        for fut in as_completed(futures):
            idx, resp = fut.result()
            ordered[idx] = resp

    return [
        {"query_index": i, "points": _scored_points_to_json(ordered[i])}
        for i in range(n)
    ]
