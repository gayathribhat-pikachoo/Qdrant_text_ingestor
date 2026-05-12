from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from crud import (
    delete_collection,
    delete_tenants,
    get_collections_with_tenants,
    get_tenants,
    load_collections,
    save_collections,
)
from helper import ProvisionResult, QdrantProvisioner
from ingestion import DEFAULT_QDRANT_CSV, collection_vector_size, ingest_csv_tenant
from parallel_query import (
    embed_text_queries,
    load_query_vectors,
    prune_unnamed_payload_fields,
    query_batch_dense,
    query_parallel_dense,
    sanitize_for_json,
)
from qdrant_config import Settings, load_settings, qdrant_client

app = typer.Typer(help="Provision Qdrant collections and tenant shard keys.")
console = Console()
DEFAULT_PERSIST_PATH = Path("data/collections.json")


def transport_label(settings: Settings) -> str:
    """Human-readable client transport (qdrant-client)."""
    if settings.prefer_grpc:
        return f"gRPC {settings.host}:{settings.grpc_port}"
    return "REST (HTTP)"


@app.callback()
def main() -> None:
    """Provision Qdrant collections."""


@app.command()
def create(
    collection_name: Annotated[str, typer.Argument(help="Qdrant collection name.")],
    tenant_names: Annotated[
        list[str] | None,
        typer.Option("--tenant-name", "--tenant_name", "-t", help="Tenant name used as shard key."),
    ] = None,
    force_recreate: Annotated[
        bool | None,
        typer.Option("--force-recreate/--no-force-recreate", help="Override env setting."),
    ] = None,
) -> None:
    settings = load_settings(force_recreate=force_recreate)
    tenant_names = tenant_names or []

    with console.status(f"Provisioning [bold]{collection_name}[/bold] on {settings.url}"):
        result = QdrantProvisioner(settings).ensure(collection_name, tenant_names)

    render_summary(settings, result)


@app.command("list")
def list_collections(
    as_dict: Annotated[
        bool,
        typer.Option("--dict", help="Print a Python dictionary instead of a table."),
    ] = False,
) -> None:
    settings = load_settings()
    client = qdrant_client(settings)
    collections = get_collections_with_tenants(client)

    if as_dict:
        console.print(collections)
        return

    render_collections(settings, collections)


@app.command("persist")
def persist_collections(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to save collection tenant mapping."),
    ] = DEFAULT_PERSIST_PATH,
) -> None:
    settings = load_settings()
    client = qdrant_client(settings)
    collections = get_collections_with_tenants(client)
    save_collections(output, collections)
    console.print(f"[green]Saved[/green] {len(collections)} collection(s) to {output}")


@app.command("ingest")
def ingest_command(
    collection_name: Annotated[
        str,
        typer.Option("--collection-name", "--collection_name", "-c", help="Qdrant collection (custom-sharded)."),
    ],
    tenant_name: Annotated[
        str,
        typer.Option("--tenant-name", "--tenant_name", "-t", help="Shard key; must match org_id in qdrant.csv."),
    ],
    csv_path: Annotated[
        Path,
        typer.Option("--csv", "-f", help="Path to qdrant.csv."),
    ] = DEFAULT_QDRANT_CSV,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Points per upsert batch."),
    ] = 400,
) -> None:
    """Ingest precomputed embeddings from qdrant.csv into one tenant shard."""
    settings = load_settings()
    client = qdrant_client(settings)
    if not csv_path.is_file():
        console.print(f"[red]CSV not found:[/red] {csv_path.resolve()}")
        raise typer.Exit(code=1)
    coll_dim = collection_vector_size(client, collection_name)
    console.print(
        f"Qdrant: {transport_label(settings)} (base URL {settings.url}) | vector dim: {coll_dim} | "
        f"client timeout: {settings.timeout_seconds}s (set QDRANT_TIMEOUT if upserts time out)"
    )
    n = ingest_csv_tenant(
        client,
        collection_name,
        tenant_name,
        csv_path,
        batch_size=batch_size,
    )
    console.print(
        f"[green]Ingested[/green] {n} point(s) into collection={collection_name!r} "
        f"shard_key={tenant_name!r} from {csv_path}"
    )


@app.command("parallel-query")
def parallel_query_command(
    collection_name: Annotated[
        str,
        typer.Option("--collection-name", "--collection_name", "-c", help="Qdrant collection to search."),
    ],
    query: Annotated[
        list[str] | None,
        typer.Option(
            "--query",
            "-Q",
            help="Natural-language question (repeatable). Embedded via OpenAI; cannot combine with --queries-file.",
        ),
    ] = None,
    queries_file: Annotated[
        Path | None,
        typer.Option(
            "--queries-file",
            "-q",
            help="JSON file of dense vectors (array of arrays or {\"vectors\": [...]}). Omit if using -Q.",
        ),
    ] = None,
    embedding_model: Annotated[
        str,
        typer.Option("--embedding-model", help="OpenAI embedding model when using -Q."),
    ] = "text-embedding-3-large",
    embedding_dimensions: Annotated[
        int,
        typer.Option(
            "--embedding-dimensions",
            help="OpenAI ``dimensions`` for -Q queries (default 3072 for text-embedding-3-large). "
            "Override if your collection uses reduced dimensions (e.g. 1536).",
        ),
    ] = 3072,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Max hits per query."),
    ] = 10,
    tenant_name: Annotated[
        str | None,
        typer.Option(
            "--tenant-name",
            "--tenant_name",
            "-t",
            help="Shard key for custom-sharded collections (optional for non-sharded search).",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Thread pool size when using parallel mode."),
    ] = 8,
    batch_api: Annotated[
        bool,
        typer.Option(
            "--batch-api/--parallel-threads",
            help="Use one query_batch_points RPC (default). Use --parallel-threads for concurrent query_points.",
        ),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON results to this path."),
    ] = None,
    with_vectors: Annotated[
        bool,
        typer.Option("--with-vectors/--no-with-vectors", help="Include vectors in hits."),
    ] = False,
    payload_keys: Annotated[
        str | None,
        typer.Option(
            "--payload-keys",
            help="Comma-separated payload field names to fetch (smaller responses). "
            "Example: text,uid,org_id. Default: all stored fields.",
        ),
    ] = None,
    full_payload: Annotated[
        bool,
        typer.Option(
            "--full-payload",
            help="Keep pandas 'Unnamed:*' keys in JSON if they exist in stored points (default: omit from output).",
        ),
    ] = False,
) -> None:
    """Run vector search from a JSON file and/or natural-language questions (embedded with OpenAI)."""
    settings = load_settings()
    client = qdrant_client(settings)

    shard_keys = get_tenants(client, collection_name)
    use_shard = bool(shard_keys)
    if tenant_name is not None:
        if not use_shard:
            raise typer.BadParameter(
                f"Collection {collection_name!r} has no shard keys; omit -t / --tenant-name."
            )
        if tenant_name not in shard_keys:
            sample = shard_keys[:15]
            suffix = "..." if len(shard_keys) > 15 else ""
            raise typer.BadParameter(
                f"Shard key {tenant_name!r} is not registered on collection {collection_name!r}. "
                f"Known keys: {', '.join(sample)}{suffix}. "
                f"Provision with: uv run python cli.py create {collection_name} -t {tenant_name!r}"
            )

    text_queries = [t.strip() for t in (query or []) if t and t.strip()]
    if text_queries and queries_file is not None:
        raise typer.BadParameter("Use either --query / -Q or --queries-file, not both.")
    if not text_queries and queries_file is None:
        raise typer.BadParameter("Provide --query / -Q (repeatable) or --queries-file.")

    if text_queries:
        with console.status(f"Embedding {len(text_queries)} question(s) with {embedding_model!r}…"):
            vectors = embed_text_queries(
                text_queries,
                model=embedding_model,
                dimensions=embedding_dimensions,
            )
        query_labels: list[str | None] = text_queries
    else:
        assert queries_file is not None
        if not queries_file.is_file():
            console.print(f"[red]File not found:[/red] {queries_file.resolve()}")
            raise typer.Exit(code=1)
        vectors = load_query_vectors(queries_file)
        query_labels = [None] * len(vectors)

    if not vectors:
        console.print("[yellow]No queries to run; nothing to do.[/yellow]")
        raise typer.Exit(code=1)

    coll_dim = collection_vector_size(client, collection_name)
    for i, vec in enumerate(vectors):
        if len(vec) != coll_dim:
            hint = (
                " Adjust --embedding-dimensions so query vectors match the collection (default 3072)."
                if text_queries
                else ""
            )
            raise typer.BadParameter(
                f"Query {i} has dimension {len(vec)}, collection expects {coll_dim}.{hint}"
            )

    mode = "batch RPC (query_batch_points)" if batch_api else f"parallel threads (workers={workers})"
    console.print(
        f"Qdrant: {transport_label(settings)} | {len(vectors)} queries | {mode} | limit={limit}"
    )

    with_payload: bool | list[str] = True
    if payload_keys is not None:
        keys = [k.strip() for k in payload_keys.split(",") if k.strip()]
        if keys:
            with_payload = keys

    if batch_api:
        results = query_batch_dense(
            client,
            collection_name,
            vectors,
            limit=limit,
            shard_key_selector=tenant_name,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
    else:
        results = query_parallel_dense(
            client,
            collection_name,
            vectors,
            limit=limit,
            shard_key_selector=tenant_name,
            max_workers=workers,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    if not full_payload:
        prune_unnamed_payload_fields(results)

    for row, label in zip(results, query_labels, strict=False):
        if label is not None:
            row["query_text"] = label

    text = json.dumps(sanitize_for_json(results), indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {len(results)} result(s) to {output}")
    else:
        console.print(text)


@app.command("restore")
def restore_collections(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to saved collection tenant mapping."),
    ] = DEFAULT_PERSIST_PATH,
    force_recreate: Annotated[
        bool | None,
        typer.Option("--force-recreate/--no-force-recreate", help="Override env setting."),
    ] = None,
) -> None:
    settings = load_settings(force_recreate=force_recreate)
    provisioner = QdrantProvisioner(settings)
    collections = load_collections(input_path)

    results = [
        provisioner.ensure(collection, tenants)
        for collection, tenants in collections.items()
    ]
    console.print(f"[green]Restored[/green] {len(results)} collection(s) from {input_path}")


@app.command("delete-collection")
def remove_collection(
    collection_name: Annotated[str, typer.Argument(help="Qdrant collection name.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    settings = load_settings()
    client = qdrant_client(settings)

    if not yes:
        typer.confirm(f"Delete collection '{collection_name}'?", abort=True)

    deleted = delete_collection(client, collection_name)
    console.print(
        f"[green]Deleted[/green] {collection_name}"
        if deleted
        else f"[yellow]Not deleted[/yellow] {collection_name}"
    )


@app.command("delete-tenants")
def remove_tenants(
    collection_name: Annotated[str, typer.Argument(help="Qdrant collection name.")],
    tenant_names: Annotated[
        list[str],
        typer.Option("--tenant-name", "--tenant_name", "-t", help="Tenant name/shard key to delete."),
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    settings = load_settings()
    client = qdrant_client(settings)

    if not yes:
        typer.confirm(
            f"Delete {len(tenant_names)} tenant shard key(s) from '{collection_name}'?",
            abort=True,
        )

    deleted, skipped = delete_tenants(client, collection_name, tenant_names)
    render_delete_tenants(collection_name, deleted, skipped)


def render_summary(settings: Settings, result: ProvisionResult) -> None:
    console.print(f"[green]Done[/green] {result.collection}")
    console.print(
        f"Qdrant: {settings.url} | transport: {transport_label(settings)} | "
        f"vector: {settings.vector_size} | shards/tenant: {settings.shards_per_tenant_name}"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Created collection", yes_no(result.collection_created))
    #table.add_row("Custom sharding", yes_no(result.custom_sharding))
    table.add_row("Tenant index created", yes_no(result.tenant_name_index_created))
    table.add_row("Shard keys created", ", ".join(result.created_shard_keys) or "-")
    table.add_row("Shard keys skipped", ", ".join(result.skipped_shard_keys) or "-")
    console.print(table)


def render_delete_tenants(collection: str, deleted: list[str], skipped: list[str]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Collection", collection)
    table.add_row("Deleted tenants", ", ".join(deleted) or "-")
    table.add_row("Skipped tenants", ", ".join(skipped) or "-")
    console.print(table)


def render_collections(settings: Settings, collections: dict[str, list[str]]) -> None:
    console.print(f"Qdrant: {settings.url} | transport: {transport_label(settings)}")
    console.print(f"Collections: {len(collections)}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Collection")
    table.add_column("Tenants")
    for collection, tenants in collections.items():
        table.add_row(collection, ", ".join(tenants) or "-")
    console.print(table)


def yes_no(value: bool) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"


if __name__ == "__main__":
    app()
