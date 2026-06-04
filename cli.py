from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from Qdrant_text_ingestor.helpers.embeddings import (
    FASTEMBED_MODEL_NAME,
    get_batch_fastembed_embedding,
)
from Qdrant_text_ingestor.fastembed_ingest import (
    DEFAULT_FASTEMBED_CSV,
    DEFAULT_MODEL_NAME as DEFAULT_FASTEMBED_UPSERT_MODEL,
    upsert_fastembed_csv,
)
from Qdrant_text_ingestor.helpers.utils import (
    batch_list,
    chunk_records_for_embedding,
    load_csv,
)
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
from snapshot import create_snapshot, download_snapshot, list_snapshots, restore_snapshot
from vector_backup import export_points, import_points

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


@app.command("fastembed-upsert")
def fastembed_upsert_command(
    collection_name: Annotated[
        str,
        typer.Option("--collection-name", "--collection_name", "-c", help="Qdrant collection name."),
    ],
    tenant_name: Annotated[
        str,
        typer.Option("--tenant-name", "--tenant_name", "-t", help="Tenant name / shard key."),
    ],
    csv_path: Annotated[
        Path,
        typer.Option("--csv", "-f", help="Path to qdrant.csv."),
    ] = DEFAULT_FASTEMBED_CSV,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="FastEmbed model name used by models.Document()."),
    ] = DEFAULT_FASTEMBED_UPSERT_MODEL,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Rows per Qdrant upsert call."),
    ] = 64,
    tenant_csv_field: Annotated[
        str,
        typer.Option("--tenant-csv-field", help="CSV column used to filter rows for --tenant-name."),
    ] = "org_id",
    text_field: Annotated[
        str,
        typer.Option("--text-field", help="CSV column used as the embedding text."),
    ] = "text",
) -> None:
    """Upsert CSV rows with qdrant-client FastEmbed inference via models.Document()."""
    settings = load_settings()
    client = qdrant_client(settings)
    console.print(
        f"Qdrant: {transport_label(settings)} (base URL {settings.url}) | "
        f"FastEmbed model: {model!r}"
    )
    count = upsert_fastembed_csv(
        client,
        collection_name=collection_name,
        tenant_name=tenant_name,
        csv_path=csv_path,
        model_name=model,
        batch_size=batch_size,
        tenant_csv_field=tenant_csv_field,
        tenant_payload_field=settings.tenant_name_field,
        text_field=text_field,
    )
    console.print(
        f"[green]Upserted[/green] {count} point(s) into collection={collection_name!r} "
        f"shard_key={tenant_name!r} from {csv_path}"
    )


@app.command("embed-fastembed")
def embed_fastembed_command(
    csv_path: Annotated[
        Path,
        typer.Option("--csv", "-f", help="Path to CSV data to chunk and embed."),
    ] = DEFAULT_QDRANT_CSV,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the embedded CSV. Defaults to updating --csv in place."),
    ] = None,
    tenant_name: Annotated[
        str | None,
        typer.Option("--tenant-name", "--tenant_name", "-t", help="Only embed rows for this tenant/org."),
    ] = None,
    tenant_field: Annotated[
        str,
        typer.Option("--tenant-field", help="CSV column used with --tenant-name."),
    ] = "org_id",
    text_field: Annotated[
        str,
        typer.Option("--text-field", help="CSV text column to chunk and embed."),
    ] = "text",
    model: Annotated[
        str,
        typer.Option("--model", help="FastEmbed model name."),
    ] = FASTEMBED_MODEL_NAME,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Text chunks per FastEmbed batch."),
    ] = 256,
    window_size: Annotated[
        int,
        typer.Option("--window-size", help="Sliding-window token size for markdown chunks."),
    ] = 400,
    overlap: Annotated[
        int,
        typer.Option("--overlap", help="Sliding-window token overlap."),
    ] = 80,
    min_tokens: Annotated[
        int,
        typer.Option("--min-tokens", help="Drop chunks below this token count."),
    ] = 100,
) -> None:
    """Generate FastEmbed embeddings for a CSV using markdown-aware sliding-window chunks."""
    if not csv_path.is_file():
        console.print(f"[red]CSV not found:[/red] {csv_path.resolve()}")
        raise typer.Exit(code=1)

    output_path = output or csv_path
    all_records = load_csv(csv_path)
    records = all_records
    if tenant_name is not None:
        records = [record for record in records if record.get(tenant_field) == tenant_name]

    if not records:
        console.print("[yellow]No matching records to embed.[/yellow]")
        raise typer.Exit(code=1)

    chunked_records = chunk_records_for_embedding(
        records,
        text_field=text_field,
        window_size=window_size,
        overlap=overlap,
        min_tokens=min_tokens,
    )
    if not chunked_records:
        console.print("[yellow]No chunks survived chunking. Lower --min-tokens or check the text column.[/yellow]")
        raise typer.Exit(code=1)

    final_data = []
    total_batches = (len(chunked_records) + batch_size - 1) // batch_size
    console.print(
        f"Embedding {len(chunked_records)} chunk(s) from {len(records)} row(s) "
        f"with {model!r} into {output_path}"
    )

    for batch_num, record_batch in enumerate(batch_list(chunked_records, batch_size), start=1):
        console.print(f"Embedding batch {batch_num}/{total_batches} ({len(record_batch)} chunks)")
        texts = [record[text_field] for record in record_batch]
        embeddings = get_batch_fastembed_embedding(texts, model=model, batch_size=batch_size)

        for record, embedding in zip(record_batch, embeddings, strict=True):
            item = dict(record)
            item["embedding"] = embedding
            final_data.append(item)

    same_output_as_input = output_path.resolve() == csv_path.resolve()
    if same_output_as_input and tenant_name is not None:
        untouched_records = [
            record for record in all_records
            if record.get(tenant_field) != tenant_name
        ]
        df = pd.DataFrame([*untouched_records, *final_data])
    else:
        df = pd.DataFrame(final_data)
    df["embedding"] = df["embedding"].apply(json.dumps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    vector_size = len(final_data[0]["embedding"]) if final_data else 0
    console.print(
        f"[green]Wrote[/green] {len(final_data)} embedded chunk row(s) to {output_path} "
        f"(vector size: {vector_size})"
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


@app.command("export-points")
def export_points_command(
    collection_name: Annotated[
        str,
        typer.Option("--collection-name", "--collection_name", "-c", help="Collection to dump."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSONL path (writes sibling .meta.json). Default: data/backups/<collection>.jsonl",
        ),
    ] = None,
    tenant_name: Annotated[
        str | None,
        typer.Option(
            "--tenant-name",
            "--tenant_name",
            "-t",
            help="Export only this shard key (optional; default = all shards).",
        ),
    ] = None,
    scroll_batch: Annotated[
        int,
        typer.Option("--scroll-batch", help="Scroll page size."),
    ] = 256,
) -> None:
    """Export all vectors + payloads to JSONL for migration (custom shards supported)."""
    settings = load_settings()
    client = qdrant_client(settings)
    out_path = output or Path("data/backups") / f"{collection_name}.jsonl"
    meta = export_points(
        client,
        settings,
        collection_name=collection_name,
        output_path=out_path,
        tenant_name=tenant_name,
        scroll_batch=scroll_batch,
    )
    console.print(
        f"[green]Exported[/green] {meta.points_total} point(s) from {collection_name!r} → {out_path.resolve()}\n"
        f"Meta: {out_path.with_suffix('.meta.json').resolve()}"
    )


@app.command("import-points")
def import_points_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="JSONL from export-points (.meta.json sidecar recommended)."),
    ],
    collection_name: Annotated[
        str | None,
        typer.Option(
            "--collection-name",
            "--collection_name",
            "-c",
            help="Target collection on this server (defaults to name in .meta.json).",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Points per upsert batch."),
    ] = 200,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Parse file and count points without upserting."),
    ] = False,
    tenant_name: Annotated[
        str | None,
        typer.Option("--tenant-name", "--tenant_name", "-t", help="Only import points for this shard key."),
    ] = None,
) -> None:
    """Import a JSONL backup (e.g. after pointing .env at the new Qdrant host)."""
    settings = load_settings()
    client = qdrant_client(settings)
    n = import_points(
        client,
        settings,
        input_path=input_path,
        collection_name=collection_name,
        batch_size=batch_size,
        dry_run=dry_run,
        tenant_name=tenant_name,
    )
    suffix = " (dry-run)" if dry_run else ""
    console.print(f"[green]Imported[/green] {n} point(s){suffix} into {collection_name or '«from meta»'}")


@app.command("restore-tenant")
def restore_tenant_command(
    collection_name: Annotated[
        str,
        typer.Option("--collection-name", "--collection_name", "-c", help="Collection that owns the tenant."),
    ],
    tenant_name: Annotated[
        str,
        typer.Option("--tenant-name", "--tenant_name", "-t", help="Tenant shard key to recreate and restore."),
    ],
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="JSONL backup file produced by export-points."),
    ],
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Points per upsert batch."),
    ] = 200,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and count matching points without writing."),
    ] = False,
) -> None:
    """Recreate a tenant shard key and restore its points from a JSONL backup."""
    settings = load_settings()
    provisioner = QdrantProvisioner(settings)

    if not dry_run:
        with console.status(f"Ensuring shard key [bold]{tenant_name}[/bold] on [bold]{collection_name}[/bold]…"):
            result = provisioner.ensure(collection_name, [tenant_name])
        if result.created_shard_keys:
            console.print(f"[green]Created[/green] shard key {tenant_name!r}")
        else:
            console.print(f"[dim]Shard key {tenant_name!r} already existed[/dim]")

    client = qdrant_client(settings)
    n = import_points(
        client,
        settings,
        input_path=input_path,
        collection_name=collection_name,
        batch_size=batch_size,
        dry_run=dry_run,
        tenant_name=tenant_name,
    )
    suffix = " (dry-run)" if dry_run else ""
    console.print(
        f"[green]Restored[/green] {n} point(s){suffix} for tenant {tenant_name!r} "
        f"in collection {collection_name!r}"
    )


@app.command("snapshot-create")
def snapshot_create_command(
    collection_name: Annotated[str, typer.Argument(help="Collection to snapshot.")],
) -> None:
    """Create a Qdrant native snapshot for a collection."""
    settings = load_settings()
    client = qdrant_client(settings)
    with console.status(f"Creating snapshot for [bold]{collection_name}[/bold]…"):
        info = create_snapshot(client, collection_name)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Name", info.name)
    table.add_row("Creation time", str(info.creation_time))
    table.add_row("Size (bytes)", str(info.size))
    console.print(table)
    console.print(
        f"[green]Snapshot created.[/green] "
        f"Download with: snapshot-download {collection_name} {info.name!r}"
    )


@app.command("snapshot-list")
def snapshot_list_command(
    collection_name: Annotated[str, typer.Argument(help="Collection to list snapshots for.")],
) -> None:
    """List all available snapshots for a collection."""
    settings = load_settings()
    client = qdrant_client(settings)
    snapshots = list_snapshots(client, collection_name)
    if not snapshots:
        console.print(f"[yellow]No snapshots found for {collection_name!r}.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("#")
    table.add_column("Name")
    table.add_column("Creation time")
    table.add_column("Size (bytes)")
    for i, s in enumerate(snapshots, start=1):
        table.add_row(str(i), s.name, str(s.creation_time), str(s.size))
    console.print(table)


@app.command("snapshot-download")
def snapshot_download_command(
    collection_name: Annotated[str, typer.Argument(help="Source collection name.")],
    snapshot_name: Annotated[str, typer.Argument(help="Snapshot name (from snapshot-list).")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Local path to save the snapshot. Default: data/snapshots/<name>"),
    ] = None,
) -> None:
    """Download a snapshot file from the Qdrant server to a local path."""
    settings = load_settings()
    out_path = output or Path("data/snapshots") / snapshot_name
    saved = download_snapshot(settings, collection_name, snapshot_name, out_path)
    console.print(f"[green]Downloaded[/green] → {saved}")
    console.print(
        f"Restore with: snapshot-restore {saved} {collection_name}"
    )


@app.command("snapshot-restore")
def snapshot_restore_command(
    snapshot_path: Annotated[Path, typer.Argument(help="Local .snapshot file to upload.")],
    collection_name: Annotated[str, typer.Argument(help="Target collection name on the server.")],
    priority: Annotated[
        str,
        typer.Option(
            "--priority",
            help="Conflict resolution: 'snapshot' (snapshot wins) or 'replica' (existing data wins).",
        ),
    ] = "snapshot",
) -> None:
    """Upload a local snapshot and restore it into a collection."""
    settings = load_settings()
    with console.status(f"Restoring [bold]{snapshot_path.name}[/bold] → [bold]{collection_name}[/bold]…"):
        restore_snapshot(settings, snapshot_path, collection_name, priority=priority)
    console.print(f"[green]Restored[/green] {snapshot_path.name} → collection {collection_name!r}")


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
