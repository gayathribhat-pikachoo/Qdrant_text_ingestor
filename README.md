# How to Run

## Help

```powershell
uv run python cli.py --help
uv run python cli.py create --help
```

## Create Collections

```powershell
uv run python cli.py create diy-dasra-prod -t DASRA
uv run python cli.py create diy-selco-prod -t SELCO_MILLET -t LEMELSON -t SELCO_INTERNAL
uv run python cli.py create diy-dasra-prod -t DASRA --force-recreate
```

## List Collections

```powershell
uv run python cli.py list
uv run python cli.py list --dict
```

## Persist Collections

Save the current Qdrant collection-to-tenant mapping:

```powershell
uv run python cli.py persist
uv run python cli.py persist --output data/collections.json
```

Restore collections and tenant shard keys from the saved mapping:

```powershell
uv run python cli.py restore
uv run python cli.py restore --input data/collections.json
```



## Delete

```powershell
uv run python cli.py delete-collection diy-dasra-prod
uv run python cli.py delete-collection diy-dasra-prod --yes
uv run python cli.py delete-tenants diy-selco-prod -t SELCO_MILLET -t LEMELSON
uv run python cli.py delete-tenants diy-selco-prod -t SELCO_MILLET --yes
```

## Naming

- `tenant_name` is the payload field that gets a keyword index.
- The same tenant name value is used as the Qdrant `shard_key`.


### Running on server
```powershell
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6334 \
  -p 6334:6334 \
  -p 6335:6335 \
  -v qdrant_storage:/qdrant/storage \
  -e QDRANT__CLUSTER__ENABLED=true \
  qdrant/qdrant:latest \
  ./qdrant --uri http://10.1.0.71:6335
```

powershell```
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6334 \
  -p 6334:6334 \
  -p 6335:6335 \
  -v qdrant_storage:/qdrant/storage \
  -e QDRANT__CLUSTER__ENABLED=true \
  -e QDRANT__CLUSTER__P2P__PORT=6335 \
  -e QDRANT__CLUSTER__URI=http://10.1.0.71:6335 \
  -e QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=512 \
  qdrant/qdrant:latest \
  ./qdrant --uri http://10.1.0.71:6335

  ```

### Dashboard View
```powershell
http://10.1.0.71:6335/dashboard
```


### Ingest
```powershell
uv run python cli.py ingest --collection_name diy-pcw-prod --tenant_name PRIMEMEGHALAYA
uv run python cli.py ingest --collection_name diy-pcw-prod --tenant_name PRIMEMEGHALAYA --csv path/to/qdrant.csv --batch-size 400
```


### Query
```powershell
uv run python cli.py ingest -c my_collection -t my_tenant -f path/to/qdrant.csv
uv run python cli.py parallel-query -c my_collection -t my_tenant -Q "your question"

### FastEmbed Embedding Generation
uv run python cli.py embed-fastembed --csv "Qdrant_text_ingestor/data/qdrant.csv" --tenant-name PCW