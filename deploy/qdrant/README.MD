docker run --rm \
  --network container:qdrant \
  -e QDRANT_HOST=localhost \
  -e QDRANT_PORT=6333 \
  -e QDRANT_GRPC_PORT=6334 \
  -e QDRANT_PREFER_GRPC=true \
  -e QDRANT_VECTOR_SIZE=1024 \
  -e QDRANT_TIMEOUT=600 \
  -v /home/ubuntu/qdrant.csv:/app/Qdrant_text_ingestor/data/qdrant.csv:ro \
  -v fastembed_cache:/root/.cache \
  multi-collection-ingestor:fastembed \
  python -c "from Qdrant_text_ingestor.fastembed import create_fastembed_collection, upsert_fastembed_csv; create_fastembed_collection('diy-pcw-prod-fastembed', ['PCW']); upsert_fastembed_csv('diy-pcw-prod-fastembed', 'PCW', embed_batch_size=16, upsert_batch_size=100)"