from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, ScalarQuantization, ScalarQuantizationConfig, ScalarType

print("--- Starting Local Vector Quantization ---")
client = QdrantClient(path="qdrant_db")
old_collection = "video_chunks"
new_collection = "video_chunks_int8"

# 1. Create the new compressed collection
print(f"Creating new INT8 compressed collection: '{new_collection}'...")
if client.collection_exists(new_collection):
    client.delete_collection(new_collection)

client.create_collection(
    collection_name=new_collection,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True
        )
    )
)

# 2. Extract and migrate the math
print(f"Migrating dense float32 vectors from '{old_collection}'...")
offset = None
total_migrated = 0

while True:
    records, offset = client.scroll(
        collection_name=old_collection,
        limit=500,
        offset=offset,
        with_vectors=True,
        with_payload=True
    )
    
    if not records:
        break
        
    client.upload_points(
        collection_name=new_collection,
        points=records
    )
    total_migrated += len(records)
    print(f"  -> Compressed and migrated {total_migrated} vectors so far...")
    
    if offset is None:
        break

print(f"\n✅ Success! {total_migrated} vectors are now compressed to 8-bit integers in '{new_collection}'.")