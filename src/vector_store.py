import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class VideoDatabase:
    def __init__(self, collection_name: str = "video_chunks", vector_size: int = 512):
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")  # Using local memory for the prototype
        
        # Recreate the collection to ensure a clean slate
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def index_video_chunk(self, video_path: str, embedding: list[float], start_sec: float, end_sec: float, camera_id: str):
        """Saves a 5-second video vector into the database with its temporal metadata."""
        point_id = str(uuid.uuid4()) # Generate a unique ID for this specific chunk
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file_path": video_path,
                        "camera_id": camera_id,
                        "start_time": start_sec,
                        "end_time": end_sec
                    }
                )
            ]
        )

    def search_videos(self, query_vector: list[float], top_results: int = 5) -> list:
        """Searches the database and returns multiple matches to build a timeline."""
        # Updated for Qdrant v1.10+ using query_points instead of search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_results
        )
        # We append .points to return the raw list of matches just like the old API did
        return results.points