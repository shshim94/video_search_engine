import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VideoDatabase:
    def __init__(self, collection_name: str = "video_chunks", vector_size: int = 512):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(path="local_vector_db") 

    def setup_collection(self):
        # 1. If an old database exists, delete it
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
            
        # 2. Create a brand new, empty database architecture
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        print(f"Fresh '{self.collection_name}' database collection created!")

    def index_video_chunk(self, video_path: str, embedding: list[float], start_sec: float, end_sec: float, camera_id: str):
        """Saves a 5-second video vector into the database with its temporal metadata."""
        point_id = str(uuid.uuid4())
        
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
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_results
        )
        return results.points