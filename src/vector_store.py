from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VideoDatabase:
    def __init__(self, vector_size: int = 512): # X-CLIP outputs 512-dimensional vectors!
        # We use ":memory:" to run it locally without a heavy server setup
        self.client = QdrantClient(":memory:")
        self.collection_name = "video_library"
        
        # Initialize the database schema
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self.next_id = 0

    def index_video(self, video_path: str, embedding: list):
        """Saves the video embedding and its file path (metadata) to the database."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=self.next_id,
                    vector=embedding,
                    payload={"file_path": video_path} # Metadata so we know which video matched
                )
            ]
        )
        print(f"Indexed {video_path} into database with ID {self.next_id}")
        self.next_id += 1

    def search_videos(self, text_embedding: list, top_results: int = 1):
        """Searches the database for the closest video match."""
        # We now use query_points() instead of search()
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=text_embedding, # This parameter changed from 'query_vector' to 'query'
            limit=top_results
        ).points # We append .points to directly grab the list of matched videos
        
        return search_result