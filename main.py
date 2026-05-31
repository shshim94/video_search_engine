import os
import argparse
import gc
import glob
from src.embedder import VideoTextEmbedder
from src.vector_store import VideoDatabase
from src.video_loader import load_video_chunks

def run_search_engine(data_folder, search_query, force_reindex):
    print("\n--- Booting up AI Models ---")
    embedder = VideoTextEmbedder()
    db = VideoDatabase(vector_size=512)

    # 1. Database Awareness: Check if we already have vectors to avoid re-processing
    try:
        collection_info = db.client.get_collection(db.collection_name)
        is_indexed = collection_info.vectors_count > 0
    except Exception:
        is_indexed = False

    # 2. Ingestion Phase
    if force_reindex or not is_indexed:
        print(f"\n--- Phase 2A: Processing Temporal Video Streams from '{data_folder}' ---")
        total_chunks_indexed = 0
        
        if force_reindex:
            print("Force reindex flag detected. Wiping old database...")
            db.setup_collection()
        
        search_pattern = os.path.join(data_folder, "Warehouse_*", "videos", "*.mp4")
        video_files = glob.glob(search_pattern)
        
        print(f"Found {len(video_files)} videos matching the Warehouse structure.")
        
        if not video_files:
            print(f"CRITICAL ERROR: No videos found matching pattern: {search_pattern}")
            return
            
        for filepath in video_files:
            # Extract just the filename from the deep path
            filename = os.path.basename(filepath)
            simulated_camera_id = filename.split('.')[0] 
            
            print(f"Processing stream: {simulated_camera_id}...")
            
            chunks = load_video_chunks(filepath, chunk_duration_sec=5.0)
            
            batch_size = 8
            chunk_batch = []
            
            for chunk in chunks:
                chunk_batch.append(chunk)
                
                if len(chunk_batch) == batch_size:
                    frames_list = [c["frames"] for c in chunk_batch]
                    vectors = embedder.get_batched_video_feature_vectors(frames_list)
                    
                    for i, v in enumerate(vectors):
                        c = chunk_batch[i]
                        db.index_video_chunk(
                            video_path=filepath,
                            embedding=v,
                            start_sec=c["start_sec"],
                            end_sec=c["end_sec"],
                            camera_id=simulated_camera_id
                        )
                        total_chunks_indexed += 1
                        
                    for c in chunk_batch:
                        if "frames" in c:
                            del c["frames"]
                    del chunk_batch
                    chunk_batch = []
                    gc.collect()
            
            if len(chunk_batch) > 0:
                frames_list = [c["frames"] for c in chunk_batch]
                vectors = embedder.get_batched_video_feature_vectors(frames_list)
                
                for i, v in enumerate(vectors):
                    c = chunk_batch[i]
                    db.index_video_chunk(
                        video_path=filepath,
                        embedding=v,
                        start_sec=c["start_sec"],
                        end_sec=c["end_sec"],
                        camera_id=simulated_camera_id
                    )
                    total_chunks_indexed += 1
                    
                for c in chunk_batch:
                    if "frames" in c:
                        del c["frames"]
                del chunk_batch
                gc.collect()
                
            del chunks
            gc.collect()
            
        print(f"\n✅ Indexing Complete! Added {total_chunks_indexed} video chunks to the database.")
    else:
        print("\n✅ Existing populated database detected. Skipping video ingestion phase.")

    # 3. Search Phase
    if search_query:
        print(f"\n--- Phase 2B: Searching for '{search_query}' ---")
        text_vector = embedder.get_text_feature_vector(search_query)
        
        search_results = db.search_videos(text_vector, top_results=5)
        
        print("\n📍 TARGET TRACKING TIMELINE:")
        if not search_results:
            print("No matches found.")
        else:
            timeline_events = sorted(search_results, key=lambda x: (x.payload.get('file_path', ''), x.payload.get('start_time', 0.0)))
            for idx, result in enumerate(timeline_events):
                payload = result.payload
                cam_id = payload.get('camera_id', 'UNKNOWN')
                start = payload.get('start_time', 0.0)
                end = payload.get('end_time', 0.0)
                print(f"[{idx+1}] Node: {cam_id} | Window: {start:05.2f}s to {end:05.2f}s | Confidence: {result.score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot MTMC Tracking Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to video directory")
    parser.add_argument("--query", type=str, default="", help="Semantic search query")
    parser.add_argument("--reindex", action="store_true", help="Force database rebuild")
    args = parser.parse_args()
    
    run_search_engine(data_folder=args.data, search_query=args.query, force_reindex=args.reindex)