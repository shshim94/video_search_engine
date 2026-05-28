import os
import argparse
from src.video_loader import load_video_chunks
from src.embedder import VideoTextEmbedder
from src.vector_store import VideoDatabase

def run_search_engine(data_folder: str, search_query: str):
    print("--- Booting up AI Models ---")
    embedder = VideoTextEmbedder()
    db = VideoDatabase(vector_size=512)

    print(f"\n--- Phase 2A: Processing Temporal Video Streams from '{data_folder}/' ---")
    
    if not os.path.exists(data_folder):
        print(f"⚠️ ERROR: Python cannot find a folder named '{data_folder}'.")
        return

    all_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".mp4")]
    total_chunks_indexed = 0
    
    # 1. Slide the temporal window across all videos
    for filename in all_files:
        filepath = os.path.join(data_folder, filename)
        
        # Simulate camera IDs based on file names for the prototype
        simulated_camera_id = f"camera_{filename.split('.')[0].lower()}"
        
        print(f"Processing stream: {filename}...")
        chunks = load_video_chunks(filepath, chunk_duration_sec=5.0)
        
        for chunk in chunks:
            # Extract the vector for this specific 5-second window
            video_vector = embedder.get_video_feature_vector(chunk["frames"])
            
            # Save it to the database with its exact timestamps
            db.index_video_chunk(
                video_path=filepath,
                embedding=video_vector,
                start_sec=chunk["start_sec"],
                end_sec=chunk["end_sec"],
                camera_id=simulated_camera_id
            )
            total_chunks_indexed += 1

    print(f"✅ Indexed {total_chunks_indexed} separate temporal chunks into the vector database.")

    print("\n--- Phase 2B: Semantic Timeline Synthesis ---")
    print(f"User Query: '{search_query}'")

    # 2. Search the database for the top 5 matches
    text_vector = embedder.get_text_feature_vector(search_query)
    
    # We lower the confidence threshold just slightly to catch events tracking across space
    search_results = db.search_videos(text_vector, top_results=5)

    if not search_results:
        print("\n❌ No matches found in the database.")
        return

    # 3. Sort the results chronologically to build a tracking timeline
    # We sort by file path first (to group by camera/video), then by start time
    timeline_events = sorted(search_results, key=lambda x: (x.payload['file_path'], x.payload['start_time']))

    print("\n📍 TARGET TRACKING TIMELINE:")
    for result in timeline_events:
        score = result.score
        cam = result.payload['camera_id']
        start = result.payload['start_time']
        end = result.payload['end_time']
        
        # Only show strong semantic matches
        if score > 0.20:  
            print(f" ⏱️ [{start:05.2f}s - {end:05.2f}s] | {cam.upper()} | Confidence: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot IVA Tracking System")
    parser.add_argument("--data", type=str, default="data", help="Path to the folder containing video streams")
    parser.add_argument("--query", type=str, required=True, help="Text description of the event to track")
    
    args = parser.parse_args()
    run_search_engine(data_folder=args.data, search_query=args.query)