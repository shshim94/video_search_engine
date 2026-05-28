import os
from src.video_loader import load_video_frames
from src.embedder import VideoTextEmbedder
from src.vector_store import VideoDatabase

def run_search_engine():
    print("--- Booting up AI Models ---")
    embedder = VideoTextEmbedder()
    db = VideoDatabase(vector_size=512)

    data_folder = "data"
    
    print("\n--- Phase 2A: Indexing Database ---")
    
    # Check if Python can actually see the folder
    if not os.path.exists(data_folder):
        print(f"⚠️ ERROR: Python cannot find a folder named '{data_folder}' in its current directory: {os.getcwd()}")
    else:
        all_files = os.listdir(data_folder)
        print(f"👀 Python sees these files in the data folder: {all_files}")
        
        # 1. Loop through all files in the data folder
        for filename in all_files:
            if filename.lower().endswith(".mp4"): 
                filepath = os.path.join(data_folder, filename)
                
                # CHANGED: X-CLIP requires exactly 8 frames for this checkpoint
                frames = load_video_frames(filepath, num_frames=32)
                video_vector = embedder.get_video_feature_vector(frames)
                
                # Save it to Qdrant
                db.index_video(video_path=filepath, embedding=video_vector)

    print("\n--- Phase 2B: Semantic Video Search ---")
    # 2. Type whatever you want to search for here!
    search_query = "Find the video of a rocket launch"
    print(f"User Query: '{search_query}'")

    # 3. Convert text to vector and search Qdrant
    text_vector = embedder.get_text_feature_vector(search_query)
    search_results = db.search_videos(text_vector, top_results=1)

    # 4. Extract and display the winner
    if search_results:
        best_match = search_results[0].payload['file_path']
        score = search_results[0].score
        
        print(f"\n🎯 Top Match Found: {best_match} (Confidence Score: {score:.4f})")
    else:
        print("\n❌ No matches found in the database.")

if __name__ == "__main__":
    run_search_engine()