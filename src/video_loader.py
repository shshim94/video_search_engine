import cv2
import numpy as np

def load_video_chunks(filepath: str, chunk_duration_sec: float = 5.0, frames_per_chunk: int = 32) -> list[dict]:
    """
    Slices a video into temporal windows and extracts a uniform sample of frames per window.
    
    Returns:
        A list of dictionaries. Each dictionary contains:
        - 'start_sec': The start time of the chunk in seconds.
        - 'end_sec': The end time of the chunk in seconds.
        - 'frames': The extracted 32 frames for the Vision Transformer.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f"⚠️ Could not open video file: {filepath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate how many raw frames make up one time window (e.g., 5 secs * 30 fps = 150 frames)
    raw_frames_per_window = int(chunk_duration_sec * fps)
    
    chunks = []
    
    # Slide the window across the video
    for window_start_idx in range(0, total_frames, raw_frames_per_window):
        window_end_idx = min(window_start_idx + raw_frames_per_window, total_frames)
        
        # Edge Case: If the remaining tail of the video is shorter than our required 32 frames, skip it.
        if (window_end_idx - window_start_idx) < frames_per_chunk:
            continue 
            
        # Uniformly sample exactly 32 frames from within this specific time window
        frame_indices = np.linspace(window_start_idx, window_end_idx - 1, frames_per_chunk, dtype=int)
        
        window_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for the Hugging Face processor
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window_frames.append(frame)
        
        # If extraction was successful, bundle it with its temporal metadata
        if len(window_frames) == frames_per_chunk:
            start_sec = window_start_idx / fps
            end_sec = window_end_idx / fps
            
            chunks.append({
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "frames": window_frames
            })
            
    cap.release()
    return chunks