import cv2

def load_video_chunks(video_path: str, chunk_duration_sec: float = 5.0):
    """
    Yields video chunks one at a time to prevent RAM exhaustion.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video framerate (default to 30 if cannot be read)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        
    frames_per_chunk = int(fps * chunk_duration_sec)
    
    current_chunk_frames = []
    start_frame = 0
    
    while True:
        ret, frame = cap.read()
        
        # If we reach the end of the video
        if not ret:
            if len(current_chunk_frames) > 0:
                end_sec = (start_frame + len(current_chunk_frames)) / fps
                yield {
                    "start_sec": start_frame / fps,
                    "end_sec": end_sec,
                    "frames": current_chunk_frames
                }
            break
            
        # OpenCV reads in BGR format, AI models expect RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_chunk_frames.append(frame_rgb)
        
        # Once we hit exactly 5 seconds of frames, pause and yield to the GPU
        if len(current_chunk_frames) >= frames_per_chunk:
            end_sec = (start_frame + frames_per_chunk) / fps
            
            yield {
                "start_sec": start_frame / fps,
                "end_sec": end_sec,
                "frames": current_chunk_frames
            }
            
            # Reset the buffer for the next 5 seconds
            start_frame += frames_per_chunk
            current_chunk_frames = []
            
    cap.release()