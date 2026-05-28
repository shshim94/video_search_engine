import cv2
import numpy as np

def load_video_frames(video_path: str, num_frames: int = 16) -> np.ndarray:
    """
    Opens a video file and samples exactly `num_frames` evenly spaced out across the video duration.
    Returns a numpy array of shape (num_frames, height, width, channels) in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        raise ValueError(f"Could not read video or video is empty: {video_path}")

    # Calculate uniform indices to sample frames across the entire duration
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            # OpenCV reads in BGR, we must convert it to RGB for Hugging Face models
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
    cap.release()

    # If the video cut short, pad it with the last valid frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames[:num_frames])
