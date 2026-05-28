import torch
from transformers import XCLIPModel, XCLIPProcessor

class VideoTextEmbedder:
    def __init__(self, model_name: str = "microsoft/xclip-base-patch16-zero-shot"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} onto execution device: {self.device}...")
        
        # Load the pre-trained processor and model weights
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model = XCLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Put layers in inference mode

    def classify_video(self, video_frames: list, candidate_labels: list[str]) -> dict:
        """
        Phase 1 Method: Takes video frames and text labels, and returns probabilities.
        """
        inputs = self.processor(
            text=candidate_labels,
            videos=[list(video_frames)],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_video = outputs.logits_per_video  
            probs = logits_per_video.softmax(dim=-1).cpu().numpy()[0]

        return {label: float(score) for label, score in zip(candidate_labels, probs)}

    def get_video_feature_vector(self, video_frames: list) -> list[float]:
        """
        Phase 2 Method: Converts video frames into a normalized mathematical vector.
        """
        inputs = self.processor(videos=list(video_frames), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # get_video_features internally handles the vision model AND the projection
            raw_outputs = self.model.get_video_features(**inputs)
            video_features = raw_outputs.pooler_output
            
            # Normalize the vector
            video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
            
        return video_features.cpu().numpy()[0].tolist()

    def get_text_feature_vector(self, text_query: str) -> list[float]:
        """
        Phase 2 Method: Converts a text search query into a normalized mathematical vector.
        """
        inputs = self.processor(text=[text_query], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get the raw output from the text sub-model directly
            text_outputs = self.model.text_model(**inputs)
            
            # Extract pooled features and manually apply the projection layer
            text_features = self.model.text_projection(text_outputs.pooler_output)
            
            # Normalize the vector
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()[0].tolist()