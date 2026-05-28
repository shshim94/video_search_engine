import torch
from transformers import XCLIPModel, XCLIPProcessor

class VideoTextEmbedder:
    def __init__(self, model_name: str = "microsoft/xclip-base-patch16-zero-shot"):
        # This will now successfully detect your RTX 2070
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} onto execution device: {self.device.upper()}...")
        
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model = XCLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_video_feature_vector(self, video_frames: list) -> list[float]:
        inputs = self.processor(videos=list(video_frames), return_tensors="pt").to(self.device)
        
        # AMP context manager for hardware acceleration
        with torch.no_grad(), torch.amp.autocast('cuda'):
            raw_outputs = self.model.get_video_features(**inputs)
            video_features = raw_outputs.pooler_output
            video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
            
        return video_features.cpu().numpy()[0].tolist()

    def get_text_feature_vector(self, text_query: str) -> list[float]:
        inputs = self.processor(text=[text_query], return_tensors="pt").to(self.device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_outputs = self.model.text_model(**inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()[0].tolist()