import torch
from transformers import XCLIPProcessor, XCLIPModel

class VideoTextEmbedder:
    def __init__(self, model_name="microsoft/xclip-base-patch16-zero-shot"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} onto execution device: {self.device}...")
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model = XCLIPModel.from_pretrained(model_name).to(self.device)

    def get_text_feature_vector(self, text_query: str):
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            raw_outputs = self.model.get_text_features(**inputs)
            
        # Safely extract the actual tensor from the Hugging Face Dataclass
        if hasattr(raw_outputs, "pooler_output"):
            embeds = raw_outputs.pooler_output
        elif hasattr(raw_outputs, "text_embeds"):
            embeds = raw_outputs.text_embeds
        elif isinstance(raw_outputs, tuple):
            embeds = raw_outputs[0]
        else:
            embeds = raw_outputs
            
        # Ensure the vector is projected to the 512-dimension size for Qdrant
        if embeds.shape[-1] != 512 and hasattr(self.model, "text_projection"):
            embeds = self.model.text_projection(embeds)
            
        # Normalize the vector for cosine similarity search
        text_features = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        
        return text_features.squeeze(0).cpu().numpy().tolist()

    def get_video_feature_vector(self, frames):
        import numpy as np
        import torch
        
        required_frames = getattr(self.model.config.vision_config, "num_frames", 8)
        
        if len(frames) > required_frames:
            indices = np.linspace(0, len(frames) - 1, required_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        elif len(frames) < required_frames:
            padding = [frames[-1]] * (required_frames - len(frames))
            sampled_frames = frames + padding
        else:
            sampled_frames = frames

        inputs = self.processor(images=sampled_frames, return_tensors="pt")
        
        if "pixel_values" not in inputs:
            inputs = self.processor(videos=sampled_frames, return_tensors="pt")
            
        pixel_values = inputs["pixel_values"].to(self.device)
        
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)

        with torch.no_grad():
            raw_outputs = self.model.get_video_features(pixel_values=pixel_values)
            
        if hasattr(raw_outputs, "pooler_output"):
            embeds = raw_outputs.pooler_output
        elif hasattr(raw_outputs, "video_embeds"):
            embeds = raw_outputs.video_embeds
        elif isinstance(raw_outputs, tuple):
            embeds = raw_outputs[0]
        else:
            embeds = raw_outputs 
            
        if embeds.shape[-1] != 512 and hasattr(self.model, "visual_projection"):
            embeds = self.model.visual_projection(embeds)
            
        video_features = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        
        return video_features.squeeze(0).cpu().numpy().tolist()

    def get_batched_video_feature_vectors(self, list_of_frames):
        import numpy as np
        import torch
        
        required_frames = getattr(self.model.config.vision_config, "num_frames", 8)
        processed_batch = []
        
        for frames in list_of_frames:
            if len(frames) > required_frames:
                indices = np.linspace(0, len(frames) - 1, required_frames, dtype=int)
                sampled = [frames[i] for i in indices]
            elif len(frames) < required_frames:
                padding = [frames[-1]] * (required_frames - len(frames))
                sampled = frames + padding
            else:
                sampled = frames
            processed_batch.append(sampled)

        inputs = self.processor(images=processed_batch, return_tensors="pt")
        if "pixel_values" not in inputs:
            inputs = self.processor(videos=processed_batch, return_tensors="pt")
            
        pixel_values = inputs["pixel_values"].to(self.device)
        
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)

        with torch.no_grad():
            raw_outputs = self.model.get_video_features(pixel_values=pixel_values)
            
        if hasattr(raw_outputs, "pooler_output"):
            embeds = raw_outputs.pooler_output
        elif hasattr(raw_outputs, "video_embeds"):
            embeds = raw_outputs.video_embeds
        elif isinstance(raw_outputs, tuple):
            embeds = raw_outputs[0]
        else:
            embeds = raw_outputs
            
        if embeds.shape[-1] != 512 and hasattr(self.model, "visual_projection"):
            embeds = self.model.visual_projection(embeds)
            
        video_features = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        
        return video_features.cpu().numpy().tolist()