# Zero-Shot Multimodal Video Retrieval Engine



A spatio-temporal AI search engine that retrieves video files based on natural language queries without relying on frame-by-frame 2D classification. This project leverages zero-shot cross-modal alignment to map raw video streams and text inputs into a shared 512-dimensional vector space for high-speed semantic retrieval.



### System Architecture



Instead of traditional, computationally heavy object detection loops, this pipeline decouples frame decoding from embedding generation.



Data Ingestion: Uniform temporal subsampling extracts exactly 32 frames per video, ensuring temporal consistency regardless of the original video's length or framerate.



Feature Extraction (The Backbone): Utilizes a pre-trained X-CLIP Vision Transformer (microsoft/xclip-base-patch16-zero-shot) to capture deep spatio-temporal dependencies.



Vector Database: High-dimensional embeddings are indexed using Qdrant (Cosine Similarity) for instantaneous semantic search, bypassing linear scan bottlenecks.



### Engineering Challenges \& Optimizations



Overcoming Memory Bottlenecks: Raw video tensors (\[Batch, Channels, Frames, Height, Width]) easily trigger Out-Of-Memory (OOM) errors. By decoupling decoding from embedding and strictly enforcing 32-frame spatial downscaling, the application scales predictably.



Complex Tensor Flattening: Navigated undocumented Hugging Face processor constraints by intercepting the raw 5D vision outputs and safely flattening them into 4D representations (\[batch\_size \* num\_frames, channels, height, width]) before pushing them through the spatio-temporal layers.



Native Dimensionality Projection: Engineered a bypass to prevent double-projection errors by isolating the pooler\_output directly, neatly extracting the finalized 512-dimensional mathematical vectors.



### Tech Stack



Machine Learning: PyTorch, Hugging Face transformers (X-CLIP)



Computer Vision: OpenCV (cv2)



Vector Search: Qdrant (qdrant-client)



Data Processing: NumPy

