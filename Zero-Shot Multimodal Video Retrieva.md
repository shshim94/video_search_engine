# 🎯 Multi-Target Multi-Camera (MTMC) Tracker

A Zero-Shot Semantic Search Engine for tracking targets across multi-camera video networks. This pipeline utilizes Microsoft's XCLIP AI model and a Qdrant vector database to allow users to locate specific subjects (e.g., "a person wearing a red shirt") across hours of unannotated spatial video feeds.

## ✨ Key Features

* **Zero-Shot Semantic Tracking:** Search for visual targets using natural language without needing pre-labeled bounding boxes or specific training data.
* **Deep Recursive Ingestion:** Intelligently scans complex, nested directory structures (e.g., `Warehouse_XXX/videos/`) to automatically locate and process unindexed `.mp4` feeds.
* **Anti-Lock Database Architecture:** Utilizes a decoupled Qdrant local storage engine (`local_vector_db`) to completely bypass Windows OS permission blocks, Antivirus scanning locks, and IDE read-locks.
* **Automated Factory Resets:** Includes a `--reindex` CLI flag that safely wipes broken/old database structures and rebuilds the collection architecture from scratch.
* **Memory-Safe Processing:** Processes video streams in memory-managed 5-second temporal chunks, utilizing garbage collection and batched GPU tensor processing to prevent RAM exhaustion.
* **Interactive UI:** A Streamlit dashboard featuring an Altair-powered Spatio-Temporal Gantt chart to visualize re-identification timelines.

---

## 🛠️ Installation & Setup

**1. Clone the Repository**
git clone https://github.com/<YOUR_USERNAME>/video_search_engine.git
cd video_search_engine

**2. Create the Virtual Environment**
python -m venv venv
venv\Scripts\activate  # On Mac/Linux use: source venv/bin/activate

**3. Install Dependencies**
pip install -r requirements.txt

**4. Download the Sample Dataset**
*Downloads the NVIDIA PhysicalAI SmartSpaces validation videos.*
python download_nvidia_data.py

---

## 🚀 Usage Guide

### Phase 1: Video Ingestion & Indexing
Before you can search, the AI needs to process the camera feeds, generate 512-dimensional feature vectors, and store them in the Qdrant database.

Run the ingestion script:
python main.py --data "data/MTMC_Tracking_2026/val"

**Need to start over?** If you want to clear the database and force a fresh ingestion, use the reindex flag:
python main.py --data "data/MTMC_Tracking_2026/val" --reindex

### Phase 2: Semantic Search Dashboard
Once the indexing is complete, launch the Streamlit frontend to start tracking targets.

streamlit run app.py

1. Open the local web address provided in your terminal.
2. Type a natural language query into the search bar.
3. Adjust the **Confidence Filter Threshold** to refine your matches.
4. View the generated spatio-temporal timeline and the verified source footage!

---

## 🗂️ Project Structure

* `main.py` - The core CLI engine for ingestion and database resets.
* `app.py` - The Streamlit interactive web dashboard.
* `src/embedder.py` - Handles the XCLIP model and PyTorch tensor generation.
* `src/vector_store.py` - Manages the Qdrant connection and PointStruct schema.
* `src/video_loader.py` - Yields memory-safe video chunks using OpenCV.
* `compress_local_db.py` - Utility to quantize float32 vectors into 8-bit integers.
