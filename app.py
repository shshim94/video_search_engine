import os
import streamlit as st
import pandas as pd
import altair as alt
from src.embedder import VideoTextEmbedder
from src.vector_store import VideoDatabase

st.set_page_config(page_title="MTMC Semantic Tracker", page_icon="🎥", layout="wide")

@st.cache_resource
def init_backend():
    embedder = VideoTextEmbedder()
    db = VideoDatabase(vector_size=512)
    return embedder, db

try:
    embedder, db = init_backend()
    backend_ready = True
except Exception as e:
    st.error(f"Error initializing AI models: {e}")
    backend_ready = False

st.title("🎥 Multi-Target Multi-Camera (MTMC) Tracker")
st.markdown("---")

if backend_ready:
    st.sidebar.header("Pipeline Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Filter Threshold", 0.0, 1.0, 0.15, 0.01)
    max_results = st.sidebar.slider("Max Results to Display", 1, 20, 8)

    search_query = st.text_input(
        "Enter target description to track across the network...",
        placeholder="e.g., 'a person wearing a red shirt'"
    )

    if st.button("Run Semantic Search", type="primary") and search_query.strip():
        with st.spinner("Synthesizing multi-camera timeline..."):
            text_vector = embedder.get_text_feature_vector(search_query)
            search_results = db.search_videos(text_vector, top_results=max_results)
            
            # Filter results based on the confidence slider
            valid_hits = [res for res in search_results if res.score >= confidence_threshold]
            
            if not valid_hits:
                st.info("No activities found matching that signature above the confidence threshold.")
            else:
                st.success(f"Tracked {len(valid_hits)} distinct target occurrences!")
                
                # Sort chronologically
                timeline_events = sorted(valid_hits, key=lambda x: x.payload['start_time'])
                
                # --- NEW: Build the Re-Identification Timeline Chart ---
                st.markdown("### 📍 Spatio-Temporal Target Tracking")
                
                # Convert our hits into a Pandas DataFrame for the chart
                chart_data = []
                for res in timeline_events:
                    chart_data.append({
                        "Camera": res.payload.get('camera_id', 'UNKNOWN'),
                        "Start Time (s)": res.payload.get('start_time', 0.0),
                        "End Time (s)": res.payload.get('end_time', 0.0),
                        "Confidence": res.score
                    })
                df = pd.DataFrame(chart_data)
                
                # Create an Altair Gantt Chart
                timeline_chart = alt.Chart(df).mark_bar(cornerRadius=3, height=20).encode(
                    x=alt.X("Start Time (s):Q", title="Video Timeline (Seconds)"),
                    x2="End Time (s):Q",
                    y=alt.Y("Camera:N", sort=None, title="Spatial Node"),
                    color=alt.Color("Confidence:Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="Match Strength")),
                    tooltip=["Camera", "Start Time (s)", "End Time (s)", "Confidence"]
                ).properties(height=200, width="container")
                
                st.altair_chart(timeline_chart, use_container_width=True)
                st.markdown("---")
                
                # --- Display the actual video clips below the timeline ---
                st.markdown("### 📼 Verified Source Footage")
                
                for idx, result in enumerate(timeline_events):
                    payload = result.payload
                    cam_id = payload.get('camera_id', 'UNKNOWN_CAM')
                    start = payload.get('start_time', 0.0)
                    file_path = payload.get('file_path', '')
                    
                    with st.container():
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            if os.path.exists(file_path):
                                st.video(file_path, start_time=int(start))
                            else:
                                st.error(f"Video missing: {file_path}")
                        with col2:
                            st.metric(label=f"Match #{idx+1} Confidence", value=f"{result.score:.4f}")
                            st.markdown(f"**Node:** `{cam_id}`\n\n**Timestamp:** `{start:05.2f}s`")
                        st.markdown("<br>", unsafe_allow_html=True)