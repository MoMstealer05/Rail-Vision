import streamlit as st
import time
import random
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Rail-Vision | High-Speed Inspection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Dark Theme Styling
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: white;
    }
    h1, h2, h3 {
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Title
# -------------------------
st.markdown("## 🚆 Rail-Vision | High-Speed Wagon Inspection")
st.markdown("**Status:** 🟢 System Ready")
st.divider()

# -------------------------
# MAIN LAYOUT
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Original Feed")
    original_placeholder = st.empty()

with col2:
    st.subheader("✨ AI Deblurred Feed")
    processed_placeholder = st.empty()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("⚙️ Control Panel")

# ✅ VIDEO INPUT OPTION (THIS IS WHAT YOU WANT)
st.sidebar.subheader("🎥 Video Input")

video_file = st.sidebar.file_uploader(
    "Upload Wagon Inspection Video",
    type=["mp4", "avi", "mov"]
)

st.sidebar.divider()

st.sidebar.subheader("System Stats")
fps_placeholder = st.sidebar.empty()
gpu_placeholder = st.sidebar.empty()

st.sidebar.divider()

st.sidebar.subheader("Detected Defects")
defect_placeholder = st.sidebar.empty()

# -------------------------
# DUMMY IMAGES (for UI)
# -------------------------
original_img = Image.new("RGB", (640, 360), "#1f2933")
processed_img = Image.new("RGB", (640, 360), "#111827")

defects = [
    "Door Dent Detected",
    "Floor Crack Detected",
    "Loose Coupling",
    "Paint Damage",
    "No Defect"
]

# -------------------------
# DISPLAY LOGIC
# -------------------------
if video_file is not None:
    st.success("✅ Video uploaded successfully")

    # Show uploaded video (THIS IS INPUT CONFIRMATION)
    st.subheader("🎬 Uploaded Video Preview")
    st.video(video_file)

    # Keep your dashboard alive
    original_placeholder.image(original_img, width=700)
    processed_placeholder.image(processed_img, width=700)

    fps_placeholder.metric("FPS", random.randint(42, 48))
    gpu_placeholder.metric("GPU Usage", f"{random.randint(55, 70)} %")

    defect_placeholder.markdown(
        "\n".join(f"- ⚠️ {d}" for d in random.sample(defects, 3))
    )

else:
    st.info("👈 Upload a wagon video from the sidebar to start inspection")
