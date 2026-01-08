import streamlit as st
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
# Main Layout
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Original Feed")
    original_placeholder = st.empty()

with col2:
    st.subheader("✨ AI Deblurred Feed")
    processed_placeholder = st.empty()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚙️ Control Panel")

# ✅ INPUT OPTIONS
st.sidebar.subheader("📂 Input Source")

video_file = st.sidebar.file_uploader(
    "🎥 Upload Wagon Video",
    type=["mp4", "avi", "mov"]
)

image_files = st.sidebar.file_uploader(
    "🖼 Upload Wagon Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.sidebar.divider()

# System stats
st.sidebar.subheader("System Stats")
fps_placeholder = st.sidebar.empty()
gpu_placeholder = st.sidebar.empty()

st.sidebar.divider()

# Defects
st.sidebar.subheader("Detected Defects")
defect_placeholder = st.sidebar.empty()

# -------------------------
# Dummy Images (for feeds)
# -------------------------
dummy_original = Image.new("RGB", (640, 360), "#1f2933")
dummy_processed = Image.new("RGB", (640, 360), "#111827")

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

# CASE 1: VIDEO UPLOADED
if video_file is not None:
    st.success("✅ Video uploaded successfully")

    st.subheader("🎬 Video Preview")
    st.video(video_file)

    original_placeholder.image(dummy_original, width=700)
    processed_placeholder.image(dummy_processed, width=700)

# CASE 2: IMAGES UPLOADED
elif image_files:
    st.success(f"✅ {len(image_files)} image(s) uploaded")

    # Show first image as feed
    img = Image.open(image_files[0])

    original_placeholder.image(img, width=700)
    processed_placeholder.image(img, width=700)

    st.subheader("🖼 Uploaded Images Preview")
    st.image(image_files, width=150)

# CASE 3: NOTHING UPLOADED
else:
    st.info("👈 Upload a video or images from the sidebar to start inspection")
    original_placeholder.image(dummy_original, width=700)
    processed_placeholder.image(dummy_processed, width=700)

# -------------------------
# Update Stats (always)
# -------------------------
fps_placeholder.metric("FPS", random.randint(42, 48))
gpu_placeholder.metric("GPU Usage", f"{random.randint(55, 70)} %")

defect_placeholder.markdown(
    "\n".join(f"- ⚠️ {d}" for d in random.sample(defects, 3))
)
