import streamlit as st
import time
from PIL import Image
import random

# -------------------------
# Page Config (Wide Layout)
# -------------------------
st.set_page_config(
    page_title="Rail-Vision | High-Speed Inspection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom Dark Theme Styling
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: white;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Title & Status
# -------------------------
st.markdown("## 🚆 Rail-Vision | High-Speed Wagon Inspection")
st.markdown("**Status:** 🟢 System Ready — Processing Input")

st.divider()

# -------------------------
# Main Layout (Two Columns)
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Original Feed (Blurry / Dark)")
    original_placeholder = st.empty()

with col2:
    st.subheader("✨ AI Deblurred Feed (Enhanced)")
    processed_placeholder = st.empty()

# -------------------------
# Sidebar - Control Panel
# -------------------------
st.sidebar.title("⚙️ Control Panel")

st.sidebar.subheader("System Stats")
fps_placeholder = st.sidebar.empty()
gpu_placeholder = st.sidebar.empty()

st.sidebar.divider()

st.sidebar.subheader("Detected Defects")
defect_placeholder = st.sidebar.empty()

st.sidebar.divider()

st.sidebar.subheader("Engine Parameters")
st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.50)
st.sidebar.selectbox(
    "Model Version",
    ["Deblur-UNet v2 (TensorRT)", "GAN-Enhancer v1", "FastMotionNet"]
)

# -------------------------
# Load Sample Images
# -------------------------
original_img = Image.new("RGB", (640, 360), "#1f2933")
processed_img = Image.new("RGB", (640, 360), "#111827")

# -------------------------
# Simulated Video Loop
# -------------------------
defects = [
    "Door Dent Detected",
    "Floor Crack Detected",
    "Loose Coupling",
    "Paint Damage",
    "No Defect"
]

for _ in range(200):  # simulate frames
    # ✅ FIXED IMAGE WIDTH (INTEGER)
    original_placeholder.image(original_img, width=700)
    processed_placeholder.image(processed_img, width=700)

    # Update system stats
    fps_placeholder.metric("FPS", random.randint(42, 48))
    gpu_placeholder.metric("GPU Usage", f"{random.randint(55, 70)} %")

    # Update defect log
    defect_list = random.sample(defects, 3)
    defect_placeholder.markdown(
        "\n".join(
            [f"- ⚠️ {d}" if d != "No Defect" else "- ✅ No Defect"
             for d in defect_list]
        )
    )

    time.sleep(0.08)  # ~12 FPS simulation
