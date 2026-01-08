import streamlit as st
import random
import datetime
import time
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Rail-Vision | Smart Wagon Inspection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Styling
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
h1, h2, h3 {
    color: #e5e7eb;
}
.alert-critical {
    padding: 12px;
    border-radius: 8px;
    background-color: #7f1d1d;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "live"

if "history" not in st.session_state:
    st.session_state.history = []

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# -------------------------
# Dummy Images
# -------------------------
original_img = Image.new("RGB", (640, 360), "#1f2933")
processed_img = Image.new("RGB", (640, 360), "#111827")

# -------------------------
# Defects Pool
# -------------------------
defects_pool = [
    ("Paint Damage", "🟢 Minor"),
    ("Door Dent", "🟡 Medium"),
    ("Loose Coupling", "🟡 Medium"),
    ("Floor Crack", "🔴 Critical")
]

# -------------------------
# Top Bar
# -------------------------
top_l, top_r = st.columns([8, 1])

with top_l:
    st.markdown("Rail-Vision | Smart Wagon Inspection")

with top_r:
    if st.button("History"):
        st.session_state.page = "history"

st.divider()

# ==================================================
# LIVE PAGE
# ==================================================
if st.session_state.page == "live":

    # -------------------------
    # Inspection Timer
    # -------------------------
    elapsed = int(time.time() - st.session_state.start_time)
    mins, secs = divmod(elapsed, 60)

    st.markdown(f"**Status:** 🟢 Live Inspection Running | ⏱ {mins:02d}:{secs:02d}")

    # -------------------------
    # Unified Input
    # -------------------------
    st.subheader("Input Source")

    wagon_file = st.file_uploader(
        "Upload Wagon Video / Image",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
    )

    # -------------------------
    # Save Session
    # -------------------------
    defects = random.sample(defects_pool, 2)
    critical_present = any(d[1].startswith("🔴") for d in defects)

    if wagon_file:
        st.session_state.history.append({
            "id": len(st.session_state.history) + 1,
            "time": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
            "file": wagon_file.name,
            "defects": defects,
            "blur_before": random.randint(70, 90),
            "blur_after": random.randint(10, 30)
        })

    # -------------------------
    # Critical Alert Mode
    # -------------------------
    if critical_present:
        st.markdown(
            '<div class="alert-critical">CRITICAL DEFECT DETECTED — MANUAL INSPECTION REQUIRED</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # -------------------------
    # Summary Cards
    # -------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Wagons", random.randint(30, 60))
    c2.metric("Defects Found", random.randint(3, 8))
    c3.metric("Critical Alerts", random.randint(1, 3))
    c4.metric("Avg Blur Reduction", f"{random.randint(55, 70)}%")
    c5.metric("Inspection Confidence", f"{random.randint(88, 96)}%")

    st.divider()

    # ==================================================
    # 🔥 THREE CAMERA SIMULTANEOUS VIEW
    # ==================================================
    st.subheader("Multi-Camera View (Simultaneous)")

    cam1, cam2, cam3 = st.columns(3)

    with cam1:
        st.markdown("### CAM-01")
        st.image(original_img, caption="Original", use_container_width=True)
        st.image(processed_img, caption="AI Enhanced", use_container_width=True)

    with cam2:
        st.markdown("### CAM-02")
        st.image(original_img, caption="Original", use_container_width=True)
        st.image(processed_img, caption="AI Enhanced", use_container_width=True)

    with cam3:
        st.markdown("### CAM-03")
        st.image(original_img, caption="Original", use_container_width=True)
        st.image(processed_img, caption="AI Enhanced", use_container_width=True)

# ==================================================
# HISTORY PAGE
# ==================================================
elif st.session_state.page == "history":

    if st.button("Back"):
        st.session_state.page = "live"

    #st.markdown("##Inspection History")
    st.divider()

    if not st.session_state.history:
        st.info("No inspection history available.")
    else:
        for s in st.session_state.history:
            st.markdown(f"""
            ### Session {s['id']}
            - **Time:** {s['time']}
            - **File:** {s['file']}
            - **Blur Reduction:** {s['blur_before']}% → {s['blur_after']}%
            - **Defects:**
              - {s['defects'][0][1]} – {s['defects'][0][0]}
              - {s['defects'][1][1]} – {s['defects'][1][0]}
            """)
            st.divider()
