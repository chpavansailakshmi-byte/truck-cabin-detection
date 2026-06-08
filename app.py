import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Truck Cabin Detection",
    page_icon="🚛",
    layout="wide"
)

# 2. Optimized Model Loading (Cached)
@st.cache_resource
def load_model():
    # Utilizing YOLOv8 model
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model 'best.pt'. Ensure the file is in the root directory. Details: {e}")
    st.stop()

# 3. Sidebar - Settings & Info
st.sidebar.title("🚛 Configuration & Info")

# Interactive Confidence Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Adjust the minimum confidence level required to display a detection."
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Model Specs:**
    * **Architecture:** YOLOv8n
    * **Target Class:** Cabin
    * **Training Epochs:** 50
    
    **Purpose:**
    Automated detection of truck cabins from CCTV and traffic camera feeds.
    """
)

# 4. Main App Interface
st.title("🚛 Truck Cabin Detection System")
st.markdown(
    "Upload an image from a traffic or CCTV feed, and the trained YOLOv8 model will automatically identify and locate truck cabins."
)

uploaded_file = st.file_uploader(
    "Choose an Image...",
    type=["jpg", "jpeg", "png"]
)

# 5. Core Processing Logic
if uploaded_file:
    # Open image and ensure RGB standard
    image = Image.open(uploaded_file).convert("RGB")
    
    # Layout preparation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with st.spinner("Running YOLOv8 Object Detection..."):
        # Direct inference using the PIL Image object (No temporary files needed)
        results = model.predict(
            source=image,
            conf=conf_threshold
        )
        
    result = results[0]
    
    # Extract plotted image (YOLO returns BGR numpy array, convert to RGB for Streamlit)
    detected_image_bgr = result.plot()
    detected_image_rgb = detected_image_bgr[:, :, ::-1] 
    
    with col2:
        st.subheader("Detection Result")
        st.image(detected_image_rgb, use_container_width=True)

    # 6. UI Metrics & Feedback
    st.markdown("---")
    st.subheader("📊 Detection Analytics")
    
    num_detections = len(result.boxes)
    
    if num_detections > 0:
        st.success(f"Successfully detected {num_detections} cabin(s)!")
        
        # Displaying metrics in columns dynamically
        metric_cols = st.columns(min(num_detections, 4)) # Cap at 4 columns per row visually
        
        for i, box in enumerate(result.boxes):
            conf_val = float(box.conf[0])
            col_index = i % 4
            
            with metric_cols[col_index]:
                st.metric(
                    label=f"Cabin {i+1}", 
                    value=f"{conf_val:.2%}"
                )
    else:
        st.warning("No Truck Cabins detected at the current confidence threshold.")

else:
    # Friendly reminder placeholder when no image is uploaded
    st.info("💡 Please upload an image above to begin detection.")

# Footer
st.markdown("---")
st.caption("Powered by YOLOv8 • Built with Streamlit")