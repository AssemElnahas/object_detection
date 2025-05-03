import streamlit as st  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from io import BytesIO

# Page Configuration
st.set_page_config(page_title="Object Detector", layout="wide")
st.title("Object Detector with Streamlit")
st.write("Upload an image to detect objects using a pre-trained YOLOv5 model.")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small


model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Detecting objects...")

    # Convert image to array
    image_np = np.array(image)

    # Perform object detection
    results = model(image_np)

    # Render results
    st.header("🎯 Detection Results")
    results.render()
    detected_image = Image.fromarray(results.ims[0])
    st.image(detected_image, caption="Detected Objects", use_column_width=True)

    # Display raw results
    st.subheader("📝 Raw Detection Results:")
    df = results.pandas().xyxy[0]
    st.dataframe(df)

    # 📊 Show bar chart of detected objects
    st.subheader("📊 Detected Object Counts")
    label_counts = df['name'].value_counts()
    st.bar_chart(label_counts)

    # 📥 Download button for the result image
    buf = BytesIO()
    detected_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("📥 Download Detected Image", byte_im, file_name="detected.png")

# Sidebar Resources
st.sidebar.header("Contact Me")
st.sidebar.markdown("---")
st.sidebar.markdown("📧 **Email:** [assem.elnahas15@gmail.com](mailto:assem.elnahas15@gmail.com)")
st.sidebar.markdown("🔗 **LinkedIn:** [Assem on LinkedIn](https://www.linkedin.com/in/assem-elnahas-28887429a/)")
st.sidebar.markdown("📸 **Instagram:** [@asem.elnahas](https://www.instagram.com/asem.elnahas/)")
st.sidebar.markdown("💻 **GitHub:** [AssemElnahas](https://github.com/AssemElnahas)")
st.sidebar.markdown("📞 **Phone:** [+201060133668]")

# Footer
footer = """
<style>
.footer {
    margin-top: 500px;
    width:100%;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #3a6c;
    color: #31333f;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
}
</style>
<div class="footer">
    Made with ❤️ by Assem | © 2025
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
