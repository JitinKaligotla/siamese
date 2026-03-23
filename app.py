import streamlit as st
import numpy as np
import cv2
from PIL import Image
from inference import load_model, predict

st.set_page_config(layout="wide")
st.title("🔥 Change Detection System")

# Load model once
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# ---------- Similarity Check ----------
def check_similarity(img1, img2, threshold=0.3):
    img1_r = cv2.resize(img1, (256, 256))
    img2_r = cv2.resize(img2, (256, 256))

    diff = np.mean(np.abs(img1_r.astype("float32") - img2_r.astype("float32"))) / 255.0
    return diff, diff < threshold


# ---------- Draw Boxes ----------
def draw_boxes(image, mask, min_area=50):
    mask = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = image.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return img_copy


# ---------- Area Percentage ----------
def calculate_area(mask):
    mask_bin = (mask > 0.5).astype(np.uint8)
    change_pixels = np.sum(mask_bin)
    total_pixels = mask_bin.size

    percentage = (change_pixels / total_pixels) * 100
    return percentage


# ---------- Upload ----------
col1, col2 = st.columns(2)

with col1:
    pre_file = st.file_uploader("Upload Pre Image", type=["jpg", "png"])

with col2:
    post_file = st.file_uploader("Upload Post Image", type=["jpg", "png"])


# ---------- Main Flow ----------
if pre_file and post_file:

    pre_img = Image.open(pre_file).convert("RGB")
    post_img = Image.open(post_file).convert("RGB")

    pre_np = np.array(pre_img)
    post_np = np.array(post_img)

    st.subheader("Input Images")
    c1, c2 = st.columns(2)
    with c1:
        st.image(pre_np, caption="Pre Image")
    with c2:
        st.image(post_np, caption="Post Image")

    # 🔍 Similarity check
    diff, similar = check_similarity(pre_np, post_np)

    if not similar:
        st.error(f"⚠️ Images are very different (diff={diff:.2f})")
    else:
        st.success(f"✅ Images are similar (diff={diff:.2f})")

    # Predict
    if st.button("🚀 Detect Changes"):

        mask = predict(model, pre_np, post_np)

        # Binary mask
        mask_bin = (mask > 0.5).astype(np.uint8)

        # Resize mask to original size
        mask_resized = cv2.resize(mask_bin, (post_np.shape[1], post_np.shape[0]))

        # Boxes
        boxed = draw_boxes(post_np, mask_resized)

        # Area %
        area = calculate_area(mask)

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(mask_bin * 255, caption="Binary Mask")

        with col2:
            st.image(boxed, caption="Detected Changes (Boxes)")

        with col3:
            st.metric(label="Change Area (%)", value=f"{area:.2f}%")