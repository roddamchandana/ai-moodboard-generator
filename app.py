import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Extract dominant color palette using KMeans
def extract_palette(image, n_colors=6):
    img = image.convert('RGB')
    img = img.resize((200, 200))  # Resize for faster processing
    img_data = np.array(img).reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(img_data)

    colors = kmeans.cluster_centers_.astype(int)

    # Plot palette
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, color in enumerate(colors):
        ax.bar(i, 1, color=np.array(color)/255)
    ax.axis('off')
    st.pyplot(fig)

    return colors

# --- Streamlit App UI ---
st.set_page_config(page_title="AI Moodboard Generator", layout="centered")
st.title("🧠🎨 AI Moodboard Generator for Fashion Concepts")

# File uploader
uploaded_file = st.file_uploader("📸 Upload your fashion image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="✨ Uploaded Image", use_column_width=True)
    
    st.subheader("🎨 Extracted Color Palette")
    palette = extract_palette(image, n_colors=6)
    
    st.write("🧾 RGB Values:")
    st.write(palette)
