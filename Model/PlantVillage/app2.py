# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os

# # --- CONFIGURATION ---
# MODEL_PATH = "plant_disease_model.pth"
# # IMPORTANT: This path must point to your PlantVillage dataset folder
# DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation" 
# NUM_CLASSES = 39 # From your training output

# # --- HELPER FUNCTIONS ---

# def get_class_names(data_dir):
#     """Gets the class names from the dataset directory in the same order as PyTorch's ImageFolder."""
#     # Check if the directory exists
#     if not os.path.isdir(data_dir):
#         st.error(f"Dataset directory not found at: {data_dir}")
#         st.stop()
    
#     # The class names are the names of the subdirectories, sorted alphabetically
#     class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
#     return class_names

# # --- MODEL LOADING ---
# @st.cache_resource
# def load_model(model_path, num_classes):
#     """Loads the pre-trained EfficientNet model with a custom classifier."""
#     # Instantiate the model architecture
#     model = models.efficientnet_b0(weights=None) # We don't need pre-trained weights, we have our own
    
#     # Replace the final layer
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    
#     # Load the trained weights (state_dict)
#     # Use map_location to ensure it works on CPU if CUDA isn't available for inference
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
#     # Set the model to evaluation mode
#     model.eval()
#     return model

# # --- PREDICTION FUNCTION ---
# def predict(image_data, model, class_names):
#     """Takes an image, runs it through the model, and returns the prediction."""
#     # Define the same transformations as the validation set
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     # Preprocess the image
#     image = Image.open(image_data).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
#     # Make prediction
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         # Get probabilities
#         probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
#         # Get the top prediction
#         _, predicted_idx = torch.max(outputs, 1)
        
#     prediction = class_names[predicted_idx.item()]
#     confidence = probabilities[predicted_idx.item()].item()
    
#     return prediction, confidence

# # --- STREAMLIT APP LAYOUT ---

# st.title("🌿 Plant Disease Detector")
# st.write("Upload an image of a plant leaf to classify its health status.")

# # Load class names and model
# class_names = get_class_names(DATA_DIR)
# model = load_model(MODEL_PATH, NUM_CLASSES)

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#     st.write("")
    
#     # Make a prediction when the button is clicked
#     if st.button("Classify Image"):
#         with st.spinner("Analyzing the leaf..."):
#             prediction, confidence = predict(uploaded_file, model, class_names)
#             st.success(f"**Prediction:** {prediction.replace('___', ' ')}")
#             st.info(f"**Confidence:** {confidence * 100:.2f}%")

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- CONFIGURATION ---
MODEL_URL = "https://drive.google.com/uc?id=1YvUoajFYHDPHckBmE0Xl1znHi89ZP2pU"
MODEL_PATH = "plant_disease_model.pth"
CLASSES_FILE = "classes.txt"
NUM_CLASSES = 39

# --- HELPER FUNCTIONS ---

def download_model_from_drive():
    """Download model from Google Drive if not present"""
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.error("Please check your internet connection and Google Drive link.")
            st.stop()
    return True

def load_class_names():
    """Load class names from classes.txt file"""
    if not os.path.exists(CLASSES_FILE):
        st.error(f"❌ Classes file not found: {CLASSES_FILE}")
        st.error("Please ensure classes.txt is in the same directory as this app.")
        st.stop()
    
    with open(CLASSES_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    if len(class_names) != NUM_CLASSES:
        st.warning(f"⚠️ Expected {NUM_CLASSES} classes, found {len(class_names)}")
    
    return class_names

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pre-trained EfficientNet model with custom classifier"""
    # Download model first
    download_model_from_drive()
    
    # Load model architecture
    model = models.efficientnet_b0(weights=None)
    
    # Replace the final layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    
    # Set to evaluation mode
    model.eval()
    return model

# --- PREDICTION FUNCTION ---
def predict_disease(image_data, model, class_names):
    """Predict plant disease from uploaded image"""
    # Define the same transformations as used in training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    image = Image.open(image_data).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted_idx = torch.max(outputs, 1)
    
    prediction = class_names[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item()
    
    return prediction, confidence

# --- STREAMLIT APP LAYOUT ---

# Configure the page
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .upload-section {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f5f5f5;
    }
    .result-card {
        background-color: #E8F5E8;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌿 Plant Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">PlantVillage Disease Classification</div>', unsafe_allow_html=True)

# Load model and class names
try:
    model = load_model()
    class_names = load_class_names()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader section
    st.markdown("### 📤 Upload Plant Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a plant leaf...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)
        
        # Prediction button
        if st.button("🔍 Analyze Disease", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing plant health..."):
                prediction, confidence = predict_disease(uploaded_file, model, class_names)
            
            # Display results
            st.markdown("### 🎯 Analysis Results")
            
            # Create result card
            st.markdown(f"""
            <div class="result-card">
                <h4>🔬 Disease Detection</h4>
                <p><strong>Disease:</strong> {prediction.replace('___', ' - ')}</p>
                <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Health status
            if "healthy" in prediction.lower():
                st.success("✅ **Plant is Healthy!**")
                st.balloons()
            else:
                st.warning("⚠️ **Disease Detected!**")
                st.info("💡 Consider consulting agricultural experts for treatment recommendations.")

with col2:
    # Information sidebar
    st.markdown("### ℹ️ How to Use")
    st.info("""
    1. **Upload** a clear image of a plant leaf
    2. **Click** 'Analyze Disease' 
    3. **View** the prediction results
    4. **Check** confidence level
    """)
    
    st.markdown("### 📋 Tips for Best Results")
    st.success("""
    • Use well-lit, clear images
    • Ensure leaf fills most of the frame
    • Avoid blurry or dark photos
    • Multiple angles can help accuracy
    """)
    
    st.markdown("### 📊 Model Information")
    st.info(f"""
    • **Classes:** {len(class_names)} diseases
    • **Architecture:** EfficientNet-B0
    • **Dataset:** PlantVillage
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌱 <strong>Plant Disease Detection Project</strong></p>
    <p>Plant Disease Detection using Deep Learning</p>
</div>
""", unsafe_allow_html=True)
