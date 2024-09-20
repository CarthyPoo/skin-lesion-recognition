import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False)

# Function to preprocess the image and extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    # Extract features using the VGG16 model
    features = model.predict(img_data)
    
    # Flatten the extracted features to a 1D vector
    features = np.reshape(features, (features.shape[0], -1))
    return features

# Function to calculate cosine similarity between two images
def calculate_similarity(img1_path, img2_path, model):
    # Extract features for both images
    features_img1 = extract_features(img1_path, model)
    features_img2 = extract_features(img2_path, model)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(features_img1, features_img2)
    return similarity[0][0]

# Paths to images
img1_path = 'image1.jpg'
img2_path = 'image2.jpg'

# Calculate similarity
similarity_score = calculate_similarity(img1_path, img2_path, base_model)

print(f"Similarity between the two images: {similarity_score:.4f}")
