import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Constants
IMAGE_SIZE = (150, 150)
CATEGORIES = ["covid-19", "no covid-19"]
MODEL_PATH = 'covid-classifier.h5'
TEST_IMAGE_PATH = "images/test/covid-1.jpeg"

def load_image(img_path):
    """Load and preprocess image"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.expand_dims(img, axis=[0, -1])
        img = img / 255.0
        return img
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap"""
    # Print shapes for debugging
    print(f"Input image shape: {img_array.shape}")
    
    # Create grad model
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        print(f"Conv output shape: {conv_output.shape}")
        print(f"Predictions shape: {predictions.shape}")
        
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]
    
    # Get gradients
    grads = tape.gradient(class_output, conv_output)
    print(f"Gradients shape: {grads.shape}")
    
    # Safe reduction accounting for actual dimensions
    if len(grads.shape) == 4:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    # Generate heatmap
    heatmap = tf.matmul(conv_output[0], pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(), class_idx.numpy()

def main():
    try:
        model = load_model(MODEL_PATH)
        img = load_image(TEST_IMAGE_PATH)
        
        # Get the last conv layer name
        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
        if not conv_layers:
            raise ValueError("No convolutional layers found in model")
        last_conv_layer = conv_layers[-1]
        
        # Generate and display heatmap
        heatmap, pred_class = make_gradcam_heatmap(img, model, last_conv_layer)
        
        # Visualize results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(img[0, :, :, 0], cmap='gray')
        plt.title('Original Image')
        
        plt.subplot(132)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        
        plt.subplot(133)
        heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
        plt.imshow(img[0, :, :, 0], cmap='gray')
        plt.imshow(heatmap_resized, alpha=0.4, cmap='jet')
        plt.title(f'Overlay\nPrediction: {CATEGORIES[pred_class]}')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()