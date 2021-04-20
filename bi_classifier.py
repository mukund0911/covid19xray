import os
import cv2 
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 150
CATEGORIES = ["covid-19", "no covid-19"]
DATA_DIR = "dataset"
MODEL_NAME = "covid_classifier.h5"

class CovidXrayClassifier:
    """
    Binary classifier for COVID-19 detection from chest X-rays.
    
    Attributes:
        training_data: List of processed training images and labels
        model: Trained CNN model
    """
    
    def __init__(self) -> None:
        """Initialize classifier with empty training data and model."""
        self.training_data: List = []
        self.model: Sequential = None
        
    def create_training_data(self) -> None:
        """
        Load and preprocess training images from dataset directory.
        
        Raises:
            FileNotFoundError: If dataset directory or images not found
        """
        try:
            for category in CATEGORIES:
                path = os.path.join(DATA_DIR, category)
                class_num = CATEGORIES.index(category)
                
                for img_file in os.listdir(path):
                    img_path = os.path.join(path, img_file)
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    self.training_data.append([new_array, class_num])
                    
            logger.info("Training data created successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def preprocess_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess training data into features and labels.
        
        Returns:
            Tuple containing features X and labels y as numpy arrays
        """
        np.random.shuffle(self.training_data)
        
        X = []
        y = []
        
        for features, label in self.training_data:
            X.append(features)
            y.append(label)
            
        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        X = X / 255.0  # Normalize
        y = np.array(y)
        
        logger.info("Image preprocessing complete")
        return X, y
    
    def build_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build and train CNN model on preprocessed data.
        
        Args:
            X: Training features
            y: Training labels
        """
        self.model = Sequential([
            Conv2D(64, (3,3), input_shape=X.shape[1:]),
            Activation("relu"),
            MaxPooling2D(pool_size=(2,2)),
            
            Conv2D(64, (3,3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2,2)),
            
            Flatten(),
            Dense(64),
            Activation("relu"),
            
            Dense(1),
            Activation('sigmoid')
        ])
        
        self.model.compile(loss="binary_crossentropy",
                         optimizer="adam",
                         metrics=['accuracy'])
                         
        self.model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
        self.model.save(MODEL_NAME)
        logger.info(f"Model trained and saved as {MODEL_NAME}")

def main():
    """Main execution function."""
    try:
        classifier = CovidXrayClassifier()
        classifier.create_training_data()
        X, y = classifier.preprocess_images() 
        classifier.build_model(X, y)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        
if __name__ == "__main__":
    main()