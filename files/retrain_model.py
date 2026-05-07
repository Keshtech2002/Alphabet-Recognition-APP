"""
Retrain alphabet recognition model with TensorFlow 2.21.0
This script replicates the training from the Colab notebook locally
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from bidict import bidict
from sklearn.utils import shuffle
import os

# Configuration
BATCH_SIZE = 16
EPOCHS = 20
DATA_DIR = 'data'
MODEL_OUTPUT = 'models/letter.h5'

# Encoder
ENCODER = bidict({
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18,
    'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
    'Y': 25, 'Z': 26
})

def load_data():
    """Load training data from numpy files"""
    print("Loading data...")
    
    # Load labels and convert to numeric format
    labels = np.load(os.path.join(DATA_DIR, 'labels.npy'), allow_pickle=True)
    labels = np.array([ENCODER[x] for x in labels])
    
    # Load and preprocess images
    imgs = np.load(os.path.join(DATA_DIR, 'images.npy'))
    imgs = imgs.astype('float32') / 255.0
    
    # Add channel dimension if not present
    if len(imgs.shape) == 3:  # (N, 50, 50) -> (N, 50, 50, 1)
        imgs = np.expand_dims(imgs, -1)
    
    print(f"Data shapes - Images: {imgs.shape}, Labels: {labels.shape}")
    
    # Shuffle data
    labels, imgs = shuffle(labels, imgs, random_state=42)
    
    # Split into train/test (75/25)
    split = int(len(labels) * 0.75)
    labels_train, labels_test = labels[:split], labels[split:]
    imgs_train, imgs_test = imgs[:split], imgs[split:]
    
    print(f"Train set: {imgs_train.shape[0]} samples")
    print(f"Test set: {imgs_test.shape[0]} samples")
    
    return (imgs_train, labels_train), (imgs_test, labels_test)

def build_model():
    """Build the same model architecture from the notebook"""
    print("Building model...")
    
    model = keras.Sequential([
        keras.Input(shape=(50, 50, 1)),
        layers.Conv2D(256, kernel_size=5, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv2D(512, kernel_size=5, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv2D(1024, kernel_size=5, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(len(ENCODER) + 1, activation='softmax')
    ])
    
    return model

def train_model(model, train_data, test_data):
    """Train the model"""
    print("Compiling model...")
    
    # Compile
    optimizer = keras.optimizers.Adam()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True
    )
    
    # Training
    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(
        train_data[0], train_data[1],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(test_data[0], test_data[1]),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, test_data):
    """Evaluate model on test set"""
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")

def save_model(model, output_path):
    """Save model in both H5 and Keras native formats"""
    print(f"\nSaving model to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save in H5 format (for compatibility with your Flask app)
    model.save(output_path)
    print(f"✓ Model saved to {output_path}")
    
    # Also save in native Keras format as backup
    keras_path = output_path.replace('.h5', '.keras')
    model.save(keras_path)
    print(f"✓ Model also saved to {keras_path}")

def main():
    """Main training pipeline"""
    print("=" * 50)
    print("Alphabet Recognition Model Retraining")
    print("=" * 50)
    
    # Load data
    train_data, test_data = load_data()
    
    # Build model
    model = build_model()
    print(f"Model created with {model.count_params()} parameters")
    
    # Train model
    history = train_model(model, train_data, test_data)
    
    # Evaluate model
    evaluate_model(model, test_data)
    
    # Save model
    save_model(model, MODEL_OUTPUT)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    return model, history

if __name__ == '__main__':
    model, history = main()
