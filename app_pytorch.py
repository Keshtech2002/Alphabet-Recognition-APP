import numpy as np
import torch
import torch.nn as nn
from bidict import bidict
from flask import (
    Flask, render_template, request,
    redirect, url_for, session
)
from random import choice
import json
import os

# Define the PyTorch model architecture
# MATCHES: pytorch_augmented_colab.ipynb (your actual trained model)
class AlphabetCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(AlphabetCNN, self).__init__()
        
        # Block 1: 1 -> 64 channels
        self.conv1a = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        
        # Block 2: 64 -> 128 channels
        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)
        
        # Block 3: 128 -> 256 channels
        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.25)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1a(x))
        x = self.relu(self.bn1(self.conv1b(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = self.relu(self.conv2a(x))
        x = self.relu(self.bn2(self.conv2b(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = self.relu(self.conv3a(x))
        x = self.relu(self.bn3(self.conv3b(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Global pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'alphabet_recognition'

# Device setup
DEVICE = torch.device('cpu')  # Use CPU for production

# Load encoder (0-indexed: A=0, B=1, ..., Z=25)
ENCODER = bidict({
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25
})

# Load model at startup
def load_model():
    """Load the trained PyTorch model"""
    model = AlphabetCNN(num_classes=26)  # 26 classes (A-Z)
    model_path = 'models/letter.pth'
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()  # Set to evaluation mode
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Model file not found: {model_path}")
        return None

# Load model when app starts
model = load_model()

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

# /add-data GET POST
@app.route('/add-data', methods=['GET'])
def add_data_get():
    message = session.get('message', '')
    letter = choice(list(ENCODER.keys()))
    return render_template('addData.html', letter=letter, message=message)

@app.route('/add-data', methods=['POST'])
def add_data_post():
    try:
        label = request.form['letter']

        labels = np.load('data/labels.npy', allow_pickle=True)
        labels = np.append(labels, label)
        np.save('data/labels.npy', labels)

        pixels = request.form['pixels']
        pixels = pixels.split(',')
        img = np.array(pixels).astype(float).reshape(1, 50, 50)

        imgs = np.load('data/images.npy')
        imgs = np.vstack([imgs, img])
        np.save('data/images.npy', imgs)

        session['message'] = f'"{label}" added to the training dataset'
        return redirect(url_for('add_data_get'))
    
    except Exception as e:
        print(f"Error in add_data_post: {e}")
        session['message'] = f'Error adding data: {str(e)}'
        return redirect(url_for('add_data_get'))

# /practice GET POST
@app.route('/practice', methods=['GET'])
def practice_get():
    letter = choice(list(ENCODER.keys()))
    return render_template("practice.html", letter=letter, correct='')

@app.route('/practice', methods=['POST'])
def practice_post():
    try:
        if model is None:
            return render_template('error.html')
        
        letter = request.form['letter']
        pixels = request.form['pixels']
        pixels = pixels.split(',')
        
        # Convert to numpy array and reshape
        img = np.array(pixels, dtype=np.float32).reshape(1, 50, 50) / 255.0
        
        # Convert to PyTorch tensor
        # Add channel dimension and convert to (N, C, H, W) format
        img_tensor = torch.from_numpy(img).unsqueeze(1).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
        
        # Convert prediction index back to letter
        try:
            pred_letter = ENCODER.inverse[pred_idx]
        except KeyError:
            # If index is out of range, treat as incorrect
            pred_letter = '?'
        
        # Check if correct
        correct = 'yes' if pred_letter == letter else 'no'
        
        # Get next letter
        next_letter = choice(list(ENCODER.keys()))
        
        return render_template("practice.html", letter=next_letter, correct=correct)

    except Exception as e:
        print(f"Error in practice_post: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
