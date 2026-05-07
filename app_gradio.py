import io
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import gradio as gr

# ----- AlphabetCNN (copy of your model class) -----
class AlphabetCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(AlphabetCNN, self).__init__()
        self.conv1a = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.25)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.bn1(self.conv1b(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.relu(self.conv2a(x))
        x = self.relu(self.bn2(self.conv2b(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.bn3(self.conv3b(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x

# Encoder mapping (A=0,...,Z=25)
ENCODER = {
    0: 'A',1: 'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
    20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'
}

# Load model
DEVICE = torch.device('cpu')
model = AlphabetCNN(num_classes=26)
model.load_state_dict(torch.load("models/letter.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocess: take uploaded image, convert to 50x50 grayscale, normalize
def preprocess_image(img: Image.Image):
    img = img.convert("L")  # grayscale
    img = ImageOps.invert(img)  # invert if your training used white on black (adjust if needed)
    img = img.resize((50,50))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.reshape(1, 1, 50, 50)
    return torch.from_numpy(arr)

def predict(image):
    try:
        tensor = preprocess_image(image)
        with torch.no_grad():
            out = model(tensor)
            idx = int(torch.argmax(out, dim=1).item())
            return ENCODER.get(idx, '?')
    except Exception as e:
        return f"Error: {e}"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Draw a letter (50x50 will be used)"),
    outputs=gr.Text(label="Predicted Letter"),
    title="Alphabet Recognition",
    description="Upload or draw a letter; model predicts A-Z."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)