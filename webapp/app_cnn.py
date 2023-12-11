from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

app = Flask(__name__)

# Load the best model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the best model
model = CNN()
model.load_state_dict(torch.load('../best_CNN_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for incoming images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        if file:
            # Save the uploaded image
            upload_path = 'static/uploads/'
            os.makedirs(upload_path, exist_ok=True)
            image_path = os.path.join(upload_path, 'uploaded_image.jpg')

            # Open and preprocess the image
            img = Image.open(file).convert("RGB")
            img = img.resize((128, 128))
            img_tensor = transform(img).unsqueeze(0)

            img.save(image_path)

            # Make the prediction
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)

            # Get the class label (replace with your own class labels)
            class_labels = ['Fake', 'Real']
            prediction_label = class_labels[predicted.item()]
            print("Image Path:", image_path)
            return render_template('index.html', image_path=image_path, prediction=prediction_label)
    
    # Render the initial page
    return render_template('index.html', image_path="", prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
