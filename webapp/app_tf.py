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
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('../best_transfer_learning_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for incoming images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
            img = img.resize((224, 224))
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
