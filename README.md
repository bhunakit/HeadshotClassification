# Headshot Classification

## Data Source
- Real headshot image dataset: https://github.com/NVlabs/ffhq-dataset/ --> Real Faces from Flickr 
- Fake headshot image dataset: https://thispersondoesnotexist.com/ --> Generated Faces from StyleGAN2 (Karras et al.)
  
<div style="display: flex; justify-content: space-around; align-items: center; margin-top: 20px;">
    <figure style="text-align: center; margin-right: 20px;">
        <img src="https://github.com/bhunakit/HeadshotClassification/blob/cfc9368fddd7c89138c40bbbeb3d2120f06a78e1/data/fake/f1003.jpg" alt="Real Image" width="300"/>
        <figcaption>Real Image</figcaption>
    </figure>
    <figure style="text-align: center; margin-left: 20px;">
        <img src="https://github.com/bhunakit/HeadshotClassification/assets/63712938/b2421bd5-ba31-4dfd-b01c-985818327973" alt="Generated Image" width="300"/>
        <figcaption>Generated Image</figcaption>
    </figure>
</div>

## Using Convolution Neural Network
- **Best Hyperparameters:**
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 10

- **Accuracy: 97.60%**

## Using Transfer Learning from ResNet18
- **Best Hyperparameters:**
  - Learning Rate: 0.005
  - Batch Size: 16
  - Epochs: 10

- **Accuracy: 99.30%**
