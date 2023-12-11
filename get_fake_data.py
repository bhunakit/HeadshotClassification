import requests
from PIL import Image
from io import BytesIO

url = "https://thispersondoesnotexist.com/"

for i in range(1000):
    response = requests.get(url)

    if response.status_code == 200:
        # Use BytesIO to create a file-like object from the binary content
        image_bytes = BytesIO(response.content)

        # Open the image using PIL
        image = Image.open(image_bytes)

        # Resize the image to 128x128
        resized_image = image.resize((128, 128))

        # Save the resized image
        resized_image.save(f"data_crossval/fake/f{i}.jpg")
        print(f"Image {i} downloaded and resized successfully.")

    else:
        print(f"Error downloading image: {response.status_code}")