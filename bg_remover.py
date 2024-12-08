import torch
from torchvision import transforms
from PIL import Image
import requests
import numpy as np
from io import BytesIO

# Step 1: Load the Image from URL
def load_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

# Step 2: Preprocessing for U-Net
def preprocess_image(image, size=(256, 256)):
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Step 3: Load a Pre-trained Segmentation Model (DeepLabV3 in this case)
def load_model():
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Step 4: Perform Inference
def segment_image(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)['out']  # Model output
        mask = output.argmax(dim=1).squeeze().cpu().numpy()  # Get segmentation mask
        return mask

# Step 5: Postprocess and Extract Object
def apply_mask(image, mask, threshold=1):
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)
    mask_resized = np.array(mask_resized) > threshold
    image_np = np.array(image)

    # Create RGBA image with transparency
    rgba_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    rgba_image[..., :3] = image_np  # Copy RGB channels
    rgba_image[..., 3] = mask_resized.astype(np.uint8) * 255  # Alpha channel based on mask
    return Image.fromarray(rgba_image)

# Input Image URL
image_url = "https://images.unsplash.com/photo-1469285994282-454ceb49e63c?q=80&w=2942&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# Load and Process the Image
image = load_image(image_url)
input_tensor = preprocess_image(image)

# Load Model and Perform Segmentation
model = load_model()
mask = segment_image(model, input_tensor)

# Apply Mask to Extract Object
result_image = apply_mask(image, mask)
result_image.show()  # Display the output
result_image.save("car_segmented.png")  # Save the processed image