import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        # Load pre-trained model (e.g., ResNet50, OpenReID, etc.)
        self.model = models.resnet50(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode

    def extract(self, frame, bbox):
        """Extract ReID feature from the frame for a given bounding box."""
        x1, y1, w, h = bbox
        crop_img = frame[y1:y1+h, x1:x1+w]
        crop_img = Image.fromarray(crop_img)
        
        # Preprocess image for the model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        crop_img = transform(crop_img).unsqueeze(0)
        
        with torch.no_grad():
            feature = self.model(crop_img)
        return feature.numpy().flatten()  # Flatten to 1D vector
