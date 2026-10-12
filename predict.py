
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.multi_scale_model import MultiScaleDefectDetector


def main():
    parser = argparse.ArgumentParser(
        description="Predict surface defects in a single image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the manufacturing image."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pth",
        help="Path to the trained model checkpoint."
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the model first using train.py."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

    model = MultiScaleDefectDetector(
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    print("\n--- Surface Defect Detection Result ---")
    print("Image:", image_path.name)
    print("Prediction:", predicted_class)
    print(f"Model confidence: {confidence:.2f}%")
    print("\nClass probabilities:")

    for index, class_name in enumerate(class_names):
        probability = probabilities[index].item() * 100
        print(f"{class_name}: {probability:.2f}%")


if __name__ == "__main__":
    main()
