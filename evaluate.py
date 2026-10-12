
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.multi_scale_model import MultiScaleDefectDetector


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained surface defect detection model."
    )
    parser.add_argument("--data-dir", default="dataset/val")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Run train.py first."
        )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Validation dataset directory not found: {data_dir}"
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

    dataset = datasets.ImageFolder(
        str(data_dir), transform=transform
    )

    if len(dataset) == 0:
        raise ValueError("The validation dataset is empty.")

    if dataset.classes != class_names:
        raise ValueError(
            f"Checkpoint classes {class_names} do not match "
            f"dataset classes {dataset.classes}."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MultiScaleDefectDetector(
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    report_text = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report_text}",
        encoding="utf-8",
    )

    with (output_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "accuracy": accuracy,
                "class_names": class_names,
                "metrics": report_dict,
            },
            file,
            indent=2,
        )

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Surface Defect Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("Device:", device)
    print(f"Validation images: {len(dataset)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:\n")
    print(report_text)
    print("Evaluation files saved to:", output_dir)


if __name__ == "__main__":
    main()
