
import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """Extract features using different convolution kernel sizes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 3
        remaining_channels = out_channels - 2 * branch_channels

        self.branch_1 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels,
                kernel_size=1, padding=0
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_5 = nn.Sequential(
            nn.Conv2d(
                in_channels, remaining_channels,
                kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(remaining_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features_1 = self.branch_1(x)
        features_3 = self.branch_3(x)
        features_5 = self.branch_5(x)

        combined = torch.cat(
            [features_1, features_3, features_5],
            dim=1
        )

        return self.fusion(combined)


class MultiScaleDefectDetector(nn.Module):
    """Classify images as non-defective or defective."""

    def __init__(self, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3,
                stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage_1 = MultiScaleBlock(32, 64)
        self.pool_1 = nn.MaxPool2d(2)

        self.stage_2 = MultiScaleBlock(64, 128)
        self.pool_2 = nn.MaxPool2d(2)

        self.stage_3 = MultiScaleBlock(128, 192)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.stage_1(x)
        x = self.pool_1(x)

        x = self.stage_2(x)
        x = self.pool_2(x)

        x = self.stage_3(x)

        x = self.global_pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = MultiScaleDefectDetector(num_classes=2)
    sample_images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        predictions = model(sample_images)

    print("Model:", model.__class__.__name__)
    print("Input shape:", sample_images.shape)
    print("Output shape:", predictions.shape)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))
