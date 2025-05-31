import torch
import torch.nn as nn
import torch.nn.functional as F

class ESTAN(nn.Module):
    """
    ESTAN: Ensemble of Spatial and Channel-wise Attention Networks for breast cancer detection.
    Adapted here for cervical cancer detection.
    """

    def __init__(self, num_classes=1):
        super(ESTAN, self).__init__()

        # Spatial Attention Branch
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Channel Attention Branch
        self.channel_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers after concatenation
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14 * 2, 512),  # times 2 because of concatenation
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        spatial_out = self.spatial_branch(x)
        channel_out = self.channel_branch(x)

        # Flatten spatial and channel outputs
        spatial_out_flat = spatial_out.view(spatial_out.size(0), -1)
        channel_out_flat = channel_out.view(channel_out.size(0), -1)

        # Concatenate features
        combined = torch.cat((spatial_out_flat, channel_out_flat), dim=1)

        out = self.classifier(combined)
        return out
