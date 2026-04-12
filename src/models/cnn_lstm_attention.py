import torch
import torch.nn as nn
from torchvision import models


class CNNLSTMAttentionBinaryClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        pretrained=False,
        dropout=0.2,
        freeze_backbone=False,
    ):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        backbone_out_dim = 576

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=backbone_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # temporal attention
        self.attention_fc = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def extract_frame_features(self, x):
        """
        x: [B*T, C, H, W]
        return: [B*T, 576]
        """
        feat = self.feature_extractor(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        return feat

    def forward(self, clips, return_attention=False):
        """
        clips: [B, T, C, H, W]
        """
        b, t, c, h, w = clips.shape
        x = clips.view(b * t, c, h, w)

        frame_feats = self.extract_frame_features(x)   # [B*T, F]
        frame_feats = frame_feats.view(b, t, -1)       # [B, T, F]

        lstm_out, _ = self.lstm(frame_feats)           # [B, T, H]

        # attention weights over time
        attn_logits = self.attention_fc(lstm_out).squeeze(-1)   # [B, T]
        attn_weights = torch.softmax(attn_logits, dim=1)        # [B, T]

        # weighted temporal pooling
        weighted_feat = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, H]

        logits = self.classifier(weighted_feat)

        if return_attention:
            return logits, attn_weights

        return logits