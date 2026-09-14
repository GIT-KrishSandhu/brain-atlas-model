import torch.nn as nn
from src.models.p3.pooling import CrossAttentionPooling


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        query_num: int,
        num_classes: int,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        num_heads: int = 4
    ):
        super(ClassificationHead, self).__init__()

        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim=embed_dim,
                query_num=query_num,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(1),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )

    def forward(self, x):
        return self.pooling(x)
