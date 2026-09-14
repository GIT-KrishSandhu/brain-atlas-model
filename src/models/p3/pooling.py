import torch
import torch.nn as nn


class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, query_num: int, num_classes: int, num_heads: int = 4, dropout: float = 0.0):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num

        # Learnable query tensor: shape [query_num, embed_dim]
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))

        # Multihead attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        # LayerNorm and Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Linear classifier: maps [query_num * embed_dim] to [num_classes]
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Feature map [B, D, H, W, L] or [B, D, N]
        Returns:
            Logits [B, num_classes]
        """
        batch_size = x.shape[0]

        if x.dim() == 5:
            x = x.flatten(2)  # [B, D, H*W*L]

        # Dimension permutation: [B, D, L] -> [L, B, D] (seq_len, batch, embed_dim)
        x = x.permute(2, 0, 1)

        # Expand query: [query_num, embed_dim] -> [query_num, B, D]
        query = self.class_query.unsqueeze(1).repeat(1, batch_size, 1)

        attended, attention_weights = self.cross_attention(
            query=query,
            key=x,
            value=x
        )

        # attended: [query_num, B, D]
        attended = self.norm(attended)
        attended = self.dropout(attended)

        # Permute: [query_num, B, D] -> [B, query_num, D]
        attended_permuted = attended.permute(1, 0, 2)

        # Flatten query and feature dimensions: [B, query_num * D]
        attended_flatten = attended_permuted.flatten(1)

        # Classify
        logits = self.classifier(attended_flatten)
        return logits
