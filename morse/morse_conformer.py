import torch
import torch.nn as nn
from torchaudio.models import Conformer

class MorseConformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        embedding_dim=80,
        num_heads=2,
        ffn_dim=160,
        num_layers=4,
        kernel_size=15,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

        self.conformer = Conformer(
            input_dim=embedding_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=kernel_size,
            dropout=dropout
        )

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)

        x, _ = self.conformer(x, lengths)
        x = self.fc(x)

        return x.transpose(0, 1)
