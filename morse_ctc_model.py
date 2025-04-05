import torch.nn as nn

class MorseCTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,   # для CTC: [T, B, F]
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # BiLSTM → умножаем на 2

    def forward(self, x):
        # x: [B, T, F] → [B, F, T] для Conv1d
        x = x.transpose(1, 2)

        x = self.cnn(x)         # [B, C, T]
        x = x.transpose(1, 2)   # [B, T, C]
        x = x.transpose(0, 1)   # [T, B, C] — нужно для LSTM и CTC

        x, _ = self.lstm(x)     # [T, B, hidden*2]
        logits = self.fc(x)     # [T, B, output_dim]

        return logits