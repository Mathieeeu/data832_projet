import torch
import torch.nn as nn

# Définition du modèle de réseau de neurones récurrent (RNN)
class RNNClassifier(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 dropout: float
                ) -> None:
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.batchnorm(out)
        out = self.fc(out)
        return out

# modèle de réseau de neurones dense (qui marche un peu aussi)
class DenseClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_classes: int,
                 dropout: float
                 ) -> None:
        super(DenseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.fc1(x))
        out = self.batchnorm1(out)
        out = self.dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.batchnorm2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
    
# modèle de réseau de neurones convolutif (CNN)
class CNNClassifier(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_classes: int,
                 dropout: float
                ) -> None:
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_size//2, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_size * input_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1)  # Transforme (batch, features) -> (batch, channels, sequence_length)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x