import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import RNNClassifier, DenseClassifier, CNNClassifier


def train_model(
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        label_encoder,
        model_type: str = 'cnn',
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.001,
        learning_rate: float = 0.003,
        num_epochs: int = 500,
        early_stopping: bool = True,
        patience: int = 10
    ) -> tuple:
    """
    Entraîne un modèle de classification sur les données fournies et retourne les pertes d'entraînement, de test et le modèle entraîné.
    """

    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    # Initialisation du modèle (à éventuellement changer si besoin)
    if model_type == 'rnn':
        model = RNNClassifier(input_size, hidden_size, num_layers, num_classes, dropout)
    elif model_type == 'dense':
        model = DenseClassifier(input_size, hidden_size, num_classes, dropout)
    elif model_type == 'cnn':
        model = CNNClassifier(input_size, hidden_size, num_classes)
    else:
        raise ValueError("Type de modèle non reconnu. Choisissez 'rnn' , 'cnn' ou 'dense'.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_loss = float('inf')
    epoches_without_improvement = 0
    train_losses = []
    test_losses = []

    # Entrainement
    for epoch in tqdm(range(num_epochs), desc="Entrainement..."):
        model.train()
        if model_type == 'rnn':
            outputs = model(X_train.unsqueeze(1))
        else:
            outputs = model(X_train) # pas besoin d'unsqueeze(1) (= redimensionner) pour le modèle dense
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # calcul de la perte sur l'ensemble de test
        model.eval()
        with torch.no_grad():
            if model_type == 'rnn':
                test_outputs = model(X_test.unsqueeze(1))
            else:
                test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

            # arret anticipé de l'entrainement si aucune amélioration pendant un certain nombre d'itérations (patience)
            if early_stopping:
                if test_loss < best_loss:
                    best_loss = test_loss
                    epoches_without_improvement = 0
                else:
                    epoches_without_improvement += 1
                if epoches_without_improvement >= patience:
                    break
    if epoches_without_improvement >= patience:
        print(f"ò.ó - fin de l'entrainement parce que pas d'amélioration depuis {patience} epochs\n")

    return model, train_losses, test_losses

def save_model(model, model_name: str = 'model') -> None:
    """
    Sauvegarde le modèle entraîné à l'emplacement spécifié.
    """
    torch.save(model.state_dict(), f"models/{model_name}.pth")

def load_model(model, model_name: str = 'model') -> nn.Module:
    """
    Charge le modèle à partir du fichier spécifié.
    """
    model.load_state_dict(torch.load(f"models/{model_name}.pth"))
    model.eval()
    return model