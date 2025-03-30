import os
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Chemin vers le répertoire où se trouve le dataset 
dataset_dir = 'D:/data/projet-data832' # A EDITER SI BESOIN

# Chemins à ne pas toucher !!!
data_dir = os.path.join(dataset_dir, 'genres_original') # répertoire contenant les fichiers .wav
features_file = './data/features-complet.csv' # fichier CSV pour stocker les caractéristiques extraites (et économiser des HEURES)

# Extraction des caractéristiques du fichier audio
def extract_features(file_path):
    # print(f"Extraction des caractéristiques de {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    return np.hstack((
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        # tempo,    
        # np.mean(tonnetz, axis=1),
        # np.mean(mel_spectrogram, axis=1)
    ))


# ici si les caractéristiques ont déjà été extraites (en csv) on les charge, sinon on les extrait
if os.path.exists(features_file):
    print("Chargement des caractéristiques à partir du fichier CSV...")
    df = pd.read_csv(features_file)
else:
    print("Fichier CSV non trouvé. Extraction des caractéristiques...")
    features = []
    labels = []

    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                all_files.append((root, file))

    for root, file in tqdm(all_files, desc="Extracting features"):
        file_path = os.path.join(root, file)
        genre = os.path.basename(root)
        feature_vector = extract_features(file_path)
        features.append(feature_vector)
        labels.append(genre)

    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(features_file, index=False)

# Prétraitement
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['label', 'label_encoded']))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['label_encoded'], test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)  # Convertir en tableau NumPy avec .values
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)    # Convertir en tableau NumPy avec .values


# # Distribution des caratéristiques audio avec courbe KDE (Kernel Density Estimation)
# audio_features = df.drop(columns=['label'])
# #Récupération des noms des colonnes

# feature_names = (pd.read_csv("./data/features_30_sec.csv")).columns.tolist()

# plt.figure(figsize=(10, 7))
# for i, col in enumerate(audio_features.columns[2:12]):  # Limité à 10
#     plt.subplot(5, 2, i + 1)
#     sns.histplot(audio_features[col], bins=30, kde=True)
#     plt.title(f'Caractéristique {col} : {feature_names[i+2]}')
#     plt.xlabel('Valeur')
#     plt.ylabel('Densité')
# plt.suptitle('Distribution des caractéristiques audio avec courbe KDE', fontsize=16)
# plt.tight_layout()
# plt.show()


def plot_pca_with_centers(X, y, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    
    # Ajouter les points moyens pour chaque genre
    centers = []
    for label in np.unique(y):
        mean_point = np.mean(X_pca[y == label], axis=0)
        centers.append(mean_point)

        plt.scatter(mean_point[0], mean_point[1], marker='x', s=200, c=[scatter.cmap(scatter.norm(label))], edgecolor='white', label=labels[label])

    cbar = plt.colorbar(scatter, ticks=range(len(labels)))
    cbar.set_label('Genres')
    cbar.set_ticks(np.arange(len(labels)))
    cbar.set_ticklabels(labels)
    
    plt.title("PCA - Réduction de Dimension avec Centres des Genres")
    plt.legend(loc='upper right')
    plt.show()

plot_pca_with_centers(X_test, y_test, label_encoder.classes_)


# Définition du modèle de réseau de neurones récurrent (RNN)
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.batchnorm(out)
        out = self.fc(out)
        return out

# modèle de réseau de neurones dense (qui marche un peu aussi)
class DenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super(DenseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
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
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_size//2, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_size * input_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(-1).permute(0, 2, 1)  # Transforme (batch, features) -> (batch, channels, sequence_length)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

##########################################################
# Hyperparamètres (à modifier si besoin)
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model_type = 'cnn' # 'rnn' ou 'dense' ou 'cnn'
hidden_size = 128
num_layers = 2
dropout = 0.001 # taux de dropout (càd le pourcentage de neurones à ignorer pendant l'entrainement pour éviter le surapprentissage)

num_epochs = 500 # maximum
learning_rate = 0.003

stop_early = True # arret anticipé de l'entrainement si aucune amélioration pendant un certain nombre d'itérations (patience)
patience = 10  # Patience pour l'arret anticipé 
##########################################################

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
        if stop_early:
            if test_loss < best_loss:
                best_loss = test_loss
                epoches_without_improvement = 0
            else:
                epoches_without_improvement += 1

            if epoches_without_improvement >= patience:
                print("\nça suffit rooh !! ò.ó\n(fin de l'entrainement parce que pas d'amélioration)")
                break

# visualisation de la courbe de pertes
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Perte d\'entrainement')
plt.plot(test_losses, label='Perte de test')
plt.title('Courbe de perte')
plt.xlabel('Epoch')
plt.ylabel('Perte')
plt.legend()
plt.show()

# Évaluation du modèle
model.eval()
with torch.no_grad():
    if model_type == 'rnn':
        outputs = model(X_test.unsqueeze(1))
    else:
        outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test, predicted)
    print("Précision:", accuracy)
    print("Report:\n", classification_report(y_test, predicted, target_names=label_encoder.classes_))

    conf_matrix = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matrice de confusion')
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.show()
    
