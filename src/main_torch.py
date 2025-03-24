import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

class MusicGenreRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MusicGenreRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class MusicGenreLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MusicGenreLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        
        # Forward propagate LSTM
        _, (h_n, _) = self.lstm(x)

        # Decode the hidden state of the last time step
        out = self.fc(h_n[-1])

        return out

seed = 18

# Importation des données
filename = "./data/features_30_sec.csv"
data = pd.read_csv(filename)

# Extraction de caractéristiques spéctrales (MFCCs, chroma, spectrogrammes)
data = data.drop(['filename', 'length'], axis=1)
print(data.head())
print(data.shape) # (1000, 58)


# Utilisation de modèles adaptés aux données temporelles (ex. Réseaux de neurones)
lb = LabelEncoder()
data['label'] = lb.fit_transform(data['label'])
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

#Redimensionnement
X_train = np.expand_dims(X_train, axis=-1)  # (samples, 58, 1)
X_test = np.expand_dims(X_test, axis=-1)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy())
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy())


# Hyperparamètres
input_size = 1 # nombre de mfcc (58)
hidden_size = 128
num_layers = 3
num_classes = 10

# model = MusicGenreRNN(input_size, hidden_size, num_layers, num_classes).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model = MusicGenreLSTM(input_size, hidden_size, num_layers, num_classes).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5, amsgrad=True, eps=1e-8, betas=(0.9, 0.999))

num_epochs = 100

for epoch in range(num_epochs):    
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

    print(f'Accuracy: {accuracy_score(y_test, predicted)}')
    print(f'F1 Score: {f1_score(y_test, predicted, average="macro")}')
    print(f'Precision: {precision_score(y_test, predicted, average="macro")}')
    print(f'Recall: {recall_score(y_test, predicted, average="macro")}')
    print(f'Classification Report: \n {classification_report(y_test, predicted)}')

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

