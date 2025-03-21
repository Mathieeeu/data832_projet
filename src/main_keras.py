import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

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


# Affichage de la répartition des classes dans les ensembles d'entraînement et de test
train_counts = pd.Series(y_train).value_counts()
test_counts = pd.Series(y_test).value_counts()
classes = pd.unique(y)

plt.figure(figsize=(12, 6))
plt.bar(classes, train_counts[classes], label="Train", alpha=0.7)
plt.bar(classes, test_counts[classes], label="Test", alpha=0.7, bottom=train_counts[classes])
plt.xlabel("Genres (Labels)")
plt.ylabel("Nombre d'échantillons")
plt.title("Répartition des classes dans les ensembles d'entraînement et de test")
plt.legend()
#plt.show()

# Neurones
n_steps = 58
n_features = 1

model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, n_features), return_sequences=False))
model_lstm.add(Dense(units=len(lb.classes_), activation="softmax"))  
model_lstm.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Entraînement
y_pred = np.argmax(model_lstm.predict(X_test), axis=-1)
print("Prédictions : ", y_test)

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels = lb.classes_, yticklabels = lb.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_test, y_pred, target_names = lb.classes_))