import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Fonction pour prétraiter les données audio extraites.
    Elle encode les étiquettes, normalise les caractéristiques et divise les données en ensembles d'entraînement et de test.
    """

    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=['label', 'label_encoded']))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['label_encoded'], test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)  # Convertir en tableau NumPy avec .values
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)    # Convertir en tableau NumPy avec .values

    return X_train, X_test, y_train, y_test, label_encoder