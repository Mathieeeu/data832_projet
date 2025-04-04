import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

# Extraction des caractéristiques du fichier audio
def extract_features_from_file(file_path: str) -> np.ndarray:
    """
    Fonction pour extraire les caractéristiques audio d'un fichier .wav via la bibliothèque librosa.

    Les caractéristiques extraites sont :
    - mfccs : coefficients cepstraux en fréquence Mel (en anglais Mel-frequency cepstral coefficients)
    - chroma : chromagramme de la piste audio, représentant l'énergie des différentes classes de hauteur chromatique au fil du temps (12 classes : Do, Do#, Re, Re#, Mi, Fa, Fa#, Sol, Sol#, La, La#, Si)
    - spectral_contrast : contraste spectral, qui mesure la différence entre les pics et les creux du spectre audio
    - tempo : le tempo de la piste audio, en bpm
    - tonnetz : c'est une représentation des relations tonales et harmoniques dans la musique (tonnetz = réseau tonal en allemand)
    - mel_spectrogram : représentation visuelle de l'énergie d'un signal audio dans le domaine temps-fréquence, mais avec une transformation qui reflète la perception humaine des fréquences (grave à aiguë, mesurée en Mel)
    """
    y, sr = librosa.load(file_path, sr=None) # y = signal audio, sr = fréquence d'échantillonnage
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    return np.hstack((
        tempo,    
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(tonnetz, axis=1),
        np.mean(mel_spectrogram, axis=1)
    ))

def extract_features_from_file_parallel(file_paths: str, max_workers: int = None) -> list:
    """
    Tentative de parallélisation de l'extraction des caractéristiques audio pour aller plus vite, ça marche peut etre :p
    """
    if max_workers is None:
        # utiliser le nombre de CPU disponibles par défaut
        max_workers = min(32, os.cpu_count() + 4)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        features = list(tqdm(executor.map(extract_features_from_file, file_paths), total=len(file_paths)))
    return features

def extract_features(data_dir: str, features_file: str) -> pd.DataFrame:
    """
    Fonction pour extraire les caractéristiques audio des fichiers .wav dans le répertoire spécifié.
    Si le fichier CSV existe déjà, il charge les caractéristiques à partir de celui-ci.
    Sinon, il extrait les caractéristiques et les sauvegarde dans le fichier CSV.

    Args:
        data_dir (str): Répertoire contenant les fichiers audio .wav.
        features_file (str): Chemin du fichier CSV pour sauvegarder les caractéristiques extraites.

    Retourne:
        pd.DataFrame: DataFrame contenant les caractéristiques audio extraites à partir des fichiers audio.
    """
    if os.path.exists(features_file):
        print("\nFichier CSV déjà existant. Chargement des caractéristiques...")
        df = pd.read_csv(features_file)
        print(f"\tCaractéristiques chargées depuis : {features_file}")
    else:
        print("\nExtraction des caractéristiques...")
        features = []
        labels = []

        all_files = [os.path.join(root, file) for root, dirs, files in os.walk(data_dir) for file in files if file.endswith('.wav')]

        features = extract_features_from_file_parallel(all_files, max_workers=10)

        for file_path in all_files:
            genre = os.path.basename(os.path.dirname(file_path))
            labels.append(genre)

        column_names = ['tempo'] + \
            [f'mfcc_{i+1}' for i in range(20)] + \
            [f'chroma_{i+1}' for i in range(12)] + \
            [f'spectral_contrast_{i+1}' for i in range(7)] + \
            [f'tonnetz_{i}' for i in range(1, 7)] + \
            [f'mel_spectrogram_{i}' for i in range(1, 129)]

        df = pd.DataFrame(features, columns=column_names)
        df['label'] = labels
        df.to_csv(features_file, index=False)
        print(f"\tCaractéristiques sauvegardées dans : {features_file}")
    
    print(f"\tNombre de fichiers audio traités : {len(df)}")
    print(f"\tNombre de caractéristiques extraites : {df.shape[1] - 1}\n")  # -1 pour exclure la colonne 'label'
    return df

if __name__ == "__main__":
    # Chemin vers le répertoire où se trouve le dataset
    data_dir = 'D:/data/projet-data832/genres_original'  # A EDITER SI BESOIN
    features_file = './data/features-complet.csv'  # fichier CSV pour stocker les caractéristiques extraites

    df = extract_features(data_dir, features_file)
    print(df.head())  # Afficher les premières lignes du DataFrame

