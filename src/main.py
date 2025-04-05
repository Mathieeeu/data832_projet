import os
import torch
import argparse
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from feature_extraction import extract_features
from preprocess import preprocess_data
from training import train_model, save_model, load_model
from visualisation import plot_pca_with_centers, plot_train_test_loss, plot_confusion_matrix

# main dans une fonction pour éventuellement mettre des arguments à l'avenir
def main(display_plots: bool = True) -> None:

    # Récupération des caractéristiques audio à partir des fichiers .wav ou chargement à partir d'un fichier CSV
    data_dir = 'D:/data/projet-data832/genres_original'  # A EDITER SI BESOIN
    features_file = './data/features-complet.csv'  # fichier CSV pour stocker les caractéristiques extraites
    df = extract_features(data_dir, features_file)

    # Prétraitement
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)

    # Visualisation des données 
    if display_plots:
        # PCA pour afficher les samples en deux dimensions
        plot_pca_with_centers(X_test, y_test, label_encoder.classes_)

    ##################################################
    # Hyperparamètres du modèle 
    model_name = 'model' # nom du modèle pour l'enregistrement
    use_pretrained = False # utiliser un modèle déjà entrainé ?
    save_new_model = False # sauvegarder le modèle après l'entrainement ?

    model_type = 'rnn' # 'rnn' ou 'dense' ou 'cnn'
    hidden_size = 256
    num_layers = 2
    dropout = 0.3
    # taux de dropout = le pourcentage de neurones à ignorer pendant l'entrainement pour éviter le surapprentissage)

    num_epochs = 500 # maximum
    learning_rate = 0.0005

    early_stopping = True # arret anticipé de l'entrainement
    patience = 10  # Patience pour l'arret anticipé 
    ##################################################

    # Entrainement du modèle
        # Utiliser le modèle déjà entrainé si dispo
    if use_pretrained and os.path.exists(f'./models/{model_name}.pth'):
        print(f"Chargement du modèle pré-entraîné depuis {model_name}.pth")
        print(f"Date de création : {os.path.getctime(f'./models/{model_name}.pth')}\n")
        model = load_model(model_name)
    else:
        print("Entrainement d'un nouveau modèle...")
        model, train_losses, test_losses = train_model(
            X_train, X_test, y_train, y_test,
            label_encoder,
            model_type=model_type,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, 
            num_epochs=num_epochs, 
            early_stopping=early_stopping,
            patience=patience
        )
        if save_new_model:
            save_model(model, model_name)
            print(f"Modèle enregistré sous {model_name}.pth\n")

        # visualisation de la courbe de pertes
        if display_plots:
            print("Affichage de la courbe de pertes...")
            plot_train_test_loss(train_losses, test_losses)

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
        if display_plots:
            print("Affichage de la matrice de confusion...")
            plot_confusion_matrix(conf_matrix, label_encoder.classes_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification de Genres Musicaux")
    parser.add_argument('--display-plots', action='store_true', help="Afficher les graphiques")
    args = parser.parse_args()
    main(display_plots=args.display_plots)

    # Exécution : 
    #  pour ne pas afficher les graphiques : python main.py 
    #  pour afficher les graphiques :        python main.py --display-plots 
