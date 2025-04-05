from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pca_with_centers(X: np.ndarray, y: np.ndarray, labels: list) -> None:
    """
    Fonction pour afficher les données en 2D après réduction de dimension avec PCA.
    Elle affiche également les points moyens pour chaque genre musical.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    
    # Points moyens pour chaque genre
    centers = []
    for label in np.unique(y):
        mean_point = np.mean(X_pca[y == label], axis=0)
        centers.append(mean_point)

        plt.scatter(mean_point[0], mean_point[1], marker='x', s=200, c=[scatter.cmap(scatter.norm(label))], label=labels[label])

    cbar = plt.colorbar(scatter, ticks=range(len(labels)))
    cbar.set_label('Genres')
    cbar.set_ticks(np.arange(len(labels)))
    cbar.set_ticklabels(labels)
    
    plt.title("PCA - Réduction de Dimension avec Centres des Genres")
    plt.legend(loc='upper right')
    plt.savefig('output/pca_plot.png')
    plt.show()

def plot_train_test_loss(train_losses: list, test_losses: list) -> None:
    """
    Fonction pour afficher la courbe de perte d'entraînement et de test.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Perte d\'entrainement')
    plt.plot(test_losses, label='Perte de test')
    plt.title('Courbe de perte')
    plt.xlabel('Epoch')
    plt.ylabel('Perte')
    plt.legend()
    plt.savefig('output/loss_curve.png')
    plt.show()

def plot_confusion_matrix(conf_matrix: np.ndarray, classes: list) -> None:
    """
    Fonction pour afficher la matrice de confusion ( heatmap)
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d', 
                xticklabels=classes, 
                yticklabels=classes,
                cmap='Greens',
            )
    plt.title('Matrice de confusion')
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.savefig('output/confusion_matrix.png')
    plt.show()