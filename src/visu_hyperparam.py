import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = 'output/'
os.makedirs(output_dir, exist_ok=True)

# résulttas du tuning des hyperparamètres
data_path = 'data/hyperparameter_tuning_results.csv'
data = pd.read_csv(data_path)


# Précision et perte par type de modèle
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accuracy', y='Loss', hue='Model Type', data=data, legend=True)
plt.title('Précision/Perte par Type de Modèle')
plt.xlabel('Précision')
plt.ylabel('Perte')
plt.savefig(os.path.join(output_dir, 'precision_loss_by_model_type.png'))
plt.close()

# Précision et perte par dropout
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accuracy', y='Loss', hue='Dropout', data=data, palette='viridis', legend=True)
plt.title('Précision/Perte par Dropout')
plt.xlabel('Précision')
plt.ylabel('Perte')
plt.savefig(os.path.join(output_dir, 'precision_loss_by_dropout.png'))
plt.close()

# Précision et perte par taille des couches cachées
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accuracy', y='Loss', hue='Hidden Size', data=data, palette='viridis', legend=True)
plt.title('Précision/Perte par Taille des Couches Cachées')
plt.xlabel('Précision')
plt.ylabel('Perte')
plt.savefig(os.path.join(output_dir, 'precision_loss_by_hidden_size.png'))
plt.close()

# Précision et perte par nombre de couches
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accuracy', y='Loss', hue='Num Layers', data=data, palette='viridis', legend=True)
plt.title('Précision/Perte par Nombre de Couches')
plt.xlabel('Précision')
plt.ylabel('Perte')
plt.savefig(os.path.join(output_dir, 'precision_loss_by_num_layers.png'))
plt.close()

# Précision et perte par taux d'apprentissage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accuracy', y='Loss', hue='Learning Rate', data=data, palette='viridis', legend=True)
plt.title('Précision/Perte par Taux d\'Apprentissage')
plt.xlabel('Précision')
plt.ylabel('Perte')
plt.savefig(os.path.join(output_dir, 'precision_loss_by_learning_rate.png'))
plt.close()

# Précision et perte sans autre critère
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accuracy', y='Loss', data=data)
plt.title('Précision/Perte')
plt.xlabel('Précision')
plt.ylabel('Perte')
plt.savefig(os.path.join(output_dir, 'precision_loss.png'))
plt.close()

# Précision et perte avec nombre de couches et taille des couches
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accuracy', y='Loss', hue='Num Layers', style='Hidden Size', data=data, palette='viridis', legend=True)
plt.title('Précision/Perte par Nombre et Taille des Couches')
plt.xlabel('Précision')
plt.ylabel('Perte')
plt.savefig(os.path.join(output_dir, 'precision_loss_by_layers_and_size.png'))
plt.close()
