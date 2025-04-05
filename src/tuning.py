import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features
from preprocess import preprocess_data
from training import train_model

# Fonction pour évaluer un modèle avec des hyperparamètres donnés
def evaluate_model(X_train, X_test, y_train, y_test, label_encoder, model_type, params):
    model, train_losses, test_losses = train_model(
        X_train, X_test, y_train, y_test,
        label_encoder,
        model_type=model_type,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        learning_rate=params['learning_rate'],
        num_epochs=params['num_epochs'],
        early_stopping=params['early_stopping'],
        patience=params['patience']
    )

    model.eval()
    with torch.no_grad():
        if model_type == 'rnn':
            outputs = model(X_test.unsqueeze(1))
        else:
            outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted)
        loss = test_losses[-1]  # Dernière perte de test

    return accuracy, loss

param_grid = {
    'model_type': ['rnn', 'cnn', 'dense'],
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3, 4, 5],
    'dropout': [0.01, 0.05, 0.1, 0.3, 0.5],
    'learning_rate': [0.001, 0.0005, 0.0002, 0.0001],
    'num_epochs': [50, 100, 200],
    'early_stopping': [True],
    'patience': [5],
}

data_dir = 'D:/data/projet-data832/genres_original'
features_file = './data/features-complet.csv'
df = extract_features(data_dir, features_file)
X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)

results = []
n = 0
for params in ParameterGrid(param_grid):
    n += 1
    print(f"Model {n}/{len(ParameterGrid(param_grid))} : {params}")
    accuracy, loss = evaluate_model(X_train, X_test, y_train, y_test, label_encoder, params['model_type'], params)
    results.append([params['model_type'], params['hidden_size'], params['num_layers'],
                    params['dropout'], params['learning_rate'], params['num_epochs'],
                    params['early_stopping'], params['patience'], accuracy, loss])

results_df = pd.DataFrame(results, columns=[
    'Model Type', 'Hidden Size', 'Num Layers', 'Dropout', 'Learning Rate',
    'Num Epochs', 'Early Stopping', 'Patience', 'Accuracy', 'Loss'
])

results_df.sort_values(by='Accuracy', ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)
results_df.to_csv('./data/hyperparameter_tuning_results.csv', index=False)
print(results_df)
