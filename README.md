# DATA832 - Machine Learning - Classification de genres musicaux

## Installation

### Étape 1 : Cloner le dépôt

Clonez ce dépôt sur votre machine locale en utilisant la commande suivante :

```bash
git clone https://github.com/Mathieeeu/data832_projet.git
cd data832_projet
```

### Étape 2 : Créer un environnement virtuel

Il est recommandé d'utiliser un environnement virtuel pour gérer les dépendances du projet. Voici comment créer et activer un environnement virtuel pour python :

#### Sur windows :

```bash
python -m venv env
.\env\Scripts\activate
```

#### Sur linux et mac :

```bash
python3 -m venv env
source env/bin/activate
```

Pour supprimer l'environnement virtuel, vous pouvez utiliser la commande suivante  :

```bash
deactivate
rmdir env /s /q
```


### Étape 3 : Installer les dépendances

Une fois l'environnement virtuel activé, installez les dépendances nécessaires en utilisant le fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

### Étape 4 : Lancer le programme

Pour lancer le programme, exécutez la commande suivante :

```bash
python src/main.py --display-plots
```

Il est possible de ne pas afficher les plots en supprimant l'argument `--display-plots`.

Il est aussi possible de regénérer les données extraites en précisant un nouveau nom de fichier csv et en indiquant le répertoire où sont stockées les données audio dans le `main.py`.