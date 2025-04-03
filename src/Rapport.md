# DATA832 - Classification de genres musicaux

_Mathieu - Louna_

-------------------------------------------

## Structure des données

Notre but est de créer un modéle capable de déterminer le genre d'une musique. Notre dataset se compose de 1 000 morceaux de 10 genres différents : blues, classical, country, disco, hiphop, jazz, métal, pop, reggae et rock (100 pistes par genre).

Les données utilisées se décomposent en deux parties. Une première composé des 1 000 fichiers audio et une seconde de fichiers csv. Ils rassemblent les données des morceaux sur 3 et 30 secondes et sur les morceaux complets. Dans ces fichiers sont extraits les features importantes des musiques telles que **à expliquer**
- le chroma
- les MFCC au temps 1 à 20
- le tempo
- l'harmonie


## Différentes approches

Nous avons testé différentes approches et librairies afin de trouver le meilleur modèle de prédiction possible.

### Keras

### Torch

### Torch & Librosa

## Appel à un ami

Nous avons principalement utilisé l'IA (notamment ChatGPT) pour résoudre des problèmes configurations de paramètres.

Le premier prompt que nous avons fait concerne la matrice de confusion (modèle utilisant les csv fournis).
```prompt
J'ai une erreur de dimensio avec la matrice de confusion
```
Nous savions que l'erreur venait d'un problème de dimension (2D au lieu de 3D) et l'IA nous a permit de gagner du temps. Nous aurions résolu le problème en cherchant par tatonnement et sur les forums mais cela ne nous aurait pas forcément apporté beaucoup. L'erreur venait bien de là et à été résolu en rajoutant une dimension à nos ensembles de test et de train
```python
X_train = np.expand_dims(X_train, axis=-1)  # (samples, 58, 1)
X_test = np.expand_dims(X_test, axis=-1)
```

Notre but est de trouver le modèle avec la meilleure précision, nous avons donc testé différents paramètres pour la fonction d'activation entre autre. Les tests ne montrant pas d'amélioration significative, nous avons intérrogé l'IA pour savoir ce qu'il ferrait et si cela améliorait notre accuracy.
**Me rappel plus de la conclusion et j'ai pas de réseau pour vérifier le prompt et le contexte 0_0'**


Paramètres différents 
- fonction d'activation
  
Explication recall, precision, f1-score (est-ce qu'on le met vraiment parce que en soit tu avais ton cours et après coup je me rends compte que j'aurais pas du l'utiliser pour ça (pensons aux ours polaires et aux petits pingouins tout mignon qui savent pas nager))


Résolution d'erreurs avec svm et xgb (les fit marchent pas >.<)
pour xgb -> attendait des valeurs numériques suffit de mettre un encoder
pour svm -> mauvais choix des paramètres



## Amélioration possible

## Conclusion