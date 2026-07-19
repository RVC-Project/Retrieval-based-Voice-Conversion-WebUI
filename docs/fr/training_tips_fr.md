Instructions et conseils pour la formation RVC
======================================
Ces conseils expliquent comment se déroule la formation des données.

# Flux de formation
Je vais expliquer selon les étapes de l'onglet de formation de l'interface graphique.

## étape 1
Définissez ici le nom de l'expérience.

Vous pouvez également définir ici si le modèle doit prendre en compte le pitch.
Si le modèle ne considère pas le pitch, le modèle sera plus léger, mais pas adapté au chant.

Les données de chaque expérience sont placées dans `/logs/nom-de-votre-experience/`.

## étape 2a
Charge et pré-traite l'audio.

### charger l'audio
Si vous spécifiez un dossier avec de l'audio, les fichiers audio de ce dossier seront lus automatiquement.
Par exemple, si vous spécifiez `C:Users\hoge\voices`, `C:Users\hoge\voices\voice.mp3` sera chargé, mais `C:Users\hoge\voices\dir\voice.mp3` ne sera pas chargé.

Comme ffmpeg est utilisé en interne pour lire l'audio, si l'extension est prise en charge par ffmpeg, elle sera lue automatiquement.
Après la conversion en int16 avec ffmpeg, convertir en float32 et normaliser entre -1 et 1.

### débruitage
L'audio est lissé par filtfilt de scipy.

### Séparation audio
Tout d'abord, l'audio d'entrée est divisé en détectant des parties de silence qui durent plus d'une certaine période (max_sil_kept = 5 secondes ?). Après avoir séparé l'audio sur le silence, séparez l'audio toutes les 4 secondes avec un chevauchement de 0,3 seconde. Pour l'audio séparé en 4 secondes, après normalisation du volume, convertir le fichier wav en `/logs/nom-de-votre-experience/0_gt_wavs` puis le convertir à un taux d'échantillonnage de 16k dans `/logs/nom-de-votre-experience/1_16k_wavs` sous forme de fichier wav.

## étape 2b
### Extraire le pitch
Extrait les informations de pitch des fichiers wav. Extraire les informations de pitch (=f0) en utilisant la méthode intégrée dans parselmouth ou pyworld et les sauvegarder dans `/logs/nom-de-votre-experience/2a_f0`. Convertissez ensuite logarithmiquement les informations de pitch en un entier entre 1 et 255 et sauvegardez-le dans `/logs/nom-de-votre-experience/2b-f0nsf`.

### Extraire l'empreinte de caractéristique
Convertissez le fichier wav en incorporation à l'avance en utilisant HuBERT. Lisez le fichier wav sauvegardé dans `/logs/nom-de-votre-experience/1_16k_wavs`, convertissez le fichier wav en caractéristiques de dimension 256 avec HuBERT, et sauvegardez au format npy dans `/logs/nom-de-votre-experience/3_feature256`.

## étape 3
former le modèle.
### Glossaire pour les débutants
Dans l'apprentissage profond, l'ensemble de données est divisé et l'apprentissage progresse petit à petit. Dans une mise à jour de modèle (étape), les données de batch_size sont récupérées et des prédictions et corrections d'erreur sont effectuées. Faire cela une fois pour un ensemble de données compte comme une époque.

Par conséquent, le temps d'apprentissage est le temps d'apprentissage par étape x (le nombre de données dans l'ensemble de données / taille du lot) x le nombre d'époques. En général, plus la taille du lot est grande, plus l'apprentissage devient stable (temps d'apprentissage par étape ÷ taille du lot) devient plus petit, mais il utilise plus de mémoire GPU. La RAM GPU peut être vérifiée avec la commande nvidia-smi. L'apprentissage peut être effectué en peu de temps en augmentant la taille du lot autant que possible selon la machine de l'environnement d'exécution.

### Spécifier le modèle pré-entraîné
RVC commence à former le modèle à partir de poids pré-entraînés plutôt que de zéro, il peut donc être formé avec un petit ensemble de données.

Par défaut :

- Si vous considérez le pitch, il charge `rvc-location/pretrained/f0G40k.pth` et `rvc-location/pretrained/f0D40k.pth`.
- Si vous ne considérez pas le pitch, il charge `rvc-location/pretrained/f0G40k.pth` et `rvc-location/pretrained/f0D40k.pth`.

Lors de l'apprentissage, les paramètres du modèle sont sauvegardés dans `logs/nom-de-votre-experience/G_{}.pth` et `logs/nom-de-votre-experience/D_{}.pth` pour chaque save_every_epoch, mais en spécifiant ce chemin, vous pouvez démarrer l'apprentissage. Vous pouvez redémarrer ou commencer à former à partir de poids de modèle appris lors d'une expérience différente.

### Index d'apprentissage
RVC sauvegarde les valeurs de caractéristique HuBERT utilisées lors de la formation, et pendant l'inférence, recherche les valeurs de caractéristique qui sont similaires aux valeurs de caractéristique utilisées lors de l'apprentissage pour effectuer l'inférence. Afin d'effectuer cette recherche à haute vitesse, l'index est appris à l'avance.
Pour l'apprentissage d'index, nous utilisons la bibliothèque de recherche de voisinage approximatif faiss. Lisez la valeur de caractéristique de `logs/nom-de-votre-experience/3_feature256` et utilisez-la pour apprendre l'index, et sauvegardez-la sous `logs/nom-de-votre-experience/add_XXX.index`.

(À partir de la version de mise à jour 20230428, elle est lue à partir de l'index, et la sauvegarde / spécification n'est plus nécessaire.)

### Description du bouton
- Former le modèle : après avoir exécuté l'étape 2b, appuyez sur ce bouton pour former le modèle.
- Former l'index de caractéristique : après avoir formé le modèle, effectuez un apprentissage d'index.
- Formation en un clic : étape 2b, formation du modèle et formation de l'index de caractéristique tout d'un coup.```
