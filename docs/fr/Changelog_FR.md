### 2023-08-13
1-Corrections régulières de bugs
- Modification du nombre total d'époques minimum à 1 et changement du nombre total d'époques minimum à 2
- Correction des erreurs d'entraînement sans utiliser de modèles pré-entraînés
- Après la séparation des voix d'accompagnement, libération de la mémoire graphique
- Changement du chemin absolu d'enregistrement de faiss en chemin relatif
- Prise en charge des chemins contenant des espaces (le chemin du jeu de données d'entraînement et le nom de l'expérience sont pris en charge, et aucune erreur ne sera signalée)
- La liste de fichiers annule l'encodage utf8 obligatoire
- Résolution du problème de consommation de CPU causé par la recherche faiss lors des changements de voix en temps réel

2-Mises à jour clés
- Entraînement du modèle d'extraction de hauteur vocale open-source le plus puissant actuel, RMVPE, et utilisation pour l'entraînement, l'inférence hors ligne/en temps réel de RVC, supportant PyTorch/Onnx/DirectML
- Prise en charge des cartes graphiques AMD et Intel via Pytorch_DML

(1) Changement de voix en temps réel (2) Inférence (3) Séparation de l'accompagnement vocal (4) L'entraînement n'est pas actuellement pris en charge, passera à l'entraînement CPU; prend en charge l'inférence RMVPE de la GPU par Onnx_Dml

### 2023-06-18
- Nouveaux modèles pré-entraînés v2 : 32k et 48k
- Correction des erreurs d'inférence du modèle non-f0
- Pour un jeu de données d'entraînement dépassant 1 heure, réalisation automatique de minibatch-kmeans pour réduire la forme des caractéristiques, afin que l'entraînement, l'ajout et la recherche d'index soient beaucoup plus rapides.
- Fourniture d'un espace huggingface vocal2guitar jouet
- Suppression automatique des audios de jeu de données d'entraînement court-circuitant les valeurs aberrantes
- Onglet d'exportation Onnx

Expériences échouées:
- ~~Récupération de caractéristiques : ajout de la récupération de caractéristiques temporelles : non efficace~~
- ~~Récupération de caractéristiques : ajout de la réduction de dimensionnalité PCAR : la recherche est encore plus lente~~
- ~~Augmentation aléatoire des données lors de l'entraînement : non efficace~~

Liste de tâches:
- ~~Vocos-RVC (vocodeur minuscule) : non efficace~~
- ~~Support de Crepe pour l'entraînement : remplacé par RMVPE~~
- ~~Inférence de précision à moitié crepe : remplacée par RMVPE. Et difficile à réaliser.~~
- Support de l'éditeur F0

### 2023-05-28
- Ajout d'un cahier v2, changelog coréen, correction de certaines exigences environnementales
- Ajout d'un mode de protection des consonnes muettes et de la respiration
- Support de la détection de hauteur crepe-full
- Séparation vocale UVR5 : support des modèles de déréverbération et de désécho
- Ajout du nom de l'expérience et de la version sur le nom de l'index
- Support pour les utilisateurs de sélectionner manuellement le format d'exportation des audios de sortie lors du traitement de conversion vocale en lots et de la séparation vocale UVR5
- L'entraînement du modèle v1 32k n'est plus pris en charge

### 2023-05-13
- Nettoyage des codes redondants de l'ancienne version du runtime dans le package en un clic : lib.infer_pack et uvr5_pack
- Correction du bug de multiprocessus pseudo dans la préparation du jeu de données d'entraînement
- Ajout de l'ajustement du rayon de filtrage médian pour l'algorithme de reconnaissance de hauteur de récolte
- Prise en charge du rééchantillonnage post-traitement pour l'exportation audio
- Réglage de multi-traitement "n_cpu" pour l'entraînement est passé de "extraction f0" à "prétraitement des données et extraction f0"
- Détection automatique des chemins d'index sous le dossier de logs et fourniture d'une fonction de liste déroulante
- Ajout de "Questions fréquemment posées et réponses" sur la page d'onglet (vous pouvez également consulter le wiki github RVC)
- Lors de l'inférence, la hauteur de la récolte est mise en cache lors de l'utilisation du même chemin d'accès audio d'entrée (objectif : en utilisant l'extraction de

 la hauteur de la récolte, l'ensemble du pipeline passera par un long processus d'extraction de la hauteur répétitif. Si la mise en cache n'est pas utilisée, les utilisateurs qui expérimentent différents timbres, index, et réglages de rayon de filtrage médian de hauteur connaîtront un processus d'attente très douloureux après la première inférence)

### 2023-05-14
- Utilisation de l'enveloppe de volume de l'entrée pour mixer ou remplacer l'enveloppe de volume de la sortie (peut atténuer le problème du "muet en entrée et bruit de faible amplitude en sortie". Si le bruit de fond de l'audio d'entrée est élevé, il n'est pas recommandé de l'activer, et il n'est pas activé par défaut (1 peut être considéré comme n'étant pas activé)
- Prise en charge de la sauvegarde des modèles extraits à une fréquence spécifiée (si vous voulez voir les performances sous différentes époques, mais que vous ne voulez pas sauvegarder tous les grands points de contrôle et extraire manuellement les petits modèles par ckpt-processing à chaque fois, cette fonctionnalité sera très pratique)
- Résolution du problème des "erreurs de connexion" causées par le proxy global du serveur en définissant des variables d'environnement
- Prise en charge des modèles pré-entraînés v2 (actuellement, seule la version 40k est disponible au public pour les tests, et les deux autres taux d'échantillonnage n'ont pas encore été entièrement entraînés)
- Limite le volume excessif dépassant 1 avant l'inférence
- Réglages légèrement ajustés de la préparation du jeu de données d'entraînement

#######################

Historique des changelogs:

### 2023-04-09
- Correction des paramètres d'entraînement pour améliorer le taux d'utilisation du GPU : A100 est passé de 25% à environ 90%, V100 : de 50% à environ 90%, 2060S : de 60% à environ 85%, P40 : de 25% à environ 95% ; amélioration significative de la vitesse d'entraînement
- Changement de paramètre : la taille de batch_size totale est maintenant la taille de batch_size par GPU
- Changement de total_epoch : la limite maximale est passée de 100 à 1000 ; la valeur par défaut est passée de 10 à 20
- Correction du problème d'extraction de ckpt reconnaissant la hauteur de manière incorrecte, causant une inférence anormale
- Correction du problème d'entraînement distribué sauvegardant ckpt pour chaque rang
- Application du filtrage des caractéristiques nan pour l'extraction des caractéristiques
- Correction du problème d'entrée/sortie silencieuse produisant des consonnes aléatoires ou du bruit (les anciens modèles doivent être réentraînés avec un nouveau jeu de données)

### 2023-04-16 Mise à jour
- Ajout d'une mini-interface graphique pour le changement de voix en temps réel, démarrage par double-clic sur go-realtime-gui.bat
- Application d'un filtrage pour les bandes de fréquences inférieures à 50Hz pendant l'entraînement et l'inférence
- Abaissement de l'extraction de hauteur minimale de pyworld du défaut 80 à 50 pour l'entraînement et l'inférence, permettant aux voix masculines graves entre 50-80Hz de ne pas être mises en sourdine
- WebUI prend en charge le changement de langue en fonction des paramètres régionaux du système (prise en charge actuelle de en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW ; défaut à en_US si non pris en charge)
- Correction de la reconnaissance de certains GPU (par exemple, échec de reconnaissance V100-16G, échec de reconnaissance P4)

### 2023-04-28 Mise à jour
- Mise à niveau des paramètres d'index de faiss pour une vitesse plus rapide et une meilleure qualité
- Suppression de la dépendance à total_npy ; le partage futur de modèles ne nécessitera pas d'entrée total

_npy
- Levée des restrictions pour les GPU de la série 16, fournissant des paramètres d'inférence de 4 Go pour les GPU VRAM de 4 Go
- Correction d'un bug dans la séparation vocale d'accompagnement UVR5 pour certains formats audio
- La mini-interface de changement de voix en temps réel prend maintenant en charge les modèles de hauteur non-40k et non-lazy

### Plans futurs :
Fonctionnalités :
- Ajouter une option : extraire de petits modèles pour chaque sauvegarde d'époque
- Ajouter une option : exporter un mp3 supplémentaire vers le chemin spécifié pendant l'inférence
- Prise en charge de l'onglet d'entraînement multi-personnes (jusqu'à 4 personnes)

Modèle de base :
- Collecter des fichiers wav de respiration pour les ajouter au jeu de données d'entraînement pour résoudre le problème des sons de respiration déformés
- Nous entraînons actuellement un modèle de base avec un jeu de données de chant étendu, qui sera publié à l'avenir
