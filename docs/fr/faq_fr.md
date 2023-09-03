## Q1: Erreur ffmpeg/erreur utf8.
Il s'agit très probablement non pas d'un problème lié à FFmpeg, mais d'un problème lié au chemin de l'audio ;

FFmpeg peut rencontrer une erreur lors de la lecture de chemins contenant des caractères spéciaux tels que des espaces et (), ce qui peut provoquer une erreur FFmpeg ; et lorsque l'audio du jeu d'entraînement contient des chemins en chinois, l'écrire dans filelist.txt peut provoquer une erreur utf8.<br>

## Q2: Impossible de trouver le fichier index après "Entraînement en un clic".
Si l'affichage indique "L'entraînement est terminé. Le programme est fermé", alors le modèle a été formé avec succès, et les erreurs subséquentes sont fausses ;

L'absence d'un fichier index 'ajouté' après un entraînement en un clic peut être due au fait que le jeu d'entraînement est trop grand, ce qui bloque l'ajout de l'index ; cela a été résolu en utilisant un traitement par lots pour ajouter l'index, ce qui résout le problème de surcharge de mémoire lors de l'ajout de l'index. Comme solution temporaire, essayez de cliquer à nouveau sur le bouton "Entraîner l'index".<br>

## Q3: Impossible de trouver le modèle dans “Inférence du timbre” après l'entraînement
Cliquez sur “Actualiser la liste des timbres” et vérifiez à nouveau ; si vous ne le voyez toujours pas, vérifiez s'il y a des erreurs pendant l'entraînement et envoyez des captures d'écran de la console, de l'interface utilisateur web, et des logs/nom_de_l'expérience/*.log aux développeurs pour une analyse plus approfondie.<br>

## Q4: Comment partager un modèle/Comment utiliser les modèles d'autres personnes ?
Les fichiers pth stockés dans rvc_root/logs/nom_de_l'expérience ne sont pas destinés à être partagés ou inférés, mais à stocker les points de contrôle de l'expérience pour la reproductibilité et l'entraînement ultérieur. Le modèle à partager doit être le fichier pth de 60+MB dans le dossier des poids ;

À l'avenir, les poids/nom_de_l'expérience.pth et les logs/nom_de_l'expérience/ajouté_xxx.index seront fusionnés en un seul fichier poids/nom_de_l'expérience.zip pour éliminer le besoin d'une entrée d'index manuelle ; partagez donc le fichier zip, et non le fichier pth, sauf si vous souhaitez continuer l'entraînement sur une machine différente ;

Copier/partager les fichiers pth de plusieurs centaines de Mo du dossier des logs au dossier des poids pour une inférence forcée peut entraîner des erreurs telles que des f0, tgt_sr, ou d'autres clés manquantes. Vous devez utiliser l'onglet ckpt en bas pour sélectionner manuellement ou automatiquement (si l'information se trouve dans les logs/nom_de_l'expérience), si vous souhaitez inclure les informations sur la hauteur et les options de taux d'échantillonnage audio cible, puis extraire le modèle plus petit. Après extraction, il y aura un fichier pth de 60+ MB dans le dossier des poids, et vous pouvez actualiser les voix pour l'utiliser.<br>

## Q5: Erreur de connexion.
Il se peut que vous ayez fermé la console (fenêtre de ligne de commande noire).<br>

## Q6: WebUI affiche 'Expecting value: line 1 column 1 (char 0)'.
Veuillez désactiver le proxy système LAN/proxy global puis rafraîchir.<br>

## Q7: Comment s'entraîner et déduire sans le WebUI ?
Script d'entraînement :<br>
Vous pouvez d'abord lancer l'entraînement dans WebUI, et les versions en ligne de commande de la préparation du jeu de données et de l'entraînement seront affichées dans la fenêtre de message.<br>

Script d'inférence :<br>
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/myinfer.py<br>

Par exemple :<br>

runtime\python.exe myinfer.py 0 "E:\codes\py39\RVC-beta\todo-songs\1111.wav" "E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index" récolte "test.wav" "weights/mi-test.pth" 0.6 cuda:0 True<br>

f0up_key=sys.argv[1]<br>
input_path=sys.argv[2]<br>
index_path=sys.argv[3]<br>
f0method=sys.argv[4]#récolte ou pm<br>
opt_path=sys.argv[5]<br>
model_path=sys.argv[6]<br>
index_rate=float(sys.argv[7])<br>
device=sys.argv[8]<br>
is_half=bool(sys.argv[9])<br>

### Explication des arguments :

1. **Numéro de voix cible** : `0` (dans cet exemple)
2. **Chemin du fichier audio d'entrée** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\INPUTS_VOCAL\vocal.wav"`
3. **Chemin du fichier index** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\logs\Hagrid.index"`
4. **Méthode pour l'extraction du pitch (F0)** : `harvest` (dans cet exemple)
5. **Chemin de sortie pour le fichier audio traité** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\INPUTS_VOCAL\test.wav"`
6. **Chemin du modèle** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\weights\HagridFR.pth"`
7. **Taux d'index** : `0.6` (dans cet exemple)
8. **Périphérique pour l'exécution (GPU/CPU)** : `cuda:0` pour une carte NVIDIA, par exemple.
9. **Protection des droits d'auteur (True/False)**.

<!-- Pour myinfer nouveau models :

runtime\python.exe myinfer.py 0 "C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\INPUTS_VOCAL\vocal.wav" "C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\logs\Hagrid.index" harvest "C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\INPUTS_VOCAL\test.wav" "C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\weights\HagridFR.pth" 0.6 cuda:0 True 5 44100 44100 1.0 1.0 True


f0up_key=sys.argv[1]                     
input_path = sys.argv[2]                 
index_path = sys.argv[3]                 
f0method = sys.argv[4]                   
opt_path = sys.argv[5]                   
model_path = sys.argv[6]                 
index_rate = float(sys.argv[7])          
device = sys.argv[8]                     
is_half = bool(sys.argv[9])              
filter_radius = int(sys.argv[10])        
tgt_sr = int(sys.argv[11])               
resample_sr = int(sys.argv[12])          
rms_mix_rate = float(sys.argv[13])       
version = sys.argv[14]                   
protect = sys.argv[15].lower() == 'false' # change for true if needed

### Explication des arguments :

1. **Numéro de voix cible** : `0` (dans cet exemple)
2. **Chemin du fichier audio d'entrée** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\INPUTS_VOCAL\vocal.wav"`
3. **Chemin du fichier index** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\logs\Hagrid.index"`
4. **Méthode pour l'extraction du pitch (F0)** : `harvest` (dans cet exemple)
5. **Chemin de sortie pour le fichier audio traité** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\INPUTS_VOCAL\test.wav"`
6. **Chemin du modèle** : `"C:\ YOUR PATH FOR THE ROOT (RVC0813Nvidia)\weights\HagridFR.pth"`
7. **Taux d'index** : `0.6` (dans cet exemple)
8. **Périphérique pour l'exécution (GPU/CPU)** : `cuda:0` pour une carte NVIDIA, par exemple.
9. **Protection des droits d'auteur (True/False)**.
10. **Rayon du filtre** : `5` (dans cet exemple)
11. **Taux d'échantillonnage cible** : `44100` (dans cet exemple)
12. **Taux d'échantillonnage pour le rééchantillonnage** : `44100` (dans cet exemple)
13. **Taux de mixage RMS** : `1.0` (dans cet exemple)
14. **Version** : `1.0` (dans cet exemple)
15. **Protection** : `True` (dans cet exemple)

Assurez-vous de remplacer les chemins par ceux correspondant à votre configuration et d'ajuster les autres paramètres selon vos besoins.
-->

## Q8: Erreur Cuda/Mémoire Cuda épuisée.
Il y a une faible chance qu'il y ait un problème avec la configuration CUDA ou que le dispositif ne soit pas pris en charge ; plus probablement, il n'y a pas assez de mémoire (manque de mémoire).<br>

Pour l'entraînement, réduisez la taille du lot (si la réduction à 1 n'est toujours pas suffisante, vous devrez peut-être changer la carte graphique) ; pour l'inférence, ajustez les paramètres x_pad, x_query, x_center, et x_max dans le fichier config.py selon les besoins. Les cartes mémoire de 4 Go ou moins (par exemple 1060(3G) et diverses cartes de 2 Go) peuvent être abandonnées, tandis que les cartes mémoire de 4 Go ont encore une chance.<br>

## Q9: Combien de total_epoch sont optimaux ?
Si la qualité audio du jeu d'entraînement est médiocre et que le niveau de bruit est élevé, 20-30 époques sont suffisantes. Le fixer trop haut n'améliorera pas la qualité audio de votre jeu d'entraînement de faible qualité.<br>

Si la qualité audio du jeu d'entraînement est élevée, le niveau de bruit est faible, et la durée est suffisante, vous pouvez l'augmenter. 200 est acceptable (puisque l'entraînement est rapide, et si vous êtes capable de préparer un jeu d'entraînement de haute qualité, votre GPU peut probablement gérer une durée d'entraînement plus longue sans problème).<br>

## Q10: Quelle durée de jeu d'entraînement est nécessaire ?
Un jeu d'environ 10 min à 50 min est recommandé.<br>

Avec une garantie de haute qualité sonore et de faible bruit de fond, plus peut être ajouté si le timbre du jeu est uniforme.<br>

Pour un jeu d'entraînement de haut niveau (ton maigre + ton distinctif), 5 min à 10 min sont suffisantes.<br>

Il y a des personnes qui ont réussi à s'entraîner avec des données de 1 min à 2 min, mais le succès n'est pas reproductible par d'autres et n'est pas très informatif. <br>Cela nécessite que le jeu d'entraînement ait un timbre très distinctif (par exemple, un son de fille d'anime aérien à haute fréquence) et que la qualité de l'audio soit élevée ;
Aucune tentative réussie n'a été faite jusqu'à présent avec des données de moins de 1 min. Cela n'est pas recommandé.<br>

## Q11: À quoi sert le taux d'index et comment l'ajuster ?
Si la qualité tonale du modèle pré-entraîné et de la source d'inférence est supérieure à celle du jeu d'entraînement, ils peuvent améliorer la qualité tonale du résultat d'inférence, mais au prix d'un possible biais tonal vers le ton du modèle sous-jacent/source d'inférence plutôt que le ton du jeu d'entraînement, ce qui est généralement appelé "fuite de ton".<br>

Le taux d'index est utilisé pour réduire/résoudre le problème de la fuite de timbre. Si le taux d'index est fixé à 1, théoriquement il n'y a pas de fuite de timbre de la source d'inférence et la qualité du timbre est plus biaisée vers le jeu d'entraînement. Si le jeu d'entraînement a une qualité sonore inférieure à celle de la source d'inférence, alors un taux d'index plus élevé peut réduire la qualité sonore. Le réduire à 0 n'a pas l'effet d'utiliser le mélange de récupération pour protéger les tons du jeu d'entraînement.<br>

Si le jeu d'entraînement a une bonne qualité audio et une longue durée, augmentez le total_epoch, lorsque le modèle lui-même est moins susceptible de se référer à la source déduite et au modèle sous-jacent pré-entraîné, et qu'il y a peu de "fuite de ton", le taux d'index n'est pas important et vous pouvez même ne pas créer/partager le fichier index.<br>

## Q12: Comment choisir le gpu lors de l'inférence ?
Dans le fichier config.py, sélectionnez le numéro de carte après "device cuda:".<br>

La correspondance entre le numéro de carte et la carte graphique peut être vue dans la section d'information de la carte graphique de l'onglet d'entraînement.<br>

## Q13: Comment utiliser le modèle sauvegardé au milieu de l'entraînement ?
Sauvegardez via l'extraction de modèle en bas de l'onglet de traitement ckpt.

## Q14: Erreur de fichier/erreur de mémoire (lors de l'entraînement) ?
Il y a trop de processus et votre mémoire n'est pas suffisante. Vous pouvez le corriger en :

1. Diminuer l'entrée dans le champ "Threads of CPU".

2. Pré-découper le jeu d'entraînement en fichiers audio plus courts.

## Q15: Comment poursuivre l'entraînement avec plus de données

étape 1 : mettre toutes les données wav dans path2.

étape 2 : exp_name2+path2 -> traiter le jeu de données et extraire la caractéristique.

étape 3 : copier les derniers fichiers G et D de exp_name1 (votre expérience précédente) dans le dossier exp_name2.

étape 4 : cliquez sur "entraîner le modèle", et il continuera l'entraînement depuis le début de votre époque de modèle exp précédente.

## Q16: erreur à propos de llvmlite.dll

OSError: Impossible de charger le fichier objet partagé : llvmlite.dll

FileNotFoundError: Impossible de trouver le module lib\site-packages\llvmlite\binding\llvmlite.dll (ou l'une de ses dépendances). Essayez d'utiliser la syntaxe complète du constructeur.

Le problème se produira sous Windows, installez https://aka.ms/vs/17/release/vc_redist.x64.exe et il sera corrigé.

## Q17: RuntimeError: La taille étendue du tensor (17280) doit correspondre à la taille existante (0) à la dimension non-singleton 1. Tailles cibles : [1, 17280]. Tailles des tensors : [0]

Supprimez les fichiers wav dont la taille est nettement inférieure à celle des autres, et cela ne se reproduira plus. Ensuite, cliquez sur "entraîner le modèle" et "entraîner l'index".

## Q18: RuntimeError: La taille du tensor a (24) doit correspondre à la taille du tensor b (16) à la dimension non-singleton 2

Ne changez pas le taux d'échantillonnage puis continuez l'entraînement. S'il est nécessaire de changer, le nom de l'expérience doit être modifié et le modèle sera formé à partir de zéro. Vous pouvez également copier les hauteurs et caractéristiques (dossiers 0/1/2/2b) extraites la dernière fois pour accélérer le processus d'entraînement.

