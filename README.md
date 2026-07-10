# Autoencoders in Spike Sorting — Scripts complémentaires

Ce dépôt contient les scripts que nous avons développés en complément du travail
original d'Ardelean et al. sur l'utilisation d'autoencodeurs comme méthode
d'extraction de caractéristiques pour le spike sorting.

## Article et dépôt original

Ce travail s'appuie entièrement sur le code et les données publiés par les auteurs
de l'article suivant :

> E.-R. Ardelean, A. Coporîie, A.-M. Ichim, M. Dînșoreanu, R. C. Mureșan,
> "A study of autoencoders as a feature extraction technique for spike sorting",
> PLOS ONE 18(3): e0282810, 2023.
> https://doi.org/10.1371/journal.pone.0282810

Dépôt GitHub original : https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting

Nos scripts (`ae_function_cb.py`, `main_sim.py`, `main_real.py`) doivent être
utilisés **en complément** du code de ce dépôt, pas comme un projet
indépendant.

## Installation

### 1. Télécharger le dépôt original

```bash
git clone https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting.git
cd Autoencoders-in-Spike-Sorting
```

### 2. Installer les dépendances

Les dépendances nécessaires sont listées dans le fichier `requirements.txt` du
dépôt original :

```bash
pip install -r requirements.txt
```

### 3. Télécharger les données

- **Données synthétiques** (simulations) :
  https://1drv.ms/u/s!AgNd2yQs3Ad0gSjeHumstkCYNcAk?e=QfGIJO
  ou https://www.kaggle.com/datasets/ardeleanrichard/simulationsdataset
- **Données réelles** :
  https://www.kaggle.com/datasets/ardeleanrichard/realdata
  ou dossier `real_data` du dépôt original

Placer les données selon l'arborescence recommandée par le dépôt original
(chemin à configurer dans `constants.py`) :

```
DATA/
├── TINS/
│   └── M045_009/      <- fichiers de données réelles
└── SIMULATIONS/       <- fichiers de données synthétiques
```

### 4. Ajouter nos fichiers

Placer nos fichiers dans le dépôt cloné, aux emplacements suivants (même
arborescence que le dépôt original) :

```
Autoencoders-in-Spike-Sorting/
├── ae_function.py          <- à remplacer par notre version (code original
│                               de l'article, inchangé, + import vers
│                               ae_function_cb.py)
├── ae_function_cb.py       <- NOUVEAU : fonctions que nous avons ajoutées
│                               (separation_data, decomposition_ondelettes,
│                               run_autoencoder_train, run_autoencoder_test,
│                               create_dossier, save_model, ...)
├── main_sim.py             <- NOUVEAU : reproduit nos résultats sur les
│                               données simulées
├── main_real.py            <- NOUVEAU : reproduit nos résultats sur les
│                               données réelles
├── validation/
│   └── performance.py      <- à remplacer par notre version (ajout de
│                               classification_DBSCAN + correction du calcul
│                               des métriques internes)
├── neural_networks/
│   └── autoencoder/
│       └── autoencoder.py  <- à remplacer par notre version (ajustements des
│                               fonctions de perte intra/inter-classe)
└── visualization/
    └── scatter_plot.py     <- à remplacer par notre version (ajout d'un
                                titre sur les graphiques 3D)
```

> Les autres fichiers du dépôt original (`dataset_parsing/`,
> `preprocess/`, `utils/`, etc.) restent inchangés.

## Lancer les scripts

Une fois l'installation ci-dessus effectuée, se placer à la racine du dépôt et
lancer :

```bash
# Reproduire les résultats sur les données simulées
python main_sim.py

# Reproduire les résultats sur les données réelles
python main_real.py
```

## Résumé de nos modifications par rapport au code original

| Fichier | Modification |
|---|---|
| `ae_function.py` → `ae_function_cb.py` | Fonctions ajoutées séparées dans un nouveau fichier, importé depuis `ae_function.py` (inchangé) |
| `validation/performance.py` | Ajout de `classification_DBSCAN` ; correction d'un bug (les métriques internes DBS/CHS/SS étaient calculées sur les labels réels au lieu des labels prédits) |
| `neural_networks/autoencoder/autoencoder.py` | Ajustements des fonctions de perte intra/inter-classe |
| `visualization/scatter_plot.py` | Ajout d'un titre sur les graphiques 3D |

## Contact

<!-- TODO : ajouter vos coordonnées / celles de votre encadrant -->
