# Autoencoder-Based Spike Sorting (ABSS)

Unsupervised feature extraction for spike sorting using autoencoders trained
with a custom, clustering-aware loss function.

This repository provides the implementation associated with:

> E. Kompaniets, S. Le Cam, R. Ranta. *Improving Feature Extraction for Spike
> Sorting Using a Custom Loss Function for an Autoencoder.* Biosignals /
> Biostec 2026.

The autoencoder architecture and baseline pipeline build upon:

> E-R. Ardelean, A. Coporîie, A-M. Ichim, M. Dînșoreanu, R.C. Mureșan. *A
> study of autoencoders as a feature extraction technique for spike sorting.*
> PLOS ONE 18(3): e0282810, 2023. https://doi.org/10.1371/journal.pone.0282810
> (original code: https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting)

---

## Overview

Spike sorting groups extracellularly recorded action potentials (spikes)
according to the neuron that generated them. The pipeline typically consists
of four steps: filtering, spike detection, **feature extraction**, and
clustering. Feature extraction is the step that most strongly determines the
separability of the resulting clusters.

This repository implements **Autoencoder-Based Spike Sorting (ABSS)**: an
unsupervised feature extraction method in which an autoencoder is trained to
produce a low-dimensional latent representation (the *code*) of spike
waveforms, subsequently clustered (K-means, GMM, DBSCAN).

### Custom loss function

Rather than optimizing only the reconstruction error (MSE), as in the
baseline autoencoder, the encoder is trained with a **combined loss** that
explicitly encourages separable clusters in the latent space:

$$
\mathcal{L}_\text{intra} = \frac{1}{K}\sum_{k=1}^{K} \mathcal{V}_k,
\qquad
\mathcal{L}_\text{inter} = \min_{\substack{k,\ell \in \{1,\dots,K\}\\ k \neq \ell}} \left\lVert \mu_k - \mu_\ell \right\rVert_2^2
$$

$$
\mathcal{L}_\text{combined} = \alpha\, \mathcal{L}_\text{intra} - \beta\, \mathcal{L}_\text{inter}
$$

where $\mu_k$ and $\mathcal{V}_k$ are the centroid and variance of class $k$
in the latent space. $\mathcal{L}_\text{intra}$ penalizes intra-cluster
dispersion, while $\mathcal{L}_\text{inter}$ rewards inter-cluster
separability.

### Pipeline

1. An unsupervised spike sorter provides an initial estimate of the clusters.
2. A representative subset of spikes (*core*) is selected from each
   estimated cluster.
3. The encoder is trained on this subset using the combined loss.
4. The trained encoder is applied to the full dataset, and clustering
   (K-means) is performed in the resulting latent space to identify single
   units.

Architecture: fully connected encoder/decoder, input layer of 79 samples per
spike waveform, hidden layers of 60/40/20 units, latent code size of 3
(dimension selected empirically).

## Repository structure

```
.
├── common/                  # shared utility functions
├── dataset_parsing/         # loading and parsing of simulated / real datasets
├── figures/                 # figure generation
├── global_analysis/         # aggregated result analysis across simulations
├── neural_networks/
│   └── autoencoder/
│       └── autoencoder.py   # autoencoder architectures and combined loss function
├── preprocess/               # spike alignment and scaling
├── real_data/                 # real recordings
├── utils/
├── validation/
│   └── performance.py        # clustering evaluation metrics (ARI, AMI, VM, DBS, CHS, SS) + DBSCAN
├── visualization/             # waveform / cluster visualization
├── weights/                   # trained model weights
├── ae_function.py             # core pipeline functions (data splitting, wavelet decomposition, train/test)
├── run.py                     # main entry point
├── sota_eval.py                # comparison against state-of-the-art feature extraction methods (PCA, ICA, Isomap)
├── requirements.txt
└── demo_nouvelles_fonctionnalites.py   # lightweight demo of the pipeline's key components
```

> **[TO COMPLETE]** — A few additional scripts are present in the repository
> whose exact purpose should be documented before publication, e.g.
> `run_simu_*.py`, `main_EK.py` / `main_SLC.py`, `inspecter_poids_h5.py`,
> `Affichage_par_nbClassesGT.py`, `creation_dataset.py`. Consider either
> documenting each briefly here, or moving exploratory/one-off scripts to a
> separate `scripts/` or `experiments/` folder so the main repository
> structure stays clean for readers of the paper.

## Data

**Synthetic data.** 95 publicly available single-channel simulations
(Department of Engineering, University of Leicester, UK; Pedreira et al.,
2012), each containing between 2 and 20 single-unit clusters plus one
multi-unit (noise) cluster, sampled at 24 kHz with 79 samples per spike
waveform.

**Real data.** Extracellular *in vivo* recordings (32-channel NeuroNexus
A32-tet probes) acquired from mouse visual cortex under visual stimulation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

> **[TO COMPLETE]** — describe the main entry point and its key arguments,
> e.g.:
> ```bash
> python run.py --simulation 4 --code_size 3 --clustering kmeans
> ```

A minimal, dependency-light demonstration of the core components
(data splitting, wavelet decomposition, DBSCAN classification, and the
corrected metric computation) is available in
[`demo_nouvelles_fonctionnalites.py`](./demo_nouvelles_fonctionnalites.py):

```bash
python demo_nouvelles_fonctionnalites.py
```

## Evaluation metrics

Six standard clustering metrics are used, following Ardelean et al. (2023):
three external metrics requiring ground-truth labels (Adjusted Rand Index,
Adjusted Mutual Information, V-Measure) and three internal metrics
(Davies-Bouldin Score, Calinski-Harabasz Score, Silhouette Score).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kompaniets2026abss,
  title     = {Improving Feature Extraction for Spike Sorting Using a Custom Loss Function for an Autoencoder},
  author    = {Kompaniets, Elizaveta and Le Cam, Steven and Ranta, Radu},
  booktitle = {Biosignals / Biostec},
  year      = {2026}
}

@article{ardelean2023autoencoders,
  title   = {A study of autoencoders as a feature extraction technique for spike sorting},
  author  = {Ardelean, Eugen-Richard and Coporîie, Andreea and Ichim, Ana-Maria and Dînșoreanu, Mihaela and Mureșan, Raul Cristian},
  journal = {PLOS ONE},
  volume  = {18},
  number  = {3},
  pages   = {e0282810},
  year    = {2023},
  doi     = {10.1371/journal.pone.0282810}
}
```

## Acknowledgments

Laboratoire CRAN (UMR 7039, CNRS – Université de Lorraine), in collaboration
with the Institut de Cancérologie de Lorraine and CHRU de Nancy.

## Contact

*(→ [TO COMPLETE] — add contact email(s) if desired)*
