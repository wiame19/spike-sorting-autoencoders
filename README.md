# Autoencoder-Based Spike Sorting (ABSS)

# Autoencoder-Based Spike Sorting (ABSS)

Unsupervised feature extraction for spike sorting, using autoencoders trained
with a clustering-aware loss function.

Code accompanying:

> E. Kompaniets, S. Le Cam, R. Ranta. **Improving Feature Extraction for Spike
> Sorting Using a Custom Loss Function for an Autoencoder.** Biosignals /
> Biostec 2026.

Built on top of the autoencoder architectures from:

> E-R. Ardelean, A. Coporîie, A-M. Ichim, M. Dînșoreanu, R.C. Mureșan. **A
> study of autoencoders as a feature extraction technique for spike
> sorting.** PLOS ONE 18(3): e0282810, 2023.
> [doi.org/10.1371/journal.pone.0282810](https://doi.org/10.1371/journal.pone.0282810)
> · [original code](https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting)

---

## Method

Spike sorting pipelines rely on filtering, spike detection, feature
extraction, and clustering — with feature extraction being the step that
most determines the separability of the resulting clusters.

ABSS trains an autoencoder to produce a low-dimensional latent representation
of spike waveforms, optimized not only for reconstruction but for
**cluster separability**:

$$
\mathcal{L}_\text{combined} = \alpha\, \mathcal{L}_\text{intra} - \beta\, \mathcal{L}_\text{inter}
$$

$$
\mathcal{L}_\text{intra} = \frac{1}{K}\sum_{k=1}^{K} \mathcal{V}_k,
\qquad
\mathcal{L}_\text{inter} = \min_{\substack{k,\ell \in \{1,\dots,K\}\\ k \neq \ell}} \left\lVert \mu_k - \mu_\ell \right\rVert_2^2
$$

$\mu_k$, $\mathcal{V}_k$: centroid and variance of class $k$ in the latent
space. $\mathcal{L}_\text{intra}$ penalizes intra-cluster dispersion,
$\mathcal{L}_\text{inter}$ rewards inter-cluster separation.

**Pipeline:** an unsupervised spike sorter gives an initial cluster estimate
→ a representative subset (*core*) is sampled from each cluster → the
encoder is trained on this subset with $\mathcal{L}_\text{combined}$ → the
trained encoder is applied to the full dataset and clustered (K-means) to
recover single units.

Architecture: fully connected encoder/decoder, 79-sample input, hidden
layers of 60/40/20 units, latent code size 3.

## Structure

```
common/              shared utilities
dataset_parsing/     loading of simulated and real datasets
neural_networks/
  autoencoder/        AE architectures + combined loss
preprocess/           spike alignment and scaling
validation/
  performance.py       clustering metrics (ARI, AMI, VM, DBS, CHS, SS) + DBSCAN
visualization/         waveform / cluster plots
real_data/
weights/               trained model weights
ae_function.py         core pipeline: data splitting, wavelet decomposition, train/test
run.py                 main entry point
sota_eval.py            comparison against PCA / ICA / Isomap
```

## Data

- **Synthetic:** 95 single-channel simulations (Pedreira et al., 2012), 2–20
  single-unit clusters + multi-unit noise, 24 kHz, 79 samples/waveform.
- **Real:** *in vivo* extracellular recordings (32-channel NeuroNexus A32-tet
  probes), mouse visual cortex.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py --simulation 4 --code_size 3 --clustering kmeans
```

## Metrics

External (require ground truth): Adjusted Rand Index, Adjusted Mutual
Information, V-Measure. Internal: Davies-Bouldin, Calinski-Harabasz,
Silhouette.

## Citation

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

CRAN (UMR 7039, CNRS – Université de Lorraine), in collaboration with the
Institut de Cancérologie de Lorraine and CHRU de Nancy.
