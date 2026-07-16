# Autoencoders-in-Spike-Sorting

Autoencoders, a type of neural network that allows for unsupervised learning,
can be used for feature extraction in spike sorting.

This repository extends the original implementation from the study published
in PLOS One:

- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282810
- DOI: 10.1371/journal.pone.0282810
- Original repository: https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting

## Citation

We would appreciate it if you cite the paper when you use this work:

```
E. Kompaniets, S. Le Cam, and R. Ranta,
"Improving Feature Extraction for Spike Sorting Using a Custom Loss Function for an Autoencoder,"
in Proc. 19th International Joint Conference on Biomedical Engineering Systems and Technologies (BIOSTEC 2026), 2026.
```

## Setup

1. Clone the original repository:

```bash
git clone https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting.git
```

2. Install the dependencies listed in the original repository's
   `requirements.txt`.

3. Download the data:

   - Synthetic data:
     https: https://www.kaggle.com/datasets/ardeleanrichard/simulationsdataset
   - Real data:
     https://www.kaggle.com/datasets/ardeleanrichard/realdata
     or the `real_data` folder of the original repository

   Set the path to the `DATA` folder in `constants.py`. Recommended
   structure:

```
DATA/
├── TINS/
│   └── M045_009/      : real data files
└── SIMULATIONS/        : synthetic data files
```

4. Add our files into the cloned repository, at the following locations
   (same folder structure as the original repository):

```
Autoencoders-in-Spike-Sorting/
├── ae_function.py
├── ae_function_cb.py
├── main_sim.py
├── main_real.py
├── neural_networks/
│   └── autoencoder/
│       └── autoencoder.py
├── validation/
│   └── performance.py
└── visualization/
    └── scatter_plot.py
```

## Run

```bash
python main_sim.py
python main_real.py
```

## Contact

If you have any questions, feel free to contact us:

- Steven Le Cam — steven.le-cam@univ-lorraine.fr
- Radu Ranta — radu.ranta@univ-lorraine.fr
