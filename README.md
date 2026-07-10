# Autoencoders-in-Spike-Sorting (extension)

Additional scripts built on top of the original Autoencoders-in-Spike-Sorting
repository, used to reproduce our results on both simulated and real data.

This work builds on the study published in PLOS One:

- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282810
- DOI: 10.1371/journal.pone.0282810
- Original repository: https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting

## Setup

1. Clone the original repository:

```bash
git clone https://github.com/ArdeleanRichard/Autoencoders-in-Spike-Sorting.git
```

2. Install the dependencies listed in the original repo's `requirements.txt`.

3. Download the data as described in the original repo's README (synthetic
   simulations and real data), and set the `DATA` path in `constants.py`.

4. Add our files into the cloned repository, keeping the same folder
   structure:

```
Autoencoders-in-Spike-Sorting/
├── ae_function.py
├── ae_function_cb.py
├── main_sim.py
├── main_real.py
```

## Run

```bash
python main_sim.py
python main_real.py
```

## Contact

<!-- add your contact info here -->
