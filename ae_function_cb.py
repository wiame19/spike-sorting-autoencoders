
import os

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from scipy.signal import resample
import pywt

from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from dataset_parsing import simulations_dataset as ds
from neural_networks.autoencoder.autoencoder import AutoencoderModel
from preprocess.data_scaling import choose_scale
from visualization import scatter_plot
from validation.performance import (
    classification_kmeans,
    classification_MDN,
    classification_DBSCAN,
    matrice_confusion,
    affichage_metrics,
)

# Gestion des dossiers / sauvegarde

def create_dossier(simulation_number):
    dir_name = f"SIM{simulation_number}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Dossier créé : {dir_name}")
    else:
        print(f"Dossier déjà existant : {dir_name}")
    return dir_name


def create_dossier_article(simulation_number):
    dir_name = f"ARTICLE_SIM{simulation_number}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Dossier créé : {dir_name}")
    else:
        print(f"Dossier déjà existant : {dir_name}")
    return dir_name


def save_figure(folder, filename="figure.png", dpi=300):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée : {path}")


def save_model(model, folder, filename):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    model.save_weights(path, overwrite=True)
    print(f"Modèle sauvegardé : {path}")


def load_model(folder, filename):
    path = os.path.join(folder, filename)
    print(f"Modèle chargé : {path}")
    return path

# Préparation des données

def separation_data(spikes, labels, pourcentage):
    n_train = round(pourcentage * len(labels))

    spikes_train = spikes[:n_train, :]
    labels_train = labels[:n_train]

    spikes_test = spikes[n_train:len(labels), :]
    labels_test = labels[n_train:len(labels)]

    return spikes_train, labels_train, spikes_test, labels_test


def decomposition_ondelettes(data, wavelet, dim):
    print("Début de la décomposition en ondelettes des signaux:")
    spikes = []
    for signal in data:
        cA, cD6, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(signal, wavelet, level=6)
        # isolation des coefficients qui n'ont pas subi l'effet de bord
        # (bande de fréquence entre ~180Hz et 12KHz)
        cD3, cD4, cD5, cD6 = cD3[3:16], cD4[3:12], cD5[2:11], cD6[2:10]
        coeffs = np.concatenate((cD3, cD4, cD5, cD6), axis=None)
        spikes.append(coeffs)
    spikes = np.vstack(spikes)

    if dim == 20:
        ind_max = [1, 8, 24, 4, 37, 33, 7, 19, 15, 36, 27, 18, 5, 35, 6, 16, 17, 25, 26, 34]
    elif dim == 10:
        ind_max = [27, 18, 5, 35, 6, 16, 17, 25, 26, 34]
    else:
        raise ValueError("dim doit être 10 ou 20 (indices prédéfinis).")

    spikes = spikes[:, ind_max]
    print("La décomposition en ondelettes est terminée.")
    print(f"Nombre de spikes: {spikes.shape[0]} | Coefficients par spike: {spikes.shape[1]}")
    return spikes, ind_max


# Entraînement / test de l'autoencodeur

def run_autoencoder_train(simulation_number, data, labels, code_size, output_activation,
                           loss_function, scale=None, shuff=True, noNoise=False,
                           decomp_ondelettes=None, nr_epochs=300, dropout=0.0,
                           weight_init='glorot_uniform', learning_rate=0.0001,
                           verbose=1, saveWeights=None, loadWeights=None):

    print("ENTRAINEMENT DU MODÈLE")

    # --- Étape 1 : extraction des données d'apprentissage ---
    print("ÉTAPE 1: Extraction des données d'apprentissage")
    if simulation_number is None:
        if data is None and labels is None:
            raise ValueError("La base d'apprentissage n'est pas choisie.")
        spikes_train, labels_train = data, labels
    elif simulation_number == "univ":
        spikes_train = np.load("spikes_apprentissage_generalise.npy", allow_pickle=True)
        labels_train = np.load("labels_apprentissage_generalise.npy", allow_pickle=True)
    else:
        if data is None and labels is None:
            spikes_train, labels_train, _ = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=True)
        else:
            spikes_train, labels_train = data, labels
    print(f"{spikes_train.shape[0]} signaux de {spikes_train.shape[1]} points, "
          f"{len(np.unique(labels_train)) - 1} neurones distincts + bruit de fond.")

    # --- Étape 2 : pré-traitement ---
    print("ÉTAPE 2: Pré-traitement des données d'apprentissage")
    if noNoise:
        spikes_train = spikes_train[labels_train != 0]
        labels_train = labels_train[labels_train != 0]
        print(f"Débruitage : {spikes_train.shape[0]} signaux, {len(np.unique(labels_train))} neurones distincts.")
    if shuff:
        spikes_train, labels_train = shuffle(spikes_train, labels_train, random_state=None)
    if scale is not None:
        spikes_train = choose_scale(spikes_train, scale)
    if decomp_ondelettes is not None:
        spikes_train, _ = decomposition_ondelettes(spikes_train, 'sym6', decomp_ondelettes)
        print(f"Décomposition en ondelettes : {decomp_ondelettes} coefficients par spike.")

    # --- Étape 3 : architecture de l'autoencodeur ---
    print("ÉTAPE 3: Définition de l'architecture de l'autoencodeur")
    ae_layers = [60, 40, 20]
    if decomp_ondelettes == 20:
        ae_layers = [18, 15, 13]
    if decomp_ondelettes == 10:
        ae_layers = [8, 5]

    autoencoder = AutoencoderModel(
        input_size=len(spikes_train[0]),
        encoder_layer_sizes=ae_layers,
        decoder_layer_sizes=ae_layers,
        code_size=code_size,
        output_activation=output_activation,
        loss_function=loss_function,
        dropout=dropout,
        initializer=weight_init,
    )

    # --- Étape 4 : entraînement ---
    print("ÉTAPE 4: Entrainement du modèle")
    if loadWeights is not None:
        encoder, autoencoder = autoencoder.return_encoder()
        autoencoder.load_weights(loadWeights)
        print(f"Modèle déjà entrainé, chargé depuis {loadWeights}")
    else:
        if loss_function in ('mse', 'mse_modified'):
            autoencoder.train_autoencodeur(spikes_train, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)
        else:
            autoencoder.train_encodeur(spikes_train, labels_train, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)
        encoder, autoencoder = autoencoder.return_encoder()

        if saveWeights is not None:
            autoencoder.save_weights(saveWeights, overwrite=True)
        print("Entrainement terminé.")

    # --- Étape 5 : validation du modèle ---
    print("ÉTAPE 5: Validation du modèle")
    features_validation = encoder.predict(spikes_train)
    _, labels_reels, labels_predits = classification_kmeans(features_validation, labels_train)
    metrics = affichage_metrics(features_validation, labels_reels, labels_predits)
    if metrics[0] >= 0.95 and metrics[1] >= 0.95 and metrics[2] >= 0.95:
        print("Entrainement validé : classification performante.")
    else:
        print("Entrainement non validé : classification peu performante.")

    # --- Étude de la variance du code latent ---
    features_validation = np.array(features_validation)
    variance_features = np.var(features_validation, axis=0)
    unique_labels = np.unique(labels_train)
    variances_par_classe = []
    variances_moyennes = []
    for label in unique_labels:
        indices = np.where(labels_train == label)
        class_features = features_validation[indices, :]
        var_dim = np.var(class_features, axis=1)
        variances_par_classe.append(var_dim)
        variances_moyennes.append(np.mean(var_dim))

    return encoder, autoencoder, metrics, variance_features, variances_par_classe, variances_moyennes


def run_autoencoder_test(simulation_test, data, labels, encoder, classification, code_size,
                          output_activation, loss_function, noNoise=False, scale=None,
                          doPlot=True, doPlotWaveforms=False, doPlotMetrics=False,
                          simulation_train=None, decomp_ondelettes=None):

    print("TEST DU MODÈLE")

    # --- Chargement des données de test ---
    if simulation_test is None:
        if data is None and labels is None:
            raise ValueError("La base de test n'est pas choisie.")
        spikes_test, labels_test = data, labels
    elif simulation_test == "all":
        spikes_test = np.load("spikes_apprentissage_bis.npy", allow_pickle=True)
        labels_test = np.load("labels_apprentissage_bis.npy", allow_pickle=True)
    else:
        if data is None and labels is None:
            spikes_test, labels_test, _ = ds.get_dataset_simulation(simNr=simulation_test, align_to_peak=True)
        else:
            spikes_test, labels_test = data, labels

    # --- Pré-traitement (doit correspondre à l'entraînement) ---
    if noNoise:
        spikes_test = spikes_test[labels_test != 0]
        labels_test = labels_test[labels_test != 0]
    if scale is not None:
        spikes_test = choose_scale(spikes_test, scale)
    if decomp_ondelettes is not None:
        spikes_test, _ = decomposition_ondelettes(spikes_test, 'sym6', decomp_ondelettes)

    # --- Prédiction des codes latents ---
    features_test = encoder.predict(spikes_test)

    # --- Clustering ---
    if classification == "k-means":
        _, labels_reels, labels_predits = classification_kmeans(features_test, labels_test)
    elif classification == "MDN":
        _, labels_reels, labels_predits = classification_MDN(features_test, labels_test)
    elif classification == "DBSCAN":
        _, labels_reels, labels_predits = classification_DBSCAN(features_test, labels_test)
    else:
        raise ValueError(f"Méthode de classification inconnue : {classification}")

    # --- Critères de performance ---
    metrics = affichage_metrics(features_test, labels_reels, labels_predits)
    noms_metrics = np.array(["ARI", "AMI", "VM", "DBS", "CHS", "SS"])

    if doPlotMetrics:
        disp = matrice_confusion(labels_reels, labels_predits)
        disp.plot()
        plt.show()

        plt.figure()
        bars = plt.bar(noms_metrics, metrics)
        plt.title(f"Critères de performances ({loss_function})")
        for bar, value in zip(bars, metrics):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}",
                      ha='center', va='bottom', fontsize=10)
        plt.show()

    # --- Visualisation des codes latents ---
    if doPlot:
        if code_size in (2, 3):
            features_plot = features_test
        else:
            pca_2d = PCA(n_components=2)
            features_plot = pca_2d.fit_transform(features_test)
            print(f"Ratio ACP : {pca_2d.explained_variance_ratio_}")

        if simulation_train == "univ":
            titre_gt = f"Modèle universel - Test sur Sim{simulation_test} - Loss: {loss_function} - labels réels"
            titre_pred = f"Modèle universel - Test sur Sim{simulation_test} - Loss: {loss_function} - labels {classification}"
        else:
            titre_gt = f"Train on Sim{simulation_train} - Test on Sim{simulation_test} - Loss: {loss_function} - labels GT"
            titre_pred = f"Train on Sim{simulation_train} - Test on Sim{simulation_test} - Loss: {loss_function} - labels {classification}"

        scatter_plot.plot(titre_gt, features_plot, labels_reels, marker='o')
        scatter_plot.plot(titre_pred, features_plot, labels_predits, marker='o')

        if scale == 'divide_amplitude':
            amplitudes = np.amax(spikes_test, axis=1).reshape(-1, 1)
            features_plot_ampl = np.hstack((features_plot, amplitudes))
            scatter_plot.plot('GT' + str(len(features_plot_ampl)), features_plot_ampl, labels_reels, marker='o')

        plt.show()

    # --- Visualisation des waveforms moyennes par classe prédite ---
    if doPlotWaveforms:
        waveforms = np.load("unique_waveforms.npy", allow_pickle=True)
        classes = np.unique(labels_test)
        for label in np.unique(labels_predits):
            label_waveform = classes[label]
            plt.figure(figsize=(10, 6))
            plt.title(f"Moyenne des spikes vs waveform - classe {label_waveform}")

            indices = np.where(labels_predits == label)[0]
            spike_moyen = np.mean(spikes_test[indices, :], axis=0)
            spike_moyen = spike_moyen / LA.norm(spike_moyen)

            waveform = resample(waveforms[label_waveform - 1], 79)
            waveform = waveform / LA.norm(waveform)
            shift = np.argmax(spike_moyen) - np.argmax(waveform)
            waveform = np.roll(waveform, shift)

            plt.plot(spike_moyen, label="Moyenne des spikes prédits")
            plt.plot(waveform, label="Waveform")
            plt.legend(loc='best', fontsize=8)
            plt.grid(True)
            plt.tight_layout()
        plt.show()

    return metrics, labels_predits