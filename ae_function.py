import os

import sys
sys.path.append(os.path.abspath("/home/elizaveta/Documents/StageElizaveta/Code/Autoencoders-in-Spike-Sorting-main"))

import numpy as np
from numpy import linalg as LA

from scipy.signal import resample
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture
from scipy import fft
import pywt

from dataset_parsing import simulations_dataset as ds
from neural_networks.autoencoder import lstm_input
from neural_networks.autoencoder.autoencoder import AutoencoderModel
from neural_networks.autoencoder.autoencoder_pca_principles.autoencoder_pca_model import AutoencoderPCAModel
from neural_networks.autoencoder.autoencoder_tied import AutoencoderTiedModel
from neural_networks.autoencoder.autoencoder_tied2 import AutoencoderTied2Model
from neural_networks.autoencoder.lstm_autoencoder import LSTMAutoencoderModel
from neural_networks.autoencoder.model_auxiliaries import get_codes
from preprocess.data_scaling import choose_scale
from visualization import scatter_plot
from validation.performance import classification_kmeans, classification_MDN, matrice_confusion, affichage_metrics, classification_DBSCAN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    path = os.path.join(folder, filename)
    model.save_weights(path, overwrite=True)
    print(f"Modèle sauvegardé : {path}")

def load_model(filename, folder=None):
    path = os.path.join(filename,folder)
    print(f"Modèle chargé : {path}")
    return path


# def separation_data(spikes, labels, pourcentage):
#     """
#     Fonction qui sépare les données en apprentissage/test en respectant le pourcentage
#     pour chaque classe.
#
#     Paramètres:
#         spikes (ndarray): matrice des spikes (nb_samples, 79)
#         labels (ndarray): vecteur des labels (nb_samples, 1)
#         pourcentage (float): proportion pour l'apprentissage
#
#     Returns:
#         spikes_train, labels_train: base de données étiquetée d'apprentissage
#         spikes_test, labels_test: base de données étiquetée de test
#     """
#
#     spikes_train, labels_train = [], []
#     spikes_test, labels_test = [], []
#
#     classes = np.unique(labels)
#
#     for c in classes:
#         indices = np.where(labels == c)[0]
#         np.random.shuffle(indices)
#
#         n_train = round(pourcentage * len(indices))
#
#         train_idx = indices[:n_train]
#         test_idx = indices[n_train:]
#
#         spikes_train.append(spikes[train_idx])
#         labels_train.append(labels[train_idx])
#         spikes_test.append(spikes[test_idx])
#         labels_test.append(labels[test_idx])
#
#     spikes_train = np.vstack(spikes_train)
#     labels_train = np.hstack(labels_train)
#     spikes_test = np.vstack(spikes_test)
#     labels_test = np.hstack(labels_test)
#
#     return spikes_train, labels_train, spikes_test, labels_test

def separation_data(spikes, labels, pourcentage):
    n_train = round(pourcentage * len(labels))
    spikes_train = spikes[:n_train, :]
    labels_train = labels[:n_train]
    spikes_test = spikes[n_train:len(labels), :]
    labels_test = labels[n_train:len(labels)]


    return spikes_train, labels_train, spikes_test, labels_test


# def decomposition_ondelettes(signal, wavelet, dim):
#     """
#     Décomposition en ondelettes d’un signal avec reconstruction
#     à partir des seuls coefficients sélectionnés.
#
#     Paramètres:
#         signal (ndarray): un spike (n_points,)
#         wavelet (str): type d’ondelette utilisée ('db4', 'sym6', etc.)
#         dim (int): nombre de coefficients conservés (utilisé pour ind_max)
#
#     Returns:
#         signal (ndarray): signal original
#         rec_signal (ndarray): signal reconstruit à partir des coeffs sélectionnés
#     """
#
#     # Décomposition
#     coeffs = pywt.wavedec(signal, wavelet, level=6)
#     cA, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
#
#     # Isolement des sous-bandes utiles
#     cD3_sel, cD4_sel, cD5_sel, cD6_sel = cD3[3:16], cD4[3:12], cD5[2:11], cD6[2:10]
#     # cD3_sel, cD4_sel, cD5_sel, cD6_sel = cD3, cD4, cD5, cD6
#
#     concat_coeffs = np.concatenate((cD3_sel, cD4_sel, cD5_sel, cD6_sel), axis=None).reshape(1, -1)
#     if dim == 20:
#         ind_max = [1, 8, 24, 4, 37, 33, 7, 19, 15, 36, 27, 18, 5, 35, 6, 16, 17, 25, 26, 34]
#     elif dim == 10:
#         ind_max = [27, 18, 5, 35, 6, 16, 17, 25, 26, 34]
#     else:
#         print("dim doit être 10 ou 20 (indices prédéfinis).")
#     spikes = concat_coeffs[:, ind_max]
#
#     # Création d’un vecteur de coeffs avec zéros partout sauf ind_max
#     spikes_coeffs_zeroed = np.zeros_like(concat_coeffs)
#     for j, idx in enumerate(ind_max):
#         spikes_coeffs_zeroed[:, idx] = spikes[:, j]
#
#     # Recréation des sous-bandes à partir de la version filtrée
#     cD3_new = np.zeros_like(cD3); cD3_new[3:16] = spikes_coeffs_zeroed[0, 0:13]
#     cD4_new = np.zeros_like(cD4); cD4_new[3:12] = spikes_coeffs_zeroed[0, 13:22]
#     cD5_new = np.zeros_like(cD5); cD5_new[2:11] = spikes_coeffs_zeroed[0, 22:31]
#     cD6_new = np.zeros_like(cD6); cD6_new[2:10] = spikes_coeffs_zeroed[0, 31:39]
#
#     # # Recréation des sous-bandes à partir de la version filtrée
#     # offset = 0
#     # cD3_new = np.zeros_like(cD3)
#     # cD3_new[:] = spikes_coeffs_zeroed[0, offset:offset+len(cD3)]
#     # offset += len(cD3)
#     #
#     # cD4_new = np.zeros_like(cD4)
#     # cD4_new[:] = spikes_coeffs_zeroed[0, offset:offset+len(cD4)]
#     # offset += len(cD4)
#     #
#     # cD5_new = np.zeros_like(cD5)
#     # cD5_new[:] = spikes_coeffs_zeroed[0, offset:offset+len(cD5)]
#     # offset += len(cD5)
#     #
#     # cD6_new = np.zeros_like(cD6)
#     # cD6_new[:] = spikes_coeffs_zeroed[0, offset:offset+len(cD6)]
#
#
#     # Tous les autres coeffs mis à zéro
#     coeffs_new = [np.zeros_like(cA), cD6_new, cD5_new, cD4_new, cD3_new,
#                   np.zeros_like(cD2), np.zeros_like(cD1)]
#
#     # Reconstruction
#     rec_signal = pywt.waverec(coeffs_new, wavelet)
#     rec_signal = rec_signal[:len(signal)]  # ajuster à la longueur originale
#
#     print(f"Shape signal original: {signal.shape}\nShape signal reconstruit: {rec_signal.shape}")
#     return signal, rec_signal






def decomposition_ondelettes(data, wavelet, dim):
    print('Début de la décomposition en ondelettes des signaux:')
    spikes = []
    for signal in data:
        cA, cD6, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(signal, 'sym6', level=6)
        cD3, cD4, cD5, cD6 = cD3[3:16], cD4[3:12], cD5[2:11], cD6[2:10] # isolation des coefficients qui n'ont pas subi l'effet de bord
        coeffs = np.concatenate((cD3, cD4, cD5, cD6), axis=None) # isolation des coefficients entre 180Hz et 12KHz
        spikes.append(coeffs)
    spikes = np.vstack(spikes)
    # var = np.var(spikes, axis=0)
    # ind_max=np.argsort(var)[-20:]
    if dim==20:
        ind_max=[ 1,  8, 24,  4, 37, 33,  7, 19, 15, 36, 27, 18,  5, 35,  6, 16, 17, 25, 26, 34]
        # ind_max=np.argsort(var)[-20:]
    if dim==10:
        ind_max=[27, 18,  5, 35,  6, 16, 17, 25, 26, 34]
        # ind_max=np.argsort(var)[-10:]
    spikes = spikes[:,ind_max]
    print('La décomposition en ondelettes est términée: ')
    print(f'Nombre de spikes: {spikes.shape[0]} \nNombre de coefficients pour chaque spike: {spikes.shape[1]}')
    return spikes, ind_max



def run_autoencoder_train(simulation_number, data, labels, code_size, output_activation, loss_function, scale=None, shuff=True, noNoise=False, decomp_ondelettes=None, nr_epochs=300, dropout=0.0, weight_init='glorot_uniform', learning_rate=0.0001, verbose=1, saveWeights=None, loadWeights=None):
    
    print("ENTRAINEMENT DU MODÈLE")
    print(" ")
    """ Récuperation des spikes et de leurs labels """
    print("ÉTAPE 1: Extraction des données d'apprentissage:")
    if simulation_number == None:
        if data is None and labels is None:
            print("La base d'apprentissage n'est pas choisie.")
        else:
            spikes_train = data
            labels_train = labels
    elif simulation_number == "univ":
            spikes_train = np.load(f'spikes_apprentissage_generalise.npy', allow_pickle=True)
            labels_train = np.load(f'labels_apprentissage_generalise.npy', allow_pickle=True)
    else:
        if data is None and labels is None:
            spikes_train, labels_train, _ = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=True)
        else:
            spikes_train = data
            labels_train = labels
    print(f"Les données d'apprentissage sont extraites: il y a {spikes_train.shape[0]} signaux constitué de {spikes_train.shape[1]} points issus de {len(np.unique(labels_train))-1} neurones distincts et du bruit de fond.")
    print("")

    """ Pré-processing des spikes """
    print("ÉTAPE 2: Pré-processing des données d'apprentissage:")
    if noNoise == True:
        spikes_train = spikes_train[labels_train != 0]
        labels_train = labels_train[labels_train!=0]
        print(f"Les données d'apprentissage sont débruités: il y a {spikes_train.shape[0]} signaux issus de {len(np.unique(labels_train))} neurones distincts.")
    if shuff == True:
        spikes_train, labels_train = shuffle(spikes_train, labels_train, random_state=None)
        print("Les données d'apprentissage sont mélangées pour un meilleur apprentissage.")
    if scale != None:
        spikes_train = choose_scale(spikes_train, scale)
        print(f"Les données d'apprentissage sont rédimensionnées sur l'echelle {scale}.")
    if decomp_ondelettes != None:
        spikes_train, _ = decomposition_ondelettes(spikes_train, 'sym6',  decomp_ondelettes)
        print(f"Les données d'apprentissage sont décomposées en ondelettes: chaque spike est défini par {decomp_ondelettes} coefficients.")
    print("")

    """ Définition architecture du modèle """
    print("ÉTAPE 3: Définition de l'architecture de l'autoencodeur:")
    #ae_layers = [40,20,10] pour reeldata
    ae_layers = [60,40,20]
    print(ae_layers)
    if decomp_ondelettes == 20:
        # ae_layers = [18, 13, 8, 5]
        ae_layers = [18,15,13]
        # ae_layers = [18, 15, 13, 11, 9, 5]
        # ae_layers = [18, 15, 13, 5]
    if decomp_ondelettes == 10:
        ae_layers = [8,5]
    autoencoder = AutoencoderModel(input_size=len(spikes_train[0]),
                                    encoder_layer_sizes=ae_layers,
                                    decoder_layer_sizes=ae_layers,
                                    code_size=code_size,
                                    output_activation=output_activation,
                                    loss_function=loss_function,
                                    dropout=dropout,
                                    initializer=weight_init)

    print(f"L'autoencodeur est défini; il est constitué: \n - d'une couche d'entrée de {len(spikes_train[0])} neurones, \n - de {len(ae_layers)} couches d'encodeur avec {', '.join(map(str, ae_layers))} neurones respectivement, \n - d'un code latent de dimension {code_size}, \n - de {len(ae_layers)} couches de décodeur avec {', '.join(map(str, ae_layers[::-1]))} neurones respectivement \n - et d'une couche de sortie de {len(spikes_train[0])} neurones.\nLa fonction d’activation de la couche de sortie est {output_activation}, la fonction de coût utilisée est {loss_function}, les poids sont initialisés selon la méthode {weight_init}, un dropout de {dropout*100}% est appliqué après chaque batch, et l’entrainement sera réalisé sur {nr_epochs} époques avec un taux d’apprentissage de {learning_rate}.")
    print("")

    """ Entrainement du modèle"""
    print("ÉTAPE 4: Entrainement du modèle:")

    if loadWeights is not None:
        encoder, autoencoder = autoencoder.return_encoder()
        autoencoder.load_weights(loadWeights)
        print(f"Le modèle est déja entrainé; il est chargé depuis {loadWeights}")
    else:
        if loss_function=='mse' or loss_function=='mse_modified':
            autoencoder.train_autoencodeur(spikes_train, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)
            encoder, autoencoder = autoencoder.return_encoder()
        else:
            autoencoder.train_encodeur(spikes_train, labels_train, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)
            encoder, autoencoder = autoencoder.return_encoder()
        if saveWeights and save_weights_path is not None:
            autoencoder.save_weights(save_weights_path, overwrite=True)
        print(f"L'entrainement est terminé.")
    print("")

    """ Phase de validation """
    print("ÉTAPE 5: Validation du modèle:")
    features_validation = encoder.predict(spikes_train)
    labels_test, labels_reels, labels_predits = classification_kmeans(features_validation, labels_train)
    print(f'Labels rééls (avant la rélabelisation): {labels_test},\nLabels réels rélabelisés: {labels_reels},\nLabels predits: {labels_predits}')
    metrics = affichage_metrics(features_validation, labels_reels, labels_predits)
    if metrics[0]>=0.95 and metrics[1]>=0.95 and metrics[2]>=0.95:
        print("Entrainement est validé: la classification est performante.")
    else:
        print("Entrainement n'est pas validé: la classification n'est pas performante.")
    print(" ")


    """ Etude de la variance des codes """
    features_validation = np.array(features_validation)
    variance_features = np.var(features_validation, axis=0)
    labels = np.unique(labels_train)
    nb_labels = len(labels)
    variances_par_classe =[]
    variances_moyennes=[]
    for label in labels:
        indices = np.where(labels_train == label)
        class_features = features_validation[indices,:]
        var_dim = np.var(class_features, axis=1)
        var_moy=np.mean(var_dim)
        variances_par_classe.append(var_dim)
        variances_moyennes.append(var_moy)

    return encoder, autoencoder, metrics, variance_features, variances_par_classe, variances_moyennes










def run_autoencoder_test(simulation_test, data, labels, encoder, classification, code_size, output_activation, loss_function, noNoise=False, scale=None,  savePlot=False, doPlot = True, doPlotWaveforms=False, doPlotMetrics=False, simulation_train=None, decomp_ondelettes = None):

    print("TEST DU MODÈLE")
    print(" ")

    """ Récuperation des spikes et de leurs labels """
    if simulation_test == None:
        if data is None and labels is None:
            print("La base de test n'est pas choisie")
        else:
            spikes_test = data
            labels_test = labels
    elif simulation_test == "all":
        spikes_test = np.load("spikes_apprentissage_bis.npy", allow_pickle=True)
        labels_test = np.load("labels_apprentissage_bis.npy", allow_pickle=True)
    else:
        if data is None and labels is None:
            spikes_test, labels_test, _ = ds.get_dataset_simulation(simNr=simulation_test, align_to_peak=True)
        else:
            spikes_test = data
            labels_test = labels

    """ Pré-processing des spikes """
    if noNoise == True:
        spikes_test = spikes_test[labels_test != 0]
        labels_test = labels_test[labels_test!=0]
    if scale != None:
        spikes_test = choose_scale(spikes_test, scale)
    if decomp_ondelettes != None:
        spikes_test, _ = decomposition_ondelettes(spikes_test, 'sym6',  decomp_ondelettes)
      

    """ Prédictions des codes latents associées aux spikes de test"""
    
    features_test = encoder.predict(spikes_test)

    """ Clustering des spikes de test """

    if classification=="k-means":
        labels_test, labels_reels, labels_predits = classification_kmeans(features_test, labels_test)
        print(f'labels réels (avant relabélisation) : {labels_test}\n labels réels: {labels_reels}\n labels predits: {labels_predits}')
    if classification=="MDN":
        labels_test, labels_reels, labels_predits = classification_MDN(features_test, labels_test)
        print(f'labels réels (avant relabélisation): {labels_test}\n labels réels: {labels_reels}\n labels predits: {labels_predits}')
    if classification=="DBSCAN":
        labels_test, labels_reels, labels_predits = classification_DBSCAN(features_test, labels_test)
        print(f'labels réels (avant relabélisation): {labels_test}\n labels réels: {labels_reels}\n labels predits: {labels_predits}')


    """ Critères de performance et matrice de confusion """

    metrics = affichage_metrics(features_test,labels_reels, labels_predits)
    noms_metrics=np.array(["ARI", "AMI", "VM", "DBS", "CHS", "SS"])

    # Critères de classification
    # metrics = affichage_metrics(labels_reels, labels_predits)
    # noms_metrics=np.array(["Precision", "Accuracy", "F1-score", "Recall"])

    # matrice de confusion
    disp = matrice_confusion(labels_reels,labels_predits)

    if doPlotMetrics == True:
        disp.plot()
        plt.show()

        plt.figure()
        bars = plt.bar(noms_metrics, metrics)
        plt.title(f"Critères de performances ({loss_function})")

        for bar, value in zip(bars, metrics):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.2f}",
                    ha='center', va='bottom', fontsize=10)    

        plt.show()


    """ VISUALISATION CLUSTERING"""

    if doPlot == True:
        if code_size == 2 or code_size == 3:
            if simulation_train=="univ":
                scatter_plot.plot(f'Modèle universel - Test sur Sim{simulation_test} - Loss: {loss_function} - labels réels', features_test, labels_reels, marker='o')
                scatter_plot.plot(f'Modèle universel - Test sur Sim{simulation_test} -  Loss: {loss_function} - labels {classification}', features_test, labels_predits, marker='o')
            else:
                scatter_plot.plot(f'Train on Sim{simulation_train} - Test on Sim{simulation_test} - Loss: {loss_function} - labels GT', features_test, labels_reels, marker='o')
                scatter_plot.plot(f'Train on Sim{simulation_train} - Test on Sim{simulation_test} -  Loss: {loss_function} - labels {classification}', features_test, labels_predits, marker='o')

            if scale == 'divide_amplitude':
                amplitudes = np.reshape(amplitudes, (len(amplitudes), -1))
                autoencoder_features_add_amplitude = np.hstack((features_test, amplitudes))
                scatter_plot.plot('GT' + str(len(features_test)), autoencoder_features_add_amplitude, labels_reels, marker='o')
            plt.show()

        if code_size > 3:
            pca_2d = PCA(n_components=2)
            autoencoder_features_2d = pca_2d.fit_transform(features_test)
            print(f'Ratio ACP: {pca_2d.explained_variance_ratio_}')

            if simulation_train=="univ":
                scatter_plot.plot(f'Modèle universel - Test sur Sim{simulation_test} - Loss: {loss_function} - labels réels', autoencoder_features_2d, labels_reels, marker='o')
                scatter_plot.plot(f'Modèle universel - Test sur Sim{simulation_test} -  Loss: {loss_function} - labels {classification}', autoencoder_features_2d, labels_predits, marker='o')
            else:
                scatter_plot.plot(f'Apprentissage sur Sim{simulation_train} - Test sur Sim{simulation_test} - Loss: {loss_function} - labels réels', autoencoder_features_2d, labels_reels, marker='o')

                scatter_plot.plot(f'Apprentissage sur Sim{simulation_train} - Test sur Sim{simulation_test} -  Loss: {loss_function} - labels {classification}', autoencoder_features_2d, labels_predits, marker='o')

            if scale == 'divide_amplitude':
                amplitudes = np.reshape(amplitudes, (len(amplitudes), -1))
                autoencoder_features_add_amplitude = np.hstack((autoencoder_features, amplitudes))
                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features_add_amplitude, labels_reels, marker='o')
            plt.show()

    """ VISUALISATION WAVEFORM """
    if doPlotWaveforms == True:
        waveforms = np.load("unique_waveforms.npy", allow_pickle=True)
        classes = np.unique(labels_test)
        for label in np.unique(labels_predits):
            label_waveform=classes[label]
            plt.figure(figsize=(10, 6))
            plt.title(f"Superposition de la moyenne des spikes et de la waveform pour la classe {label_waveform}")
            indices = np.where(labels_predits == label)[0]
            spikes_par_class = spikes_test[indices, :]
            spike_moyen = np.mean(spikes_par_class, axis=0)
            spike_moyen = spike_moyen / LA.norm(spike_moyen)
            waveform = waveforms[label_waveform - 1]
            waveform = resample(waveform, 79)
            waveform = waveform / LA.norm(waveform)
            peak_waveform = np.argmax(waveform) 
            peak_spike = np.argmax(spike_moyen)
            shift = peak_spike - peak_waveform
            waveform = np.roll(waveform, shift)
            plt.plot(spike_moyen, label="Moyenne des spikes predites")
            plt.plot(waveform, label="Waveform")
            plt.legend(loc='best', fontsize=8)
            plt.grid(True)
            plt.tight_layout()
    plt.show()

    return metrics, labels_predits


