## SCRIPT DONNÉES RÉELLES M045

import os
import struct

import sys
sys.path.append(os.path.abspath("/home/elizaveta/Documents/StageElizaveta/Code/Autoencoders-in-Spike-Sorting-main"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from scipy import fft
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pywt

import random
import tensorflow as tf

from ae_function import run_autoencoder_train, run_autoencoder_test, create_dossier, create_dossier_article, save_figure, save_model, load_model
from dataset_parsing import simulations_dataset as ds
from visualization import scatter_plot
from validation.performance import reassign_labels, matrice_confusion, affichage_metrics, affichage_metrics_class

from sklearn.preprocessing import LabelEncoder
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.chdir("/home/elizaveta")

DATASET_PATH = os.path.expanduser("~/Documents/StageWiame/git/realdata/")
WAVEFORM_LENGTH = 58
NR_CHANNELS = 33

def parse_ssd(path):
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        if fp.endswith(".ssd"):
            lines = np.array([l.rstrip() for l in open(fp).readlines()])
            idx = np.where(lines == 'Number of spikes in each unit:')[0][0]
            count = 1
            while str(lines[idx+count]).isdigit():
                count += 1
            spu = lines[idx+1:idx+count].astype('int')
            elec = np.array([i.strip('El_') for i in lines
                             if str(i).startswith('El_')]).astype('int')
            return spu, elec

def read_bin(filename, data_type):
    with open(filename, 'rb') as f:
        data = []
        val = f.read(4)
        data.append(struct.unpack(data_type, val)[0])
        while val:
            val = f.read(4)
            try:
                data.append(struct.unpack(data_type, val)[0])
            except:
                break
    return np.array(data)

def sep_by_unit(spu, data, length):
    res, s = [], 0
    for n in spu:
        res.append(data[s*length:(s+n)*length])
        s += n
    return np.array(res, dtype=object)

def units_by_ch(unit_electrode, data, length, nr_ch):
    units  = [[] for _ in range(nr_ch)]
    labels = [[] for _ in range(nr_ch)]
    for u, ch in enumerate(unit_electrode):
        wf = np.reshape(data[u], (-1, length))
        units[ch-1].extend(wf.tolist())
        labels[ch-1].extend([u]*len(wf))
    reset = []
    for lab in labels:
        if lab:
            lab = np.array(lab)
            lab = lab - lab.min() + 1
            reset.append(lab.tolist())
        else:
            reset.append([])
    return units, reset

##fonction bestkmeans
def best_kmeans_silhouette(features, k, n_try=20):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    best_ss  = -2
    best_lbl = None
    best_km  = None

    for _ in range(n_try):
        km  = KMeans(n_clusters=k, n_init=1, random_state=None)
        lbl = km.fit_predict(features_scaled)
        if len(np.unique(lbl)) < 2:
            continue

        try:
            ss = silhouette_score(features_scaled, lbl)
        except:
            ss = -2

        if ss > best_ss:
            best_ss  = ss
            best_lbl = lbl
            best_km  = km

    return best_lbl, best_km, best_ss

print("=== CHARGEMENT DONNÉES RÉELLES M045 ===")
spu, unit_electrode = parse_ssd(DATASET_PATH)
wf_file = None
for fn in os.listdir(DATASET_PATH):
    if fn.endswith(".ssduw"):
        wf_file = os.path.join(DATASET_PATH, fn)

waveforms  = read_bin(wf_file, 'f')
wf_by_unit = sep_by_unit(spu, waveforms, WAVEFORM_LENGTH)
units_in_ch, labels_ch = units_by_ch(
    unit_electrode, wf_by_unit, WAVEFORM_LENGTH, NR_CHANNELS)

print(f"Données réelles chargées — {len(spu)} unités, {int(np.sum(spu))} spikes")
print("-"*50)


RECORDINGS = [17,4,6,26]

for ch in RECORDINGS:
    dossier_ch = f"REALDATA_CH{ch}"
    if not os.path.exists(dossier_ch):
        os.makedirs(dossier_ch)
        print(f"Dossier créé : {dossier_ch}/")

iter = 0
while iter<100:

    iter+=1

    for i in range(len(RECORDINGS)):
        SIM_TRAIN = RECORDINGS[i]

        dossier_check = f"REALDATA_CH{SIM_TRAIN}"
        fichier_iter = f"{dossier_check}/metrics_ch{SIM_TRAIN}_iteration{iter}"
        if os.path.exists(fichier_iter):
            print(f"Itération {iter} déjà faite pour canal {SIM_TRAIN}, on passe à la suivante.")
            continue

        print(" ")
        print(f"CANAL {SIM_TRAIN} — DONNÉES RÉELLES M045")

        """ PARTIE 1 -------------------- APPRENTISSAGE NON SUPERVISÉ PAR AUTOENCODEUR AVEC MSE (on cherche à récuperer les classes des spikes et construire un ensemble d'apprentissage supervisé par un second autoencodeur -------------------- """

        SIM_TEST = SIM_TRAIN
        dossier = f"REALDATA_CH{SIM_TRAIN}"

        spikes = np.array(units_in_ch[SIM_TRAIN-1])
        labels = np.array(labels_ch[SIM_TRAIN-1])
        k = len(np.unique(labels))
        print(f"  {len(spikes)} spikes, {k} unités")

        """ Décomposition en ondelettes """
        DIM_DECOMP = None

        """Nombre d'époques"""
        EPOCHS = 300
        """Dimension du code"""
        CODE = 2
        """Fonction de perte"""
        loss_function = 'mse'
        """ Taux d'apprentissage"""
        learning_rate=0.0001

        """Méthode de classification"""
        CLASS ='k-means'

        """Chargement du modèle existant"""
        loadWeights = None

        """Création et entrainement du modèle"""
        encoder_mse, autoencoder_mse, perf_val, variances_code, variances_par_classe, variances_moyennes = run_autoencoder_train(
                simulation_number=SIM_TRAIN, data=spikes, labels=labels,
                decomp_ondelettes=DIM_DECOMP, code_size=CODE, output_activation='linear',
                loss_function=loss_function, scale='minmax', shuff=True, noNoise=False,
                nr_epochs=EPOCHS, learning_rate=learning_rate, dropout=0.0,
                verbose=1, loadWeights=loadWeights)

        """Enregistrement du modèle"""
        save_model(autoencoder_mse, dossier, f'model_ch{SIM_TRAIN}_{loss_function}_code{CODE}_epochs{EPOCHS}_iteration{iter}.weights.h5')

        """ Validation du modèle : for k=1:20, classif=kmeans, SS=silhouette """

        spikes = np.array(units_in_ch[SIM_TRAIN-1])
        features_mse = encoder_mse.predict(spikes, verbose=0)
        if np.any(np.isnan(features_mse)):
            print("  Phase 1 : features contiennent NaN, on passe à l'itération suivante.")
            continue

        lbl_mse, kmeans, ss_mse = best_kmeans_silhouette(features_mse, k, n_try=100)

        if lbl_mse is None:
            print("  Phase 1 : K-Means échoué, on passe à l'itération suivante.")
            continue

        print(f"  Phase 1 — Silhouette Score = {ss_mse:.4f}")
        labels = np.array(labels_ch[SIM_TRAIN-1])
        le = LabelEncoder()
        labels_transformed = le.fit_transform(labels)
        labels_transformed = np.array(labels_transformed)
        labels_predits = reassign_labels(labels_transformed, lbl_mse)
        metrics_MSE = affichage_metrics_class(labels_transformed, labels_predits)
        metrics_MSE.extend(affichage_metrics(features_mse, labels_transformed, labels_predits))
        print(metrics_MSE)

        #affichage matrice de confusion
        # disp = matrice_confusion(labels_transformed,labels_predits)
        # disp.plot()
        # plt.show()
        # save_figure(dossier, f"matrice_confusion_{loss_function}")

        #affichage criteres de performances
        # plt.figure()
        # noms_metrics=np.array(["ARI", "AMI", "VM", "DBS", "CHS", "SS"])
        # bars = plt.bar(noms_metrics, metrics_MSE)
        # plt.title("Critères de performances (MSE)")
        # for bar, value in zip(bars, metrics_MSE):
        #     plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.2f}",
        #             ha='center', va='bottom', fontsize=10)
        # plt.show()
        # save_figure(dossier, f"criteres_{loss_function}")

        #affichage répartition des features dans l'espace latent — Phase 1 MSE
        #affichage labels predits par K-means (répartition dans l'espace comme le tableau du prof)
#affichage répartition des features dans l'espace latent — Phase 1 MSE
#affichage labels predits par K-means (répartition dans l'espace comme le tableau du prof)
        if CODE == 2 or CODE == 3:
            scaler_viz = StandardScaler()
            features_mse_viz = scaler_viz.fit_transform(features_mse)
            scatter_plot.plot(
                f'CH{SIM_TRAIN} - MSE - labels predits - SS={ss_mse:.4f} - ARI={metrics_MSE[4]:.4f}',

                features_mse_viz,
                lbl_mse,
                marker='o'
            )
            if True:
                save_figure(dossier, f"espace_latent_mse_iteration{iter}")
            plt.show()

        """ Création de la base de donnée d'apprentissage pour la deuxième partie du modèle """
        centres = kmeans.cluster_centers_

        train_data = []
        train_labels = []
        scaler_mse = StandardScaler()
        features_mse_scaled = scaler_mse.fit_transform(features_mse)
        for label in np.unique(lbl_mse):
            indices = np.where(lbl_mse == label)[0]
            features_diff = features_mse_scaled[indices]-np.tile(centres[label],(len(indices),1))
            features_norm = np.linalg.norm(features_diff, axis = 1)
            pourcentage=0.5
            nb_voisins_proches=int(len(indices)*pourcentage)
            ind_proches_centre=np.argsort(features_norm)[:nb_voisins_proches]
            train_data.append(spikes[indices[ind_proches_centre],:])
            train_labels.append(lbl_mse[indices[ind_proches_centre]])
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)



        """ -------------------- PARTIE 2 - APPRENTISSAGE SUPERVISÉ PAR AUTOENCODEUR AVEC COMBINED LOSS -------------------- """

        """ Décomposition en ondelettes """
        DIM_DECOMP = None
        """Nombre d'époques"""
        EPOCHS = 300
        """Dimension du code"""
        CODE = 3

        """Fonction de perte"""
        loss_function = 'combined_loss'

        """ Taux d'apprentissage"""
        learning_rate=0.0001
        """Méthode de classification"""
        CLASS ='k-means'

        """Chargement du modèle existant"""
        loadWeights = None

        """Création et entrainement du modèle"""
        encoder, autoencoder, perf_val, variances_code, variances_par_classe, variances_moyennes = run_autoencoder_train(
                simulation_number=SIM_TRAIN, data=train_data, labels=train_labels,
                decomp_ondelettes=DIM_DECOMP, code_size=CODE, output_activation='linear',
                loss_function=loss_function, scale='minmax', shuff=True, noNoise=False,
                nr_epochs=EPOCHS, learning_rate=learning_rate, dropout=0.0,
                verbose=1, loadWeights=loadWeights)

        """Enregistrement du modèle"""
        save_model(autoencoder, dossier, f'model_ch{SIM_TRAIN}_{loss_function}_code{CODE}_epochs{EPOCHS}_iteration{iter}.weights.h5')

        spikes = np.array(units_in_ch[SIM_TRAIN-1])
        features_comb = encoder.predict(spikes, verbose=0)
        if np.any(np.isnan(features_comb)):
            print("  Phase 2 : features contiennent NaN (loss divergée), on garde Phase 1.")
            features_comb    = features_mse
            lbl_comb         = lbl_mse
            ss_comb          = ss_mse
            metrics_combined = metrics_MSE.copy()
        else:
            lbl_comb, _, ss_comb = best_kmeans_silhouette(features_comb, k, n_try=100)

            if lbl_comb is None:
                print("  Phase 2 : K-Means échoué, on garde Phase 1.")
                features_comb    = features_mse
                lbl_comb         = lbl_mse
                ss_comb          = ss_mse
                metrics_combined = metrics_MSE.copy()
            else:
                print(f"  Phase 2 — Silhouette Score = {ss_comb:.4f}")
                labels = np.array(labels_ch[SIM_TRAIN-1])
                le = LabelEncoder()
                labels_transformed = le.fit_transform(labels)
                labels_transformed = np.array(labels_transformed)
                labels_predits = reassign_labels(labels_transformed, lbl_comb)
                metrics_combined = affichage_metrics_class(labels_transformed, labels_predits)
                metrics_combined.extend(affichage_metrics(features_comb, labels_transformed, labels_predits))
                print(metrics_combined)

        #affichage matrice de confusion
        # disp = matrice_confusion(labels_transformed,labels_predits)
        # disp.plot()
        # plt.show()
        # save_figure(dossier, f"matrice_confusion_{loss_function}")

        # #affichage criteres de performances
        # plt.figure()
        # noms_metrics=np.array(["ARI", "AMI", "VM", "DBS", "CHS", "SS"])
        # bars = plt.bar(noms_metrics, metrics_combined)
        # plt.title("Critères de performances (Combined)")
        # for bar, value in zip(bars, metrics_combined):
        #     plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.2f}",
        #             ha='center', va='bottom', fontsize=10)
        # plt.show()
        # save_figure(dossier, f"criteres_{loss_function}")

        #affichage répartition des features dans l'espace latent — Phase 2 Combined
        #affichage labels predits par K-means (répartition dans l'espace comme le tableau du prof)
#affichage répartition des features dans l'espace latent — Phase 1 MSE
#affichage labels predits par K-means (répartition dans l'espace comme le tableau du prof)
        if True:

            pca = PCA(n_components=2)
            scaler_viz_comb = StandardScaler()
            features_comb_viz = scaler_viz_comb.fit_transform(features_comb)
            features_comb_2D = pca.fit_transform(features_comb_viz)


            scatter_plot.plot(
                f'CH{SIM_TRAIN} - Combined Loss - labels predits - SS={ss_comb:.4f} - ARI={metrics_combined[4]:.4f}',
                features_comb_2D,
                lbl_comb,
                marker='o'
            )


            if True:
                save_figure(dossier, f"espace_latent_comb_iteration{iter}")
            plt.show()

        #affichage 2D après PCA
        # pca = PCA(n_components=2)
        # X_pca = pca.fit_transform(features_comb)
        # scatter_plot.plot('',X_pca, labels_predits, marker='o')
        # plt.title("")
        # plt.xlabel("")
        # plt.ylabel("")
        # plt.grid(False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('equal')
        # plt.show()

        #Print Résultats
        print(" ")
        print(f"CRITÈRES MSE: {metrics_MSE}")
        print(" ")
        print(f"CRITÈRES COMBINED LOSS: {metrics_combined}")
        print(" ")
        print(f"  SS Phase 1 = {ss_mse:.4f}  |  SS Phase 2 = {ss_comb:.4f}  |  Gain = {ss_comb-ss_mse:+.4f}")
        print(" ")
        print(metrics_MSE[0]<metrics_combined[0],  #accuracy_score
        metrics_MSE[1]<metrics_combined[1],         #precision_score
        metrics_MSE[2]<metrics_combined[2],         #recall
        metrics_MSE[3]<metrics_combined[3],         #f1
        metrics_MSE[4]<metrics_combined[4],         #ARI
        metrics_MSE[5]<metrics_combined[5],         #AMI
        metrics_MSE[6]<metrics_combined[6],         #V-Measure
        metrics_MSE[7]>metrics_combined[7],         #DBS
        metrics_MSE[8]<metrics_combined[8],         #CHS
        metrics_MSE[9]<metrics_combined[9])         #SS

        #sauvegarde variables
        fichierSauvegarde = open(f"{dossier}/metrics_ch{SIM_TRAIN}_iteration{iter}","wb")
        pickle.dump(metrics_MSE, fichierSauvegarde)
        pickle.dump(metrics_combined, fichierSauvegarde)
        fichierSauvegarde.close()

        plt.close('all')

print("\n=== DONNÉES RÉELLES TERMINÉES ===")
print("Résultats sauvegardés dans : REALDATA_CH17")
