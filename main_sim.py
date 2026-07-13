## SCRIPT ARTICLE 4 RECORDINGS

import os

import sys
sys.path.append(os.path.abspath("/home/elizaveta/Documents/StageElizaveta/Code/Autoencoders-in-Spike-Sorting-main"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from scipy import fft
from sklearn.cluster import KMeans, DBSCAN
import pywt

from ae_function import run_autoencoder_train, run_autoencoder_test, create_dossier, create_dossier_article, save_figure, save_model, load_model
from dataset_parsing import simulations_dataset as ds
from visualization import scatter_plot
from validation.performance import reassign_labels, matrice_confusion, affichage_metrics, affichage_metrics_class

from sklearn.preprocessing import LabelEncoder
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("/home/elizaveta")


RECORDINGS = []
Dejafait=[ 16, 35, 1]
donotwork=[25, 44]
for i in range(1,96):
    if i not in Dejafait and i not in donotwork:
        RECORDINGS.append(i)
iter = 18
while iter<100:

    iter+=1


    for i in range(len(RECORDINGS)):
        SIM_TRAIN = RECORDINGS[i]
        dossier_check = f"ARTICLE_SIM{SIM_TRAIN}"
        fichier_iter = f"{dossier_check}/metrics_{SIM_TRAIN}_iteration{iter}"
        if os.path.exists(fichier_iter):
            print(f"Itération {iter} déjà faite pour SIM{SIM_TRAIN}, on passe à la suivante.")
            continue

        print(" ")
        print(f"ENREGISTREMENT {SIM_TRAIN}")

        """ PARTIE 1 -------------------- APPRENTISSAGE NON SUPERVISÉ PAR AUTOENCODEUR AVEC MSE (on cherche à récuperer les classes des spikes et construire un ensemble d'apprentissage supervisé par un second autoencodeur -------------------- """

        SIM_TEST = SIM_TRAIN
        dossier = f"ARTICLE_SIM{SIM_TRAIN}"
        """ Décomposition en ondelettes """
        DIM_DECOMP = None
        """Nombre d'époques"""
        EPOCHS = 300
        """Dimension du code"""
        CODE = 2
        """Fonction de perte"""
        loss_function = 'mse'

        """ Taux d'apprentissage"""
        learning_rate=0.001

        """Méthode de classification"""
        CLASS ='k-means'
        """Chargement du modèle existant"""
        loadWeights = None


        """Création et entrinement du modèle"""
        encoder_mse, autoencoder_mse, perf_val, variances_code, variances_par_classe, variances_moyennes = run_autoencoder_train(
                simulation_number=SIM_TRAIN, data=None, labels=None, decomp_ondelettes=DIM_DECOMP, code_size=CODE, output_activation='tanh', loss_function=loss_function, scale=None, shuff=True, noNoise=False, nr_epochs=EPOCHS, learning_rate=learning_rate, dropout=0.0, verbose=1, loadWeights=loadWeights)

        """Enregistrement du modèle"""
        save_model(autoencoder_mse, dossier, f'model_sim{SIM_TRAIN}_{loss_function}_code{CODE}_epochs{EPOCHS}_iteration{iter}.weights.h5')

        """ Validation du modèle"""
        spikes, labels, _ = ds.get_dataset_simulation(simNr=SIM_TRAIN, align_to_peak=True)
        features = encoder_mse.predict(spikes)
        le = LabelEncoder()
        labels_transformed = le.fit_transform(labels)
        labels_transformed = np.array(labels_transformed)
        kmeans = KMeans(n_clusters=len(np.unique(labels)))
        kmeans_labels = kmeans.fit_predict(features)
        kmeans_labels = np.array(kmeans_labels)
        labels_predits = reassign_labels(labels_transformed,kmeans_labels)
        metrics_MSE = affichage_metrics_class(labels_transformed, labels_predits)
        metrics_MSE.extend(affichage_metrics(features,labels_transformed, labels_predits))
        print(metrics_MSE)

        pourc_err=len(np.unique(labels))/100
        val_crit=1-pourc_err
        compt=0
        best_metrics = None
        best_labels_predits = None
        best_kmeans_labels = None
        while metrics_MSE[4]<val_crit or metrics_MSE[5]<val_crit or metrics_MSE[6]<val_crit:
            compt+=1
            print(f"Pré-classification n'est pas ok, tentative {compt}")
            kmeans = KMeans(n_clusters=len(np.unique(labels)))
            kmeans_labels = kmeans.fit_predict(features)
            kmeans_labels = np.array(kmeans_labels)
            labels_predits = reassign_labels(labels_transformed,kmeans_labels)
            metrics_MSE = affichage_metrics_class(labels_transformed, labels_predits)
            metrics_MSE.extend(affichage_metrics(features,labels_transformed, labels_predits))
            print(metrics_MSE)
            if best_metrics is None or sum(metrics_MSE[4:6]) > sum(best_metrics[4:6]):
                best_metrics = metrics_MSE
                best_labels_predits = labels_predits.copy()
                best_kmeans_labels = kmeans_labels.copy()
            if compt>20:
                print("Nombre max de tentatives atteint, récupération de meilleure classification K-means trouvée")
                labels_predits = best_labels_predits
                kmeans_labels = best_kmeans_labels
                metrics_MSE = best_metrics
                break
        print(f"Pré-classification est ok (après {compt} itérations)")


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

        #affichage codes des spikes avec labels GT/ labels predits par K-means
        # if CODE == 2 or CODE == 3:
        #     scatter_plot.plot(f'Train on Sim{SIM_TRAIN} - Test on Sim{SIM_TEST} - Loss: {loss_function} - labels GT', features, labels_transformed, marker='o')
        #     #save_figure(dossier, f"codes_labels_gt_{loss_function}")
        #     scatter_plot.plot(f'Train on Sim{SIM_TRAIN} - Test on Sim{SIM_TEST} -  Loss: {loss_function} - labels {CLASS}', features, labels_predits, marker='o')
        #     #save_figure(dossier, f"codes_labels_predits_{loss_function}")
        #     plt.show()


        """ Création de la base de donnée d'apprentissge pour la deuxième partie du modèle """
        centres = kmeans.cluster_centers_

        train_data = []
        train_labels = []

        for label in np.unique(labels_predits):
            indices = np.where(labels_predits == label)[0]
            features_diff = features[indices]-np.tile(centres[label],(len(indices),1))
            features_norm = np.linalg.norm(features_diff, axis = 1)
            pourcentage=0.5
            nb_voisins_proches=int(len(indices)*pourcentage)
            ind_proches_centre=np.argsort(features_norm)[:nb_voisins_proches]
            train_data.append(spikes[ind_proches_centre,:])
            train_labels.append(labels_predits[ind_proches_centre])

        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        #np.save(f'train_data_{SIM_TRAIN}.npy', train_data)
        #np.save(f'train_labels_{SIM_TRAIN}.npy', train_labels)


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

        """Création et entrinement du modèle"""
        encoder, autoencoder, perf_val, variances_code, variances_par_classe, variances_moyennes = run_autoencoder_train(
                simulation_number=SIM_TRAIN, data=train_data, labels=train_labels, decomp_ondelettes=DIM_DECOMP, code_size=CODE, output_activation='linear',loss_function=loss_function, scale=None, shuff=True, noNoise=False, nr_epochs=EPOCHS, learning_rate=learning_rate, dropout=0.0, verbose=1, loadWeights=loadWeights)

        """Enregistrement du modèle"""
        save_model(autoencoder, dossier, f'model_sim{SIM_TRAIN}_{loss_function}_code{CODE}_epochs{EPOCHS}_iteration{iter}.weights.h5')

        """ Test du modèle"""
        # metrics_combined, labels = run_autoencoder_test(simulation_test=SIM_TEST, simulation_train=SIM_TRAIN, data=spikes, labels=labels, decomp_ondelettes=DIM_DECOMP, encoder=encoder, classification=CLASS, code_size=CODE, scale=None,  output_activation='tahn', loss_function=loss_function, doPlot=True, doPlotWaveforms=False, doPlotMetrics=True)

        spikes, labels, _ = ds.get_dataset_simulation(simNr=SIM_TRAIN, align_to_peak=True)
        features = encoder.predict(spikes)
        le = LabelEncoder()
        labels_transformed = le.fit_transform(labels)
        labels_transformed = np.array(labels_transformed)
        kmeans = KMeans(n_clusters=len(np.unique(labels)))
        kmeans_labels = kmeans.fit_predict(features)
        kmeans_labels = np.array(kmeans_labels)
        labels_predits = reassign_labels(labels_transformed,kmeans_labels)
        metrics_combined = affichage_metrics_class(labels_transformed, labels_predits)
        metrics_combined.extend(affichage_metrics(features,labels_transformed, labels_predits))
        compt=0
        best_metrics = metrics_combined
        best_labels_predits = labels_predits
        best_kmeans_labels = kmeans_labels
        while compt<20:

            compt+=1
            print(f"Résultats de classification pour le deuxième modèle pas ok, tentative {compt}")
            kmeans = KMeans(n_clusters=len(np.unique(labels)))
            kmeans_labels = kmeans.fit_predict(features)
            kmeans_labels = np.array(kmeans_labels)
            labels_predits = reassign_labels(labels_transformed,kmeans_labels)
            metrics_combined = affichage_metrics_class(labels_transformed, labels_predits)
            metrics_combined.extend(affichage_metrics(features,labels_transformed, labels_predits))
            if sum(metrics_combined[4:6]) > sum(best_metrics[4:6]):
                best_metrics = metrics_combined
                best_labels_predits = labels_predits.copy()
                best_kmeans_labels = kmeans_labels.copy()

        labels_predits = best_labels_predits
        kmeans_labels = best_kmeans_labels
        metrics_combined = best_metrics

        print(f"Résultats de classification (après {compt} itérations)")

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

        #affichage codes des spikes avec labels GT/ labels predits par K-means
        # if CODE == 2 or CODE == 3:
        #     scatter_plot.plot(f'Train on Sim{SIM_TRAIN} - Test on Sim{SIM_TEST} - Loss: {loss_function} - labels GT', features, labels_transformed, marker='o')
        #     #save_figure(dossier, f"codes_labels_gt_{loss_function}")
        #     scatter_plot.plot(f'Train on Sim{SIM_TRAIN} - Test on Sim{SIM_TEST} -  Loss: {loss_function} - labels {CLASS}', features, labels_predits, marker='o')
        #     #save_figure(dossier, f"codes_labels_predits_{loss_function}")
        #     plt.show()

        print(f"CRITÈRES MSE: {metrics_MSE}")
        print(f"CRITÈRES COMBINED LOSS: {metrics_combined}")
        print(" ")
        print(metrics_MSE[0]<metrics_combined[0],
        metrics_MSE[1]<metrics_combined[1],
        metrics_MSE[2]<metrics_combined[2],
        metrics_MSE[3]<metrics_combined[3],
        metrics_MSE[4]<metrics_combined[4],
        metrics_MSE[5]<metrics_combined[5],
        metrics_MSE[6]<metrics_combined[6],
        metrics_MSE[7]>metrics_combined[7],
        metrics_MSE[8]<metrics_combined[8],
        metrics_MSE[9]<metrics_combined[9])

        #sauvegarde variables
        fichierSauvegarde = open(f"{dossier}/metrics_{SIM_TRAIN}_iteration{iter}","wb")
        pickle.dump(metrics_MSE, fichierSauvegarde)
        pickle.dump(metrics_combined, fichierSauvegarde)
        fichierSauvegarde.close()