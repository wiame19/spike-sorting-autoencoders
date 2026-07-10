
import keras
from keras.layers import *
from keras import Model
from keras.regularizers import l1
import tensorflow as tf
import numpy as np
from keras import backend as K

from keras.layers import *
from keras import Model
from keras.regularizers import l1

import numpy as np
from neural_networks.autoencoder.model_auxiliaries import get_codes, get_loss_function, get_activation_function


# model.add(RepeatVector(3))
# repeats input n times
# now: model.output_shape == (x, y, 3)
# TimeDistributed(layer)
# This wrapper allows to apply a layer to every temporal slice of an input.


class LossMonitorCallback(keras.callbacks.Callback):
    def __init__(self, min_inter_class_distance=0.25):
        super().__init__()
        self.threshold = min_inter_class_distance

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        if loss is not None:
            if tf.math.is_nan(loss):
                print(f'Epoch {epoch}: Loss = nan')
            # elif loss < -self.threshold:
            #     print(f' Epoch {epoch}: Bonne séparation inter-classes (distance > {self.threshold})')
            # else:
            #     print(f'Epoch {epoch}: Mauviase séparation inter-classe (distance = {-loss:.4f} < {self.threshold})')
            else:
                print(f' Epoch {epoch}: Loss = {-loss:.4f}')

class LSTMAutoencoderModel(Model):

    def __init__(self, input_size, lstm_layer_sizes, code_size):
        super(LSTMAutoencoderModel, self).__init__()

        self.input_size = input_size

        self.input_spike = Input(shape=(self.input_size[1], self.input_size[2]))

        current_layer = self.input_spike
        for hidden_layer_size in lstm_layer_sizes:
            hidden_layer_lstm = LSTM(hidden_layer_size, activation='relu', return_sequences=True)(current_layer)
            current_layer = hidden_layer_lstm

        # self.code_result = LSTM(code_size, activation='tanh', return_sequences=True, activity_regularizer=l1(10e-7))(current_layer)
        self.code_result = Dense(code_size, activation='tanh', activity_regularizer=l1(10e-7))(current_layer)

        lstm_layer_sizes = np.flip(lstm_layer_sizes)

        # self.code_input = Input(shape=(code_size,))
        current_layer = self.code_result
        for hidden_layer_size in lstm_layer_sizes:
            hidden_layer_lstm = LSTM(hidden_layer_size, activation='relu', return_sequences=True)(current_layer)
            current_layer = hidden_layer_lstm

        # self.output_spike = TimeDistributed(Dense(self.input_size[1], activation='tanh'))(current_layer)
        # assertion error
        self.output_spike = TimeDistributed(Dense(self.input_size[2], activation='tanh'))(current_layer)

        self.autoencoder = Model(self.input_spike, self.output_spike)

        self.encoder = Model(self.input_spike, self.code_result)

    def train(self, training_data, epochs=50, verbose="auto"):
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(training_data, training_data, epochs=epochs, verbose=verbose)


    def train_autoencodeur(self, training_data, labels, epochs=50, verbose="auto", learning_rate=0.001):
        """
        Autoencoder training for a number of epochs
        :param training_data: matrix - the points of the dataset
        :param epochs: integer - the number of epochs for pretraining
        :param learning_rate: integer - the learning rate
        """

        # model_var = self.return_autoencoder()
        encoder, autoencoder = self.return_encoder()
        lam = 100


        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)
            W = K.transpose(model_var.get_layer('code').get_weights()[0])  # N_hidden x N
            h = model_var.get_layer('code').output
            dh = h * (1 - h)  # N_batch x N_hidden

            # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
            contractive = K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)
            return mse + contractive


        def mse_modified(y_true, y_pred):

            # nombre d'échantillons
            N_mid = 30  # de 16 à 45 inclus (30 points)
            N_forward = 15  # de 1 à 15 (15 points)
            N_backward = 34  # de 46 à 79 (34 points)

            # pondération de poids
            w_milieu = 2.0
            w_others = 1.0

            # partie avant
            forward = K.square(y_true[:,0:15] - y_pred[:,0:15])
            forward_loss = (w_others / N_forward) * K.sum(forward, axis=1)
            # partie centre
            middle = K.square(y_true[:,15:45] - y_pred[:,15:45])
            middle_loss = (w_milieu / N_mid) * K.sum(middle, axis=1)
            # partie après
            backward = K.square(y_true[:,45:] - y_pred[:,45:])
            backward_loss = (w_others / N_backward) * K.sum(backward, axis=1)

            total_loss = forward_loss + middle_loss + backward_loss
            return K.mean(total_loss)

        def contractive_loss2(x, x_bar):
            mse = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
            W = tf.transpose(model_var.get_layer('code').get_weights()[0])
            h = model_var.get_layer('code').output
            dh = h * (1 - h)
            contractive = lam * tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)
            total_loss = mse + contractive
            return total_loss

        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        if self.loss_function == 'contractive':
            loss = contractive_loss
        elif self.loss_function == 'mse_modified':
            loss = mse_modified
        else:
            loss = get_loss_function(self.loss_function)

        # # TRAIN POUR LOSS(SPIKE ORIGINAL,SPIKE RECONSTRUIT)
        self.autoencoder.compile(optimizer=opt, loss=loss)
        self.autoencoder.fit(training_data, training_data, epochs=epochs, verbose=verbose)


    def train_encodeur(self, training_data, labels, epochs=100, verbose=1, learning_rate=0.001):

        encoder, autoencoder = self.return_encoder()

        # calcul des codes latents pour tous les spikes (necessaires pour le calcul des centroides)
        latent_codes = encoder.predict(training_data)
        # latent_codes = get_codes(training_data, encoder)
        tf.print('latent_codes:', latent_codes)

        # calcul des centroïdes pour chaque classe
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        num_classes = unique_labels.shape[0]
        code_dim = latent_codes.shape[1]

        class_centroids = np.zeros((num_classes, code_dim), dtype=np.float32)
        for i, label in enumerate(unique_labels):
            class_latents = latent_codes[labels == label] #on recupère les codes correspondants à cette classe
            print(f"codes d'une classe: {class_latents}")
            class_centroids[i] = np.mean(class_latents, axis=0) #on calcule la moyenne (le centroide) de cette classe
        tf_class_centroids = tf.constant(class_centroids, dtype=tf.float32) #conversion en objet TF

        def intra_class_variance_loss(y_true, y_pred):
            """
            Fonction de perte qui cherche à minimiser la variance intra-classe
            y_true: labels des spikes [nombre de spikes x 1]
            y_pred: codes latents correspondant à chaque spike [dimension du code latent x nombre de spikes]
            """

            y_true = tf.cast(y_true, tf.int32)
            tf.print('y_true:', y_true)
            tf.print('y_pred:', y_pred)

            centers = tf.gather(tf_class_centroids, y_true)
            tf.print('les centres:',centers)

            distances = tf.reduce_sum(tf.square(y_pred - centers), axis=1)
            tf.print('les distances:',distances)

            variance = tf.reduce_mean(distances)
            tf.print('intra-class loss:', variance)



            return variance


        def inter_class_variance_loss(y_true, y_pred):
            """
            Fonction de perte qui cherche à maximiser la distance entre les centroides de chaque classe
            y_true: labels des spikes [nombre de spikes x 1]
            y_pred: codes latents correspondant à chaque spike [dimension du code latent x nombre de spikes]
            """

            y_true = tf.cast(y_true, tf.int32)

            nb_spikes = tf.shape(y_pred)[0]
            code_dim = tf.shape(y_pred)[1]

            y_true = tf.reshape(y_true, [nb_spikes])
            # tf.print('y_true = ', y_true)
            # tf.print('y_pred = ', y_pred)

            nb_classes = tf.reduce_max(y_true)+1
            # tf.print('nb_classes',nb_classes)

            # nombre de spikes par classe
            class_counts = tf.math.unsorted_segment_sum(
                tf.ones_like(y_true, dtype=tf.float32), y_true, nb_classes)
            # tf.print("Nombre d'éléments par classe:", class_counts)

            # # calcul des centroïdes
            # # # # # # # # centroids = tf.math.divide_no_nan(sum_classes, class_counts)
            centroids = tf.math.unsorted_segment_mean(y_pred, y_true, nb_classes)
            mask1 = tf.reduce_any(centroids != 0, axis=1)
            centroids = tf.boolean_mask(centroids, mask1)
            nb_classes = tf.shape(centroids)[0]
            # tf.print('centroids = ', centroids)

            # calcul des distances entre chaque paire de centroïdes
            # on étend les dimensions
            centroids_exp1 = tf.expand_dims(centroids, axis=1)  # [nb_classes, 1, code_dim]
            centroids_exp2 = tf.expand_dims(centroids, axis=0)  # [1, nb_classes, code_dim]
            # tf.print('centroids_exp1 = ', centroids_exp1)
            # tf.print('centroids_exp2 = ', centroids_exp2)

            # on crée une matrice des distances entre centroïdes
            distances = tf.norm(centroids_exp1 - centroids_exp2, axis=2)
            # tf.print('distances = ', distances)

            # on supprime les distances diagonales
            mask2 = tf.logical_not(tf.eye(nb_classes, dtype=tf.bool))

            # on extrait les distances inter-classes hors diagonale
            inter_class_distances = tf.boolean_mask(distances, mask2)
            # tf.print('inter_class_distances:', inter_class_distances)
            # inter_class_distances = tf.square(inter_class_distances)

            # loss = -tf.reduce_mean(inter_class_distances)
            loss = -tf.reduce_min(inter_class_distances)
            # tf.print('inter-class loss:', loss)
            # if tf.math.reduce_any(tf.math.is_nan(loss)):
            #     tf.print("Loss = nan")
            #     break

            return loss

            # # # # # tf.print('inter-class loss:', -tf.reduce_min(inter_class_distances))
            # # # # #
            # # # # # return -tf.reduce_min(inter_class_distances)



        def combined_loss(y_true, y_pred):
            """
            Fonction de perte combinée qui minimise la variance intra-classe et maximise la distance inter-classe
            y_true: labels des spikes [nombre de spikes x 1]
            y_pred: codes latents correspondant à chaque spike [dimension du code latent x nombre de spikes]
            """
            y_true = tf.cast(y_true, tf.int32)

            nb_classes = tf.reduce_max(y_true)+1
            nb_spikes = tf.shape(y_pred)[0]
            code_dim = tf.shape(y_pred)[1]

            y_true = tf.reshape(y_true, [nb_spikes])

            sum_classes = tf.math.unsorted_segment_sum(y_pred, y_true, nb_classes)

            # nombre de spikes d'une meme classe
            class_counts=tf.math.unsorted_segment_sum(tf.ones_like(y_true, dtype=tf.float32), y_true, nb_classes)
            class_counts=tf.reshape(class_counts, [-1, 1])

            # calcul des centroïdes
            centroids=tf.math.divide_no_nan(sum_classes, class_counts)

            centers=tf.gather(centroids, y_true)
            # tf.print('les centres:',centers)
            # print('les labels:', labels)
            distances=tf.reduce_sum(tf.square(y_pred - centers), axis=1)
            # tf.print('les distances:',distances)
            intra_loss=tf.reduce_mean(distances)

            inter_loss=inter_class_variance_loss(y_true, y_pred)

            alpha=1 #poids de la perte intra-classe (à minimiser)
            beta=0.5 #poids de la perte inter-classe (on minimise -inter_class_loss)

            total_loss=alpha*intra_loss+beta*inter_loss  #inter_loss est déjà négatif
            # tf.print('intra_loss:',intra_loss)
            # tf.print('inter_loss:',-inter_loss)
            # total_loss = intra_loss/(intra_loss-inter_loss)

            # tf.print('intra loss:', intra_loss)
            # tf.print('inter loss:', inter_loss)
            # tf.print('combined loss:', total_loss)

            return total_loss


        # compilation et entraînement
        if self.loss_function == 'contractive':
            loss = contractive_loss
        elif self.loss_function == 'mse_modified':
            loss = mse_modified
        elif self.loss_function == 'triplet_loss':
            loss = triplet_loss
        elif self.loss_function == 'intra_class_variance_loss':
            loss = intra_class_variance_loss
        elif self.loss_function == 'inter_class_variance_loss':
            loss = inter_class_variance_loss
        elif self.loss_function == 'combined_loss':
            loss = combined_loss
        else:
            loss = get_loss_function(self.loss_function)
        self.encoder.compile(optimizer='adam', loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])
        self.encoder.fit(training_data, labels, epochs=epochs, verbose=verbose, callbacks=[LossMonitorCallback(min_inter_class_distance=0.25)],validation_data=(training_data, labels))
        # self.encoder.fit(training_data, labels, epochs=epochs, verbose=verbose,validation_data=(training_data, labels))

    def return_encoder(self):
        return self.encoder, self.autoencoder

    def return_autoencoder(self):
        return self.autoencoder
