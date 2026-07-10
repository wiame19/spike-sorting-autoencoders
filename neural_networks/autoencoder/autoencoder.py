import keras
from keras.layers import *
from keras import Model
from keras.regularizers import l1
import tensorflow as tf
import numpy as np
from keras import backend as K

from neural_networks.autoencoder.model_auxiliaries import get_codes, get_loss_function, get_activation_function

class LossMonitorCallback(keras.callbacks.Callback):
    """
    Classe qui réalise un Callback Keras pour le suivi de la perte à la fin de chaque époque.

    Paramètres:
        min_inter_class_distance (float): seuil pour la distance minimale (rq. ce paramètre n'est plus utilisé car difficle à définir)

    Fonctionnement:
        - Récuperation et affichage de la valeur de loss
        - Détection et signalement des valeurs NaN
    """

    def __init__(self, min_inter_class_distance=0.25):
        super().__init__()
        self.threshold = min_inter_class_distance # n'est plus utilisé

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

class AutoencoderModel(Model):
    """
    Classe qui réalise la construction automatique de l'encodeur/décodeur.

    Cette classe peut construire:
        - un autoencodeur complet (entrée -> code latent -> sortie)
        - un encodeur (entrée -> code latent)

    Paramètres:
        input_size (int): dimension d'entrée des spikes (79 géneralement sauf si dans le cas de décomposition en ondelettes)
        encoder_layer_sizes (list[int]): tailles des couches cachées de l'encodeur
        decoder_layer_sizes (list[int]): tailles des couches cachées du décodeur
        code_size (int): dimension du code latent
        output_activation (str | callable): fonction d'activation de la couche de sortie (ex: 'tanh', 'sigmoid', etc.)
        loss_function (str | callable): nom de la fonction de coût à utiliser lors de la compilatio
        dropout (float): taux de dropout appliqué aux couches cachées (0.0 par défaut)
        initializer (str): stratégie d'initialisation des poids ('glorot_uniform' par défaut)

    Attributs:
        autoencoder (keras.Model): modèle complet entrée->sortie
        encoder (keras.Model): modèle entrée->code latent
        input_spike (keras.Input): tenseur d'entrée
        code_result (keras.Layer): couche latente (code)
        output_spike (keras.Layer): couche de sortie
    """

    def __init__(self, input_size, encoder_layer_sizes, decoder_layer_sizes, code_size, output_activation, loss_function, dropout=0.0, initializer='glorot_uniform'):
        super(AutoencoderModel, self).__init__()
        self.input_size = input_size
        self.loss_function = loss_function
        activation = get_activation_function(output_activation)

        self.input_spike = Input(shape=(self.input_size,))

        current_layer = self.input_spike
        for hidden_layer_size in encoder_layer_sizes:
            hidden_layer = Dense(hidden_layer_size, activation='relu', kernel_initializer=initializer)(current_layer)
            if dropout == True:
                dropout_layer = Dropout(dropout)(hidden_layer)
                current_layer = dropout_layer
            else:
                current_layer = hidden_layer
        self.code_result = Dense(code_size, activation=activation, activity_regularizer=l1(10e-7), name="code")(current_layer)

        decoder_layer_sizes = np.flip(decoder_layer_sizes)

        # self.code_input = Input(shape=(code_size,))
        current_layer = self.code_result
        for hidden_layer_size in decoder_layer_sizes:
            hidden_layer = Dense(hidden_layer_size, activation='relu', kernel_initializer=initializer)(current_layer)
            if dropout != 0:
                dropout_layer = Dropout(dropout)(hidden_layer)
                current_layer = dropout_layer
            else:
                current_layer = hidden_layer
        self.output_spike = Dense(self.input_size, activation=activation)(current_layer)

        self.autoencoder = Model(self.input_spike, self.output_spike) # spike reconstruit
        self.encoder = Model(self.input_spike, self.code_result) #code latent


    # def pre_train(self, training_data, autoencoder_layer_sizes, epochs):
    #     """
    #     Greedy layer-wise pretraining [1]
    #     [1] A. Sagheer and M. Kotb, ‘Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems’, Sci Rep, vol. 9, no. 1, p. 19038, Dec. 2019, doi: 10.1038/s41598-019-55320-6.
    #     :param training_data: matrix - the points of the dataset
    #     :param autoencoder_layer_sizes: vector - the number of neurons per layer
    #     :param epochs: integer - the number of epochs for pretraining
    #     """
    #     encoder_layer_weights = []
    #     decoder_layer_weights = []
    #
    #     current_training_data = training_data
    #     current_input_size = self.input_size
    #
    #     for code_size in autoencoder_layer_sizes:
    #         input = Input(shape=(current_input_size,))
    #         code_out = Dense(code_size, activation='tanh', activity_regularizer=l1(10e-8))(input)
    #         code_in = Input(shape=(code_size,))
    #         output = Dense(current_input_size, activation='tanh')(code_in)
    #
    #         encoder = Model(input, code_out)
    #         decoder = Model(code_in, output)
    #
    #         input = Input(shape=(current_input_size,))
    #         code = encoder(input)
    #         decoded = decoder(code)
    #
    #         pretrained = Model(input, decoded)
    #
    #         # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    #         pretrained.compile(optimizer='adam', loss=self.loss_function)
    #         pretrained.fit(current_training_data, current_training_data, epochs=epochs, verbose=0)
    #
    #         encoder_layer_weights.append(pretrained.get_weights()[0])
    #         encoder_layer_weights.append(pretrained.get_weights()[1])
    #         decoder_layer_weights.append(pretrained.get_weights()[3])
    #         decoder_layer_weights.append(pretrained.get_weights()[2])
    #
    #         current_input_size = code_size
    #         current_training_data = get_codes(current_training_data, encoder)
    #
    #     decoder_layer_weights = decoder_layer_weights[::-1]
    #     encoder_layer_weights.extend(decoder_layer_weights)
    #     return encoder_layer_weights


    def train_autoencodeur(self, training_data, epochs=50, verbose="auto", learning_rate=0.001):
        """
        Fonction qui réalise un entraînement du modèle autoencodeur (reconstruction).

        Paramètres:
            training_data (ndarray): matrice des données d'entraînement (nb_samples, input_size)
            epochs (int): nombre d’époques d’apprentissage (50 par défaut)
            verbose (int/str): verbosité Keras ('auto' par défaut)
            learning_rate (float): taux d’apprentissage pour l’optimiseur Adam (0.001 par défaut)

        Returns:
            None
        """

        encoder, autoencoder = self.return_encoder()

        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)
            W = K.transpose(autoencoder.get_layer('code').get_weights()[0])  # N_hidden x N
            h = autoencoder.get_layer('code').output
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
            lam = 100
            mse = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
            W = tf.transpose(autoencoder.get_layer('code').get_weights()[0])
            h = autoencoder.get_layer('code').output
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

        self.autoencoder.compile(optimizer=opt, loss=loss) #compilation
        self.autoencoder.fit(training_data, training_data, epochs=epochs, verbose=verbose) #entrainement


    def train_encodeur(self, training_data, labels, epochs=100, verbose=1, learning_rate=0.001):
        """
        Fonction qui réalise un entraînement supervisé de l'encodeur avec des fonctions de pertes qui cherchent à amèliorer l'extraction des codes des spikes.

        Paramètres:
            training_data (ndarray): matrice des spikes d'entraînement (nb_samples, input_size)
            labels (ndarray): labels des spikes (nb_samples, 1)
            epochs (int): nombre d’époques (100 par défaut)
            verbose (int): verbosité Keras (1 par défaut)
            learning_rate (float): taux d’apprentissage pour Adam (0.001 par défaut)

        Returns:
            None
        """

        encoder, autoencoder = self.return_encoder()

        # calcul des codes latents pour tous les spikes (necessaires pour le calcul des centroides):
        latent_codes = encoder.predict(training_data)
        # latent_codes = get_codes(training_data, encoder)
        # tf.print('latent_codes:', latent_codes)

        # calcul des centroïdes pour chaque classe:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        num_classes = unique_labels.shape[0]
        code_dim = latent_codes.shape[1]
        class_centroids = np.zeros((num_classes, code_dim), dtype=np.float32)
        for i, label in enumerate(unique_labels):
            class_latents = latent_codes[labels == label] #on recupère les codes correspondants à cette classe
            class_centroids[i] = np.mean(class_latents, axis=0) #on calcule la moyenne (le centroide) de cette classe
        tf_class_centroids = tf.constant(class_centroids, dtype=tf.float32) #conversion en objet TF

        def intra_class_variance_loss(y_true, y_pred):
            """
            Fonction de perte qui cherche à minimiser la variance intra-classe.

            Paramètres:
                y_true: labels des spikes [nombre de spikes x 1]
                y_pred: codes latents correspondant à chaque spike [dimension du code latent x nombre de spikes]

            Returns:
                variance (float): variance intra-classe
            """

            y_true = tf.cast(y_true, tf.int32)
            #tf.print('y_true:', y_true)
            #tf.print('y_pred:', y_pred)

            centers = tf.gather(tf_class_centroids, y_true)
            #tf.print('les centres:',centers)

            # distances = tf.reduce_sum(tf.square(y_pred - centers), axis=1)
            # tf.print('les distances:',distances)
            #
            # variance = tf.reduce_mean(distances)
            # tf.print('intra-class loss:', variance)

            distances = tf.reduce_max(tf.square(y_pred - centers), axis=1)
            #tf.print('les distances:',distances)

            variance = tf.reduce_mean(distances)
            #tf.print('intra-class loss:', variance)


            return variance


        def inter_class_variance_loss(y_true, y_pred):
            """
            Fonction de perte qui cherche à maximiser la distance entre les centroides de chaque classe.

            Paramètres:
                y_true: labels des spikes [nombre de spikes x 1]
                y_pred: codes latents correspondant à chaque spike [dimension du code latent x nombre de spikes]
            Returns:
                loss (float): distance minimale entre les centres des clusters
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
            Fonction de perte combinée qui minimise la variance intra-classe et maximise la distance inter-classe.

            Paramètres:
                y_true: labels des spikes [nombre de spikes x 1]
                y_pred: codes latents correspondant à chaque spike [dimension du code latent x nombre de spikes]

            Returns:
                total_loss (float): combinaison des fonctions perte intra-lss et intra-loss
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
            distances=tf.reduce_sum(tf.square(y_pred - centers), axis=1)
            # tf.print('les distances:',distances)
            intra_loss=tf.reduce_mean(distances)

            # distances = tf.reduce_max(tf.square(y_pred - centers), axis=1)
            # intra_loss = tf.reduce_mean(distances)

            inter_loss=inter_class_variance_loss(y_true, y_pred)

            alpha=1 #poids de la perte intra-classe (à minimiser)
            beta=1 #poids de la perte inter-classe (on minimise -inter_class_loss)

            total_loss=alpha*intra_loss+beta*inter_loss  #inter_loss est déjà négatif
            #tf.print('intra_loss:',intra_loss)
            #tf.print('inter_loss:',-inter_loss)
            # total_loss = intra_loss/(intra_loss-inter_loss)
            # code_dim = tf.cast(code_dim, tf.float32)
            # total_loss = (code_dim*intra_loss-inter_loss*intra_loss)/(code_dim*inter_loss-inter_loss*intra_loss)

            #inter_loss=-inter_loss
            #total_loss=(intra_loss-intra_loss*inter_loss)/(inter_loss-intra_loss*inter_loss)

            # tf.print('intra loss:', intra_loss)
            # tf.print('inter loss:', inter_loss)
            # tf.print('combined loss:', total_loss)

            return total_loss


        # compilation et entraînement
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        if self.loss_function == 'intra_class_variance_loss':
            loss = intra_class_variance_loss
        elif self.loss_function == 'inter_class_variance_loss':
            loss = inter_class_variance_loss
        elif self.loss_function == 'combined_loss':
            loss = combined_loss
            print("I USE THE COMBINED LOSS")
        else:
            loss = get_loss_function(self.loss_function)
        self.encoder.compile(optimizer=opt, loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])
        print('""""""""""""""""""""""""""""')
        self.encoder.fit(training_data, labels, epochs=epochs, verbose=0, callbacks=[LossMonitorCallback(min_inter_class_distance=0.25)],validation_data=(training_data, labels))
        # self.encoder.fit(training_data, labels, epochs=epochs, verbose=verbose,validation_data=(training_data, labels))


    def return_encoder(self):
        """
        Fonction qui donne l'accès au couple (encodeur, autoencodeur).

        Returns:
            tuple[keras.Model, keras.Model]: (self.encoder, self.autoencoder).
        """

        return self.encoder, self.autoencoder


    def return_autoencoder(self):
        """
        Fonction qui donne l'accès à l'autoencodeur.

        Returns:
            keras.Model: modèle autoencodeur
        """

        return self.autoencoder


