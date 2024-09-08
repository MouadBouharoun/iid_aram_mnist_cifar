import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


'''
Cette foncton implémente l'initialisation Glorot
Ref : Glorot, X., & Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. 
In international conference on artificial intelligence and statistics, 2010.
Cette fonction est conçue pour maintenir la variance des gradients relativement constante à travers les couches du VAE.
'''
def glorot(in_shape):
    val = tf.random.normal(shape=in_shape, stddev=1./tf.sqrt(in_shape[0]/2.))
    return val

'''
Cette fonction définie l'architecture du VAE (see Image 1 in README.md)
initialise les matrices de poids et les vecteurs de biais pour l'encodeur et le décodeur en utilisant l'initialisation Glorot.
'''
def initialize_weights_and_biases(input_dimension, neural_network_dimension, latent_variable_dimension):
    Weight = {
        "weight_matrix_encoder_hidden": tf.Variable(glorot([input_dimension, neural_network_dimension])),
        "weight_mean_hidden": tf.Variable(glorot([neural_network_dimension, latent_variable_dimension])),
        "weight_stddev_hidden": tf.Variable(glorot([neural_network_dimension, latent_variable_dimension])),
        "weight_matrix_decoder_hidden": tf.Variable(glorot([latent_variable_dimension, neural_network_dimension])),
        "weight_decoder": tf.Variable(glorot([neural_network_dimension, input_dimension]))}

    Bias = {
        "bias_matrix_encoder_hidden": tf.Variable(glorot([neural_network_dimension])),
        "bias_mean_hidden": tf.Variable(glorot([latent_variable_dimension])),
        "bias_stddev_hidden": tf.Variable(glorot([latent_variable_dimension])),
        "bias_matrix_decoder_hidden": tf.Variable(glorot([neural_network_dimension])),
        "bias_decoder": tf.Variable(glorot([input_dimension]))}
    
    return Weight, Bias

'''
L'objectif de cette fonction est de trouver les paramètres \phi de l'encodeur probabiliste (q_{\phi} ~ N(\mu, \sigma))
\phi = {Weight["weight_matrix_encoder_hidden"] + Bias["bias_matrix_encoder_hidden"]}
Les valeurs pris par \phi sont utilisé pour construire la couche (\mu, \phi); par conséquent : \mu = \mu(\phi), et \sigma = \sigma(\phi)
les paramètres \mu, \sigma sont utilisés pour créer des échnatillons z de l'espace latent avec z ~ N(\mu, \sigma)
'''
def encoder(x, Weight, Bias):
    encoder_layer = tf.add(tf.matmul(x, Weight["weight_matrix_encoder_hidden"]), Bias["bias_matrix_encoder_hidden"])
    encoder_layer = tf.nn.tanh(encoder_layer)
    mean_layer = tf.add(tf.matmul(encoder_layer, Weight["weight_mean_hidden"]), Bias["bias_mean_hidden"])
    stddev_layer = tf.add(tf.matmul(encoder_layer, Weight["weight_stddev_hidden"]), Bias["bias_stddev_hidden"])
    return mean_layer, stddev_layer

'''
L'optimisation de la fonction de perte totale total_loss() fait intervenir deux termes dont le premier (data_fidelity) 
représente une espérance par rapport à q_{\phi} de z. Ceci rend le calcul de la différentielle par rapport au paramètre \phi 
n'est pas possible car $\dif_{\phi} \E_{q_{\phi}}{z} \neq \E_{\phi}{z} \dif_{\phi}$ 
L'alternative est de remplacer q_{\phi} par une distribution normale centrée réduite \epsilon, et écrire z sous une forme linéaire 
en epsilon z = \mu(\phi) + \epsilon \sigma(\phi).
Dans ce cas : E_{q_{\phi}}{z} = E_{\epsilon}{\mu(\phi) + \epsilon \sigma(\phi)}
Cette alternative est proposée dans ce papier : 
Kingma, D. P., Salimans, T., & Welling, M. Variational dropout and the local reparameterization trick. In neural information processing systems, 2015.
'''
def reparameterize(mean, stddev):
    epsilon = tf.random.normal(tf.shape(stddev), dtype=tf.float32)
    return mean + tf.exp(0.5 * stddev) * epsilon

'''
L'objectif de cette fonction est de trouver les paramètres \phi de l'encodeur probabiliste (p_{\theta} ~ N(\mu, \sigma))
\theta = {Weight["weight_matrix_decoder_hidden"] + Bias["bias_matrix_decoder_hidden"]}
Cette fonction permet de reconstruire un autre vecteur à partir de l'espace latent
            $$$$$ x -> z -> \hat{x} $$$$$
'''

def decoder(z, Weight, Bias):
    decoder_layer = tf.add(tf.matmul(z, Weight["weight_matrix_decoder_hidden"]), Bias["bias_matrix_decoder_hidden"])
    decoder_layer = tf.nn.tanh(decoder_layer)
    decoder_output_layer = tf.add(tf.matmul(decoder_layer, Weight["weight_decoder"]), Bias["bias_decoder"])
    return decoder_output_layer
'''
Divergence de Kullback Leiber
'''
def kl_div_loss(original_data, reconstructed_data, mean, stddev):
    kl_div_loss = -0.5 * tf.reduce_sum(1 + stddev - tf.square(mean) - tf.exp(stddev), axis=1)
    return tf.reduce_mean(kl_div_loss)
'''
Erreur de reconstruction
'''
def data_fidelity_loss(original_data, reconstructed_data, mean, stddev):
    data_fidelity_loss = tf.reduce_mean(tf.square(original_data - reconstructed_data), axis=1)
    return tf.reduce_mean(data_fidelity_loss)

'''
Erreur global = Divergence de Kullback Leiber + Erreur de reconstruction
'''
def total_loss(original_data, reconstructed_data, mean, stddev):
    data_fidelity_loss = tf.reduce_mean(tf.square(original_data - reconstructed_data), axis=1)
    kl_div_loss = -0.5 * tf.reduce_sum(1 + stddev - tf.square(mean) - tf.exp(stddev), axis=1)
    total_loss = tf.reduce_mean(data_fidelity_loss + kl_div_loss)
    return total_loss


'''
Building the computation Graph
'''

def train_vae(x_train, epochs, batch_size, input_dimension, neural_network_dimension, latent_variable_dimension, learning_param):
    Weight, Bias = initialize_weights_and_biases(input_dimension, neural_network_dimension, latent_variable_dimension)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_param)

    total_losses = []
    kl_div_losses = []
    data_fidelity_losses = []
    for i in tqdm.tqdm(range(epochs)):
        indices = np.random.randint(0, x_train.shape[0], batch_size)
        x_batch = x_train[indices]
        with tf.GradientTape() as tape:
            mean, stddev = encoder(x_batch, Weight, Bias)
            latent = reparameterize(mean, stddev)
            reconstruction = decoder(latent, Weight, Bias)
            tot_loss = total_loss(x_batch, reconstruction, mean, stddev)
            k_d_loss = kl_div_loss(x_batch, reconstruction, mean, stddev)
            d_f_loss = data_fidelity_loss(x_batch, reconstruction, mean, stddev)
            total_losses.append(tot_loss)
            kl_div_losses.append(k_d_loss)
            data_fidelity_losses.append(d_f_loss)
        gradients = tape.gradient(tot_loss, list(Weight.values()) + list(Bias.values()))
        optimizer.apply_gradients(zip(gradients, list(Weight.values()) + list(Bias.values())))
        if i % 1000 == 0:
            print("({0},{1})".format(i, tot_loss))

    plt.plot(total_losses, label='Total Loss')
    plt.plot(kl_div_losses, label='KL Divergence Loss')
    plt.plot(data_fidelity_losses, label='Data Fidelity Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.show()
    return (total_losses, kl_div_losses, data_fidelity_losses, Weight, Bias)

'''
Cette première fonction de test permet d'imprimer les erreurs totales pour l'ensemble de test
'''
def detect_malicious_modifications(x_test, Weight, Bias):
    reconstruction_errors = []
    for data_point in x_test:
        mean, stddev = encoder(data_point.reshape(1, -1), Weight, Bias)
        latent = reparameterize(mean, stddev)
        reconstructed_data = decoder(latent, Weight, Bias)
        reconstruction_error = total_loss(data_point, reconstructed_data, mean, stddev)
        reconstruction_errors.append(reconstruction_error)
    return reconstruction_errors

'''
Cette deuxième fonction de test permet d'imprimer une liste contenant des binaires indiquant
si chaque point de données est considéré comme malveillant (1) ou non (0).
'''
def detect_malicious_modifications_2(x_test, threshold, Weight, Bias):
    binary_results = []
    for data_point in x_test:
        mean, stddev = encoder(data_point.reshape(1, -1), Weight, Bias)
        latent = reparameterize(mean, stddev)
        reconstructed_data = decoder(latent, Weight, Bias)
        reconstruction_error = total_loss(data_point, reconstructed_data, mean, stddev)
        # Check if reconstruction error is greater than the threshold
        if reconstruction_error > threshold:
            binary_results.append(1)
        else:
            binary_results.append(0)
    return binary_results

'''
Calcul du seuil pour l'entrainement du VAE
'''
def vae_threshold(losses, alpha=1):
    m = np.mean(losses)
    sigma = np.std(losses)
    n = len(losses)

    return m + alpha * (sigma / np.sqrt(n))
