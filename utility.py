import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

'''
Cette fonction permet de convertir tout les données du dataset sous format numérique
'''
def preprocess(df):
    src_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_SRC_ADDR"].unique()))}
    dst_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_DST_ADDR"].unique()))}
    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(lambda name: src_ipv4_idx[name])
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(lambda name: dst_ipv4_idx[name])
    df = df.drop('Attack', axis=1)
    X=df.iloc[:, :-1].values
    y=df.iloc[:, -1].values
    X = (X - X.min()) / (X.max() - X.min())
    y = y.reshape(-1, 1)
    return (X, y)


'''
Cette fonction permet d'initialiser les modèles
Le premier modèle est celle utilisé en fédération
Le deuxième modèle est effectuer pour effectuer l'attaque d'inconsistence (canary gradient attack)
'''
def initialiseMLP(input_shape, lr=0.01):
    model = Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  
])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return model

def initialisePropMLP(input_shape, lr=0.01):
    model = Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(2, activation='relu')
])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

'''
Cette fonction permet de distribuer les données uniformément sur les participants
'''
def load_data(dataset_key):
    if dataset_key == "mnist":
        return tf.keras.datasets.mnist.load_data()
    elif dataset_key == "cifar10":
        return tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    
def data_distribution(dataset_key, num_clients):
    (train_images, train_labels), (test_images, test_labels) = load_data(dataset_key)
    num_train_samples = len(train_images)
    chunk_size = num_train_samples // num_clients
    client_data_chunks = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + chunk_size
        if i == num_clients - 1: 
            end_idx = num_train_samples
        client_data_chunks.append((train_images[start_idx:end_idx], train_labels[start_idx:end_idx]))
        start_idx = end_idx
    
    return client_data_chunks


'''
Cette fonction calcul un gradient à partir du modèle global et des données locales
'''
def client_update(local_model, data):
    with tf.GradientTape() as tape:
        predictions = local_model(data['X_train'])
        loss = tf.keras.losses.sparse_categorical_crossentropy(data['y_train'], predictions)
    gradients = tape.gradient(loss, local_model.trainable_variables)
    return gradients

'''
Cette fonction permet d'agréger les gradients
'''
def server_aggregate(global_model, client_updates):
    averaged_gradients = [tf.zeros_like(var) for var in global_model.trainable_variables]
    num_clients = len(client_updates)
    for client_update in client_updates:
        for i, grad in enumerate(client_update):
            averaged_gradients[i] += grad / num_clients
    return averaged_gradients

'''
Fonction pour convertir un modèle en un vecteur 1D numpy ndarray
Utile comme étape du processing pour entrainer le VAE
'''
def model_to_vector(models):
    new_models = []
    for model in models:
        weights_list = []
        for layer in model.layers:
            weights = layer.get_weights()
            for w in weights:
                weights_list.append(w.flatten())  
        model_vector = np.concatenate(weights_list)
        new_models.append(model_vector)
    return np.array(new_models).astype(np.float32)

'''
Cette fonction permet d'insérer les poids du modèle inconsistent dans les poids du modèle globale
'''
def insert_weights_MI_to_GM(model_inconsistency, global_model):
    model_inconsistency_weights = model_inconsistency.get_weights()
    partial_model_weights = model_inconsistency_weights[2:4]
    current_weights = global_model.layers[1].get_weights()
    updated_weights = current_weights.copy()
    updated_weights[0][:, :2] = partial_model_weights[0]  # Update weights
    updated_weights[1][:2] = partial_model_weights[1]  # Update biases
    global_model.layers[1].set_weights(updated_weights)
    global_model.layers[0].set_weights(model_inconsistency_weights[0:2])
    return global_model

'''
Cette fonction permet de préparer une séquence de vérités de terrain (ground truth) composée de 1 et de 0, 
où la moitié de la séquence est composée de 1 et l'autre moitié de 0.
'''
def ground_truth(test_size):
    if test_size % 2 != 0:
        raise ValueError("test_size must be congruent to 2.")
    
    num_ones = test_size // 2
    num_zeros = test_size - num_ones

    sequence = [1] * num_ones + [0] * num_zeros
    
    return sequence


"""Cette fonction crée plusieurs datasets de propriétés à partir des configurations données."""
def create_property_datasets(main_dataset, properties_config):
    property_datasets = []
    for i, config in enumerate(properties_config, start=1):
        print(f"Création de property_dataset_{i} pour '{config['feature']}' avec seuil {config['value']}")
        property_dataset = main_dataset.copy()
        property_dataset = property_dataset.drop('Label', axis=1)
        if config['comparison'] == ">=":
            property_dataset['Label'] = (property_dataset[config['feature']] >= config['value']).astype(int)
        elif config['comparison'] == "<=":
            property_dataset['Label'] = (property_dataset[config['feature']] <= config['value']).astype(int)
        else:
            raise ValueError("Opérateur de comparaison non supporté")
        property_datasets.append(property_dataset)
    return property_datasets

"""
Les clients sont traités comme des objets de classe; chacun est reconnu avec un identifiant numérique, ces données locales et son modèle local.
"""
class Client:
    def __init__(self, client_id, data):
        self.client_id = client_id
        self.local_data = data
        self.local_model = None

    def set_global_model(self, global_model):
        self.local_model = tf.keras.models.clone_model(global_model)
        self.local_model.compile(optimizer=global_model.optimizer,
                                 loss=tf.keras.losses.sparse_categorical_crossentropy,
                                 metrics=['accuracy'])

    def get_local_model(self):
        return self.local_model
    
    def preprocess(self, images, labels):
        images = images / 255.0
        return images, labels
    def get_data(self):
        X_train, y_train = self.local_data
        X_train, y_train = self.preprocess(X_train, y_train)
        return {'X_train': X_train,
                'y_train': y_train}



class Server:
    def __init__(self, model_fn):
        self.global_model = model_fn()
        self.clients = []

    def add_client(self, client):
        self.clients.append(client)
        client.set_global_model(self.global_model)

    def aggregate_weights(self):
        num_clients = len(self.clients)
        weight_sums = [np.zeros_like(var) for var in self.global_model.get_weights()]

        for client in self.clients:
            weights = client.get_weights()
            for i in range(len(weight_sums)):
                weight_sums[i] += np.array(weights[i])

        # Average weights
        for i in range(len(weight_sums)):
            weight_sums[i] /= num_clients

        self.global_model.set_weights(weight_sums)
    
    def evaluate(self, test_data):
        (test_images, test_labels) = test_data
        test_images = test_images / 255.0
        test_loss, test_acc = self.global_model.evaluate(test_images, test_labels, verbose=2)
        return test_loss, test_acc
