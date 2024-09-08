import tqdm
import random
import os
import argparse
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vae_utility import train_vae, detect_malicious_modifications, vae_threshold, detect_malicious_modifications_2
from utility import *
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_settings(settings_file):
    spec = importlib.util.spec_from_file_location("settings", settings_file)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


def load_data(dataset_key):
    if dataset_key == "mnist":
        return tf.keras.datasets.mnist.load_data()
    elif dataset_key == "cifar10":
        return tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")

def create_model(dataset_key):
    if dataset_key == "mnist":
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    elif dataset_key == "cifar10":
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def federated_learning(dataset_key, num_clients, num_rounds):
    global_model = create_model(dataset_key)
    print(f"Building {num_clients} virtual client and generating {num_rounds} instances")
    client_data_chunks = data_distribution(dataset_key, num_clients)
    clients = [Client(client_id, data_chunk) for client_id, data_chunk in enumerate(client_data_chunks)]    
    global_models = []
    for _ in tqdm.tqdm(range(num_rounds)):
        for client in clients:
            client.set_global_model(global_model)
        client_updates = [client_update(client.get_local_model(), client.get_data()) for client in clients]
        aggregated_gradients = server_aggregate(global_model, client_updates)
        global_model.optimizer.apply_gradients(zip(aggregated_gradients, global_model.trainable_variables))
        global_models.append(global_model)
    
    return model_to_vector(global_models)

def train_and_evaluate_vae(global_models):
    x_train, x_test = train_test_split(global_models, test_size=0.2, random_state=42)
    print("# Define VAE Hyper-parameters")
    learning_param = 0.001
    epochs = 3000
    batch_size = 32
    input_dimension = global_models[0].shape[0]
    neural_network_dimension = 512
    latent_variable_dimension = 2
    print("# Train the VAE")
    total_losses, _, _, Final_Weight, Final_Bias = train_vae(x_train, epochs, batch_size, input_dimension, neural_network_dimension, latent_variable_dimension, learning_param)
    threshold = vae_threshold(total_losses)
    return x_test, Final_Weight, Final_Bias, threshold

def detect_anomalies(x_test, Final_Weight, Final_Bias, threshold):
    print("# Test the VAE")
    print("Testing benign instances from x_test")
    reconstruction_errors = detect_malicious_modifications(x_test, Final_Weight, Final_Bias)
    malicious_indices = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
    print("Potentially malicious modifications detected at indices:", malicious_indices)
    return malicious_indices
def main(settings_file, num_clients, num_rounds):
    settings = load_settings(settings_file)
    dataset_key = settings.dataset_key
    fcnn = create_model(dataset_key)
    (train_images, train_labels), (test_images, test_labels) = load_data(dataset_key)
    # Preprocess data
    if dataset_key == "mnist":
        train_images, test_images = train_images / 255.0, test_images / 255.0
    elif dataset_key == "cifar10":
        train_images, test_images = train_images / 255.0, test_images / 255.0
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    
    global_models = federated_learning(dataset_key, num_clients, num_rounds)
    x_test, Final_Weight, Final_Bias, threshold = train_and_evaluate_vae(global_models)
    detect_anomalies(x_test, Final_Weight, Final_Bias, threshold)
    #fcnn.fit(train_images, train_labels, epochs=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on a dataset, Number of clients, Number of rounds")
    parser.add_argument('-s', '--settings', default="default_settings.py", required=True, help="Path to the settings file")
    parser.add_argument("-n", "--num_clients", type=int, default=50, help="Number of clients to use. Default is 50.")
    parser.add_argument("-r", "--num_rounds", type=int, default=100, help="Number of rounds to run. Default is 100.")
    args = parser.parse_args()
    main(args.settings, args.num_clients, args.num_rounds)
