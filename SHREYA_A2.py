import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.regularizers import l2

# Data Generator Class from the First File
class ECG200DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_filepath, batch_size):
        df = pd.read_csv(dataset_filepath, sep='\t', header=None)
        labels = df.iloc[:, 0].values
        sequences = df.iloc[:, 1:].values.astype(np.float32)
        sequences = sequences[:, :, np.newaxis]
        
        # Normalize each sequence individually
        for i in range(len(sequences)):
            seq = sequences[i, :, 0]
            mean = np.mean(seq)
            std = np.std(seq)
            if std > 0:
                sequences[i, :, 0] = (seq - mean) / std
            else:
                sequences[i, :, 0] = seq - mean
        
        labels_mapped = (labels + 1) // 2
        self.labels_onehot = tf.keras.utils.to_categorical(labels_mapped, num_classes=2)
        self.sequences = sequences
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.sequences))
        x = self.sequences[start_idx:end_idx]
        y = self.labels_onehot[start_idx:end_idx]
        return x, y

    def on_epoch_end(self):
        indices = np.arange(len(self.sequences))
        np.random.shuffle(indices)
        self.sequences = self.sequences[indices]
        self.labels_onehot = self.labels_onehot[indices]

# CNN Model from the Second File
def ecg200_cnn_model(training_data_filepath, validation_data_filepath, kernel_size=5, epochs=50):
    batch_size = 32
    train_gen = ECG200DataGenerator(training_data_filepath, batch_size)
    val_gen = ECG200DataGenerator(validation_data_filepath, batch_size)
    sequence_length = train_gen.sequences.shape[1]
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(sequence_length, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=0)
    
    train_perf = model.evaluate(train_gen, verbose=0)
    val_perf = model.evaluate(val_gen, verbose=0)
    
    return model, tuple(train_perf), tuple(val_perf), history

# RNN Model from the First File
def ecg200_rnn_model(training_data_filepath, validation_data_filepath, hidden_units=128, epochs=50):
    batch_size = 32
    train_gen = ECG200DataGenerator(training_data_filepath, batch_size)
    val_gen = ECG200DataGenerator(validation_data_filepath, batch_size)
    sequence_length = train_gen.sequences.shape[1]
    
    model = Sequential([
        LSTM(units=hidden_units, input_shape=(sequence_length, 1)),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=0)
    
    train_perf = model.evaluate(train_gen, verbose=0)
    val_perf = model.evaluate(val_gen, verbose=0)
    
    return model, tuple(train_perf), tuple(val_perf), history

# Helper function for kernel size experiment (CNN)
def experiment_kernel_size(training_data_filepath, validation_data_filepath):
    kernel_sizes = [3, 5, 7, 9]
    train_accuracies = []
    val_accuracies = []
    
    for kernel_size in kernel_sizes:
        print(f"Training CNN with kernel size: {kernel_size}")
        _, train_perf, val_perf, _ = ecg200_cnn_model(
            training_data_filepath, validation_data_filepath, kernel_size=kernel_size)
        train_accuracies.append(train_perf[1])
        val_accuracies.append(val_perf[1])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(kernel_sizes, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(kernel_sizes, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Kernel Size')
    plt.ylabel('Accuracy')
    plt.title('CNN Performance vs Kernel Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_kernel_size_experiment.png')
    plt.close()
    
    return kernel_sizes, train_accuracies, val_accuracies

# Helper function for hidden state size experiment (RNN)
def experiment_hidden_units(training_data_filepath, validation_data_filepath):
    hidden_units = [32, 64, 128, 256]
    train_accuracies = []
    val_accuracies = []
    
    for units in hidden_units:
        print(f"Training RNN with hidden units: {units}")
        _, train_perf, val_perf, _ = ecg200_rnn_model(
            training_data_filepath, validation_data_filepath, hidden_units=units)
        train_accuracies.append(train_perf[1])
        val_accuracies.append(val_perf[1])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_units, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(hidden_units, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Hidden Units')
    plt.ylabel('Accuracy')
    plt.title('RNN Performance vs Hidden Units')
    plt.legend()
    plt.grid(True)
    plt.savefig('rnn_hidden_units_experiment.png')
    plt.close()
    
    return hidden_units, train_accuracies, val_accuracies

# Helper function for epochs experiment
def experiment_epochs(training_data_filepath, validation_data_filepath):
    epochs_range = [10, 20, 30, 40, 50]
    
    # CNN experiment
    cnn_train_accuracies = []
    cnn_val_accuracies = []
    for epochs in epochs_range:
        print(f"Training CNN with epochs: {epochs}")
        _, train_perf, val_perf, _ = ecg200_cnn_model(
            training_data_filepath, validation_data_filepath, epochs=epochs)
        cnn_train_accuracies.append(train_perf[1])
        cnn_val_accuracies.append(val_perf[1])
    
    # RNN experiment
    rnn_train_accuracies = []
    rnn_val_accuracies = []
    for epochs in epochs_range:
        print(f"Training RNN with epochs: {epochs}")
        _, train_perf, val_perf, _ = ecg200_rnn_model(
            training_data_filepath, validation_data_filepath, epochs=epochs)
        rnn_train_accuracies.append(train_perf[1])
        rnn_val_accuracies.append(val_perf[1])
    
    # Plotting CNN
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, cnn_train_accuracies, label='CNN Training Accuracy', marker='o')
    plt.plot(epochs_range, cnn_val_accuracies, label='CNN Validation Accuracy', marker='o')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('CNN Performance vs Number of Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_epochs_experiment.png')
    plt.close()
    
    # Plotting RNN
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, rnn_train_accuracies, label='RNN Training Accuracy', marker='o')
    plt.plot(epochs_range, rnn_val_accuracies, label='RNN Validation Accuracy', marker='o')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('RNN Performance vs Number of Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('rnn_epochs_experiment.png')
    plt.close()
    
    return (epochs_range, cnn_train_accuracies, cnn_val_accuracies,
            rnn_train_accuracies, rnn_val_accuracies)

# Helper function to find best models and evaluate on test set
def evaluate_best_models(training_data_filepath, validation_data_filepath, test_data_filepath):
    # Find best kernel size for CNN
    kernel_sizes, cnn_train_accs, cnn_val_accs = experiment_kernel_size(
        training_data_filepath, validation_data_filepath)
    best_kernel_size = kernel_sizes[np.argmax(cnn_val_accs)]
    
    # Find best hidden units for RNN
    hidden_units, rnn_train_accs, rnn_val_accs = experiment_hidden_units(
        training_data_filepath, validation_data_filepath)
    best_hidden_units = hidden_units[np.argmax(rnn_val_accs)]
    
    # Train final models with best parameters
    print(f"Training final CNN with kernel size: {best_kernel_size}")
    best_cnn_model, cnn_train_perf, cnn_val_perf, _ = ecg200_cnn_model(
        training_data_filepath, validation_data_filepath, kernel_size=best_kernel_size)
    
    print(f"Training final RNN with hidden units: {best_hidden_units}")
    best_rnn_model, rnn_train_perf, rnn_val_perf, _ = ecg200_rnn_model(
        training_data_filepath, validation_data_filepath, hidden_units=best_hidden_units)
    
    # Evaluate on test set
    test_gen = ECG200DataGenerator(test_data_filepath, batch_size=32)
    cnn_test_perf = best_cnn_model.evaluate(test_gen, verbose=0)
    rnn_test_perf = best_rnn_model.evaluate(test_gen, verbose=0)
    
    return (best_cnn_model, best_kernel_size, cnn_train_perf, cnn_val_perf, cnn_test_perf,
            best_rnn_model, best_hidden_units, rnn_train_perf, rnn_val_perf, rnn_test_perf)

# Main Execution
if __name__ == "__main__":
    training_data_filepath = "ECG200_TRAIN.tsv"
    validation_data_filepath = "ECG200_VALIDATION.tsv"
    test_data_filepath = "ECG200_TEST.tsv"
    
    print("Running experiment on kernel sizes (CNN)")
    kernel_sizes, cnn_train_accs, cnn_val_accs = experiment_kernel_size(
        training_data_filepath, validation_data_filepath)
    print("Kernel sizes experiment results:")
    print("Kernel sizes:", kernel_sizes)
    print("Training accuracies:", cnn_train_accs)
    print("Validation accuracies:", cnn_val_accs)
    print("Comment: Generally, larger kernel sizes might capture broader patterns but could lead to "
          "information loss due to larger receptive fields. The optimal kernel size depends on the specific "
          "patterns in the ECG data.")
    
    print("\nRunning experiment on hidden units (RNN)")
    hidden_units, rnn_train_accs, rnn_val_accs = experiment_hidden_units(
        training_data_filepath, validation_data_filepath)
    print("Hidden units experiment results:")
    print("Hidden units:", hidden_units)
    print("Training accuracies:", rnn_train_accs)
    print("Validation accuracies:", rnn_val_accs)
    print("Comment: More hidden units generally improve model capacity but may lead to overfitting "
          "if too large. There's typically a sweet spot where validation performance peaks.")
    
    print("\nRunning experiment on number of epochs")
    epochs_results = experiment_epochs(training_data_filepath, validation_data_filepath)
    print("Epochs experiment results:")
    print("Epochs:", epochs_results[0])
    print("CNN Training accuracies:", epochs_results[1])
    print("CNN Validation accuracies:", epochs_results[2])
    print("RNN Training accuracies:", epochs_results[3])
    print("RNN Validation accuracies:", epochs_results[4])
    print("Comment: More epochs generally improve training performance but may lead to overfitting "
          "on validation data. The optimal number of epochs balances training and validation performance.")
    
    print("\nEvaluating best models on test set")
    results = evaluate_best_models(training_data_filepath, validation_data_filepath, test_data_filepath)
    
    print("\nBest CNN Results:")
    print(f"Best kernel size: {results[1]}")
    print(f"Training Performance (loss, accuracy): {results[2]}")
    print(f"Validation Performance (loss, accuracy): {results[3]}")
    print(f"Test Performance (loss, accuracy): {results[4]}")
    
    print("\nBest RNN Results:")
    print(f"Best hidden units: {results[6]}")
    print(f"Training Performance (loss, accuracy): {results[7]}")
    print(f"Validation Performance (loss, accuracy): {results[8]}")
    print(f"Test Performance (loss, accuracy): {results[9]}")