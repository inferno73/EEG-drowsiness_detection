import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt
from scipy.fft import fft
import pywt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Apply Notch filter to remove power line noise
def notch_filter(data, freq, fs, quality_factor=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, quality_factor)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

# Extract more statistical features
def extract_statistical_features(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    min_val = np.min(data, axis=1)
    max_val = np.max(data, axis=1)
    median = np.median(data, axis=1)
    q25 = np.percentile(data, 25, axis=1)
    q75 = np.percentile(data, 75, axis=1)
    iqr = q75 - q25
    skewness = np.apply_along_axis(lambda x: pd.Series(x).skew(), 1, data)
    kurtosis = np.apply_along_axis(lambda x: pd.Series(x).kurt(), 1, data)
    features = np.vstack((mean, std, min_val, max_val, median, q25, q75, iqr, skewness, kurtosis)).T
    return features

# Apply Wavelet Transform
def wavelet_transform(data, wavelet='db5', level=4):
    coeffs = [pywt.wavedec(d, wavelet, level=level) for d in data]
    features = np.array([np.concatenate(c) for c in coeffs])
    return features

# Apply Fourier Transform
def fourier_transform(data):
    fft_features = np.abs(fft(data, axis=1))
    return fft_features

# Preprocess the data
def preprocess_data(data, fs=250):
    # Drop rows with any NaN values
    data = data.dropna()

    # Extract labels
    y = data['eyeDetection'].values

    # Apply Notch filter to remove power line noise at 50Hz
    filtered_data = notch_filter(data.drop(columns=['eyeDetection']).values, 50, fs)

    # Extract statistical features
    statistical_features = extract_statistical_features(filtered_data)

    # Apply Wavelet Transform
    wavelet_features = wavelet_transform(filtered_data)

    # Apply Fourier Transform
    fourier_features = fourier_transform(filtered_data)

    # Combine all features
    X = np.hstack((statistical_features, wavelet_features, fourier_features))

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Create and compile the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Custom early stopping function
def custom_early_stopping(history, threshold=0.94):
    accuracies = history.history['accuracy']
    if any(acc >= threshold for acc in accuracies):
        return True
    return False

def main():
    # Load and preprocess the data
    data = load_data('EEG_Eye_State_Classification.csv')  # Replace with your file path
    fs = 250  # Example sampling rate, replace with your actual sampling rate
    X, y = preprocess_data(data, fs)

    # Reshape data for CNN
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Initialize KFold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the model
        model = create_cnn_model(X_train.shape[1:])

        for epoch in range(100):  # Manually iterate over epochs
            history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            if custom_early_stopping(history, threshold=0.94):
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        accuracies.append(accuracy)
        print(f"Fold Accuracy: {accuracy}")

    print(f"Mean Cross-Validation Accuracy: {np.mean(accuracies)}")

    # Plot the accuracy
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#saving model so we can test it
    model.save('trained_model.h5')
    #keras.saving.save_model(model)

if __name__ == "__main__":
    main()
