import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, welch

# Load EEG data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Define the notch filter function
def notch_filter(data, freq, fs, quality_factor=30):
    nyq = 0.5 * fs
    norm_freq = freq / nyq
    b, a = iirnotch(norm_freq, quality_factor)
    filtered_data = data.apply(lambda x: filtfilt(b, a, x), axis=0)
    return filtered_data

# Plotting function for time-domain signals
def plot_time_domain(data, title):
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(data[column], label=column)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Plotting function for frequency-domain signals
def plot_frequency_domain(data, fs, title):
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        f, Pxx = welch(data[column], fs, nperseg=1024)
        plt.semilogy(f, Pxx, label=column)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.legend()
    plt.show()

# Set parameters
fs = 256  # Example sampling frequency, replace with actual one if different
freq_to_remove = 50  # Example frequency to remove, typically 50 or 60 Hz

# Load data
file_path = 'train.csv'
data = load_data(file_path)

# Plot original time-domain signal
plot_time_domain(data, 'Original EEG Signal')

# Plot original frequency-domain signal
plot_frequency_domain(data, fs, 'Original Signal Spectrum')

# Apply notch filter
filtered_data = notch_filter(data, freq_to_remove, fs)

# Plot filtered time-domain signal
plot_time_domain(filtered_data, 'Filtered EEG Signal')

# Plot filtered frequency-domain signal
plot_frequency_domain(filtered_data, fs, 'Filtered Signal Spectrum')
