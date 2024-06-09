import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, medfilt
import pywt


def load_eeg_data(filepath, header='infer', column_names=None):
    try:
        # Učitamo CSV fajl u DataFrame
        eeg_data = pd.read_csv(filepath, header=header)

        # Ako je zaglavlje None, postavimo imena kolona
        if header is None and column_names is not None:
            eeg_data.columns = column_names

        return eeg_data
    except Exception as e:
        print("Došlo je do greške prilikom učitavanja fajla: {e}")
        return None


def plot_eeg(data, channels=None, start=0, end=None, sampling_rate=1, title='EEG Signal'):
    """
    Funkcija za plotanje EEG signala.

    :param data: DataFrame sa EEG podacima.
    :param channels: Lista kanala za plotanje. Ako je None, plotaju se svi kanali.
    :param start: Početni uzorak za plotanje.
    :param end: Krajnji uzorak za plotanje. Ako je None, plotaju se svi uzorci do kraja.
    :param sampling_rate: Frekvencija uzorkovanja u Hz (koristi se za kreiranje vremenske ose).
    :param title: Naslov grafa.
    """
    if channels is None:
        channels = data.columns

    if end is None:
        end = len(data)

    # Kreiranje vremenske ose
    time = (start + np.arange(end - start)) / sampling_rate

    plt.figure(figsize=(15, 8))
    for channel in channels:
        plt.plot(time, data[channel].iloc[start:end], label=channel)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.show()


def notch_filter(data, freq, fs, quality_factor=30):
    """
    Primjena Notch filtera na EEG podatke.

    :param data: EEG podaci (DataFrame).
    :param freq: Frekvencija koja se uklanja (Hz).
    :param fs: Frekvencija uzorkovanja (Hz).
    :param quality_factor: Kvalitet filtera.
    :return: Filtrirani podaci (DataFrame).
    """
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, quality_factor)
    filtered_data = data.apply(lambda x: filtfilt(b, a, x), axis=0)
    return filtered_data


def fourier_transform(data, fs):
    freqs = np.fft.fftfreq(len(data), 1/fs)
    fft_values = np.fft.fft(data)
    return freqs, np.abs(fft_values)


def wavelet_transform(data, wavelet='db5', level=4):
    """
    Primjena Wavelet transformacije na EEG podatke.

    :param data: EEG podaci (DataFrame).
    :param wavelet: Tip wavelet-a (npr. 'db4').
    :param level: Broj nivoa dekompozicije.
    :return: Koeficijenti wavelet transformacije.
    """
    coeffs = {}
    for column in data.columns:
        coeffs[column] = pywt.wavedec(data[column], wavelet, level=level)
    return coeffs


def plot_wavelet_coeffs(coeffs, channels, sampling_rate=1, title='Wavelet Coefficients'):
    """
    Funkcija za plotanje wavelet koeficijenata.

    :param coeffs: Koeficijenti wavelet transformacije.
    :param channels: Lista kanala za plotanje.
    :param sampling_rate: Frekvencija uzorkovanja u Hz (koristi se za kreiranje vremenske ose).
    :param title: Naslov grafa.
    """
    num_levels = len(next(iter(coeffs.values())))

    plt.figure(figsize=(15, 8))
    for channel in channels:
        plt.subplot(len(channels), 1, channels.index(channel) + 1)
        for i, coef in enumerate(coeffs[channel]):
            plt.plot(coef, label=f'Level {i + 1}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title(f'{title} - {channel}')
        plt.legend()

    plt.tight_layout()
    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Kreira band-pass Butterworth filter.

    :param lowcut: Donja frekvencijska granica (Hz).
    :param highcut: Gornja frekvencijska granica (Hz).
    :param fs: Frekvencija uzorkovanja (Hz).
    :param order: Red filtra.
    :return: Koeficijenti filtra.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Primjena band-pass Butterworth filtera na EEG podatke.

    :param data: EEG podaci (DataFrame).
    :param lowcut: Donja frekvencijska granica (Hz).
    :param highcut: Gornja frekvencijska granica (Hz).
    :param fs: Frekvencija uzorkovanja (Hz).
    :param order: Red filtra.
    :return: Filtrirani podaci (DataFrame).
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = data.apply(lambda x: filtfilt(b, a, x), axis=0)
    return filtered_data


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def apply_median_filter(data, kernel_size=3):
    return data.apply(lambda x: medfilt(x, kernel_size), axis=0)


def main():
    file_path = 'test.csv'
    eeg_data = load_eeg_data(file_path)
    filtered_data = butter_bandpass_filter(eeg_data, 0.5, 30, 250)
    median_filtered = apply_median_filter(eeg_data)
    wavelet_coeffs = wavelet_transform(eeg_data, wavelet='db5', level=4)
    cutoff = 30.0  # Frekvencija odsecanja
    lowpassed_data = eeg_data.apply(lambda x: butter_lowpass_filter(x, cutoff, 250), axis=0)

    if eeg_data is not None:
        print(eeg_data.head())
        print(wavelet_coeffs)
        eeg_data = load_eeg_data(file_path)

    if eeg_data is not None:
        plot_eeg(eeg_data, channels=None, start=0, end=1000, sampling_rate=250, title='Originalni EEG signal')
        plot_wavelet_coeffs(wavelet_coeffs, channels=['AF4'], sampling_rate=250)
        plot_eeg(filtered_data, channels=None, start=0, end=1000, sampling_rate=250, title='Butterworth EEG signal')
        plot_eeg(median_filtered, channels=None, start=0, end=1000, sampling_rate=250,title='Median filter')
        plot_eeg(lowpassed_data,channels=None,start=0,end=1000,sampling_rate=250,title='Lowpass')


if main is not None:
    main()