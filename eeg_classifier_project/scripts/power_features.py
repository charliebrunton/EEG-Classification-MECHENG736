import numpy as np

def compute_band_powers(epoch_data, fs=200):
    # compute alpha and theta band powers
    # input: epoch_data -> column vector for a single epoch (data sampled at fs)
    # outputs: pow_a -> alpha band power, pow_t -> theta band power

    # convert to freq domain
    n = len(epoch_data)
    fft_vals = np.fft.rfft(epoch_data) / n # one-sided FFT (rfft -> 0Hz to Nyquist), normalised by number of samples in epoch 
                                           # (FFT scales with signal length -> power shouldnt)
    # fft_vals -> array of bins, need corresponding freq axis
    freqs = np.fft.rfftfreq(n, d=1/fs) # maps each FFT bin to its frequency

    # convert to power spectrum
    amplitudes = np.abs(fft_vals)
    power_spectrum = amplitudes ** 2 * 2 # power (average, instantaneous)= amplitude^2, * 2 (for both sides of FFT)
    power_spectrum[0] /= 2 # dont double DC component

    # mask freq bins
    theta_mask = (freqs >= 4) & (freqs < 8) # boolean array
    alpha_mask = (freqs >= 8) & (freqs <= 12)

    # average power
    pow_a = np.mean(power_spectrum[alpha_mask])
    pow_t = np.mean(power_spectrum[theta_mask])

    # convert to dB -> may improve classification accuracy
    pow_a = 10 * np.log10(pow_a)
    pow_t = 10 * np.log10(pow_t)

    return pow_a, pow_t
