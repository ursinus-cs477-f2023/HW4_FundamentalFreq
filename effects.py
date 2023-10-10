import numpy as np

def get_amplitude(x, win_length):
    """
    Compute estimates of instantaneous amplitude of the signal

    Parameters
    ----------
    x: ndarray(N)
        Original audio samples
    win_length: int
        Window length to use for energy calculations
    
    Returns
    -------
    ndarray(N-win_length)
        Amplitude estimates in each window
    """
    energy = np.cumsum(x**2)
    return np.sqrt((energy[win_length::] - energy[0:-win_length])/win_length)


def sonify_pure_tone(x, freqs, hop_length, sr):
    """
    Sonify instantaneous frequency estimates with numerical integration inside
    of a pure cosine

    Parameters
    ----------
    x: ndarray(N)
        Original audio samples
    freqs: ndarray(N//hop_length)
        List of frequencies in each hop.  Nan if not confident enough
    hop_length: int
        Hop length between frequency estimates
    sr: int
        Audio sample rate
    """
    freqs = np.array(freqs)
    freqs = freqs[:, None]*np.ones((1, hop_length))
    freqs = freqs.flatten()
    freqs[np.isnan(freqs)] = 0
    y = np.cos(2*np.pi*np.cumsum(freqs)/sr)
    energy = get_amplitude(x, hop_length)
    N = min(y.size, energy.size)
    return energy[0:N]*y[0:N]


def autotune(x, freqs, hop_length, sr):
    pass