"""
Programmer: Chris Tralie
Purpose: Various audio tools to support fundamental frequency
tracking and pitch shifting
"""
import numpy as np
from scipy.io import wavfile

def load_audio(path):
    """
    Load in a wav file, mix down to mono, and normalize to [-1, 1]
    
    Parameters
    ----------
    path: string
        Path to file
    
    Returns
    -------
    sr: int 
        Sample rate
    x: ndarray(N)
        Audio samples
    """
    sr, x = wavfile.read(path)
    x = np.array(x, dtype=float)
    if len(x.shape) > 1:
        # Mix down multichannel audio
        x = np.mean(x, axis=1)
    # Normalize audio
    x = x/np.max(np.abs(x))
    return x, sr

def save_audio(x, sr, path):
    """
    Parameters
    -------
    sr: int 
        Sample rate
    x: ndarray(N)
        Audio samples
    path: string
        Path to which to write file
    """
    x = np.array(32768*x/np.max(np.abs(x)), dtype=np.int16)
    wavfile.write(path, sr, x)


def frame_audio(x, hop_length, frame_length):
    """
    Separate audio into overlapping frames

    Parameters
    ----------
    x: ndarray(N)
        Audio
    hop_length: int
        Hop length between frames
    frame_length: int
        Frame length
    
    Returns
    -------
    F: ndarray(frame_length, ceil(N-frame_length/hop_length)+1)
    """
    M = int(np.ceil((x.size-frame_length)/hop_length)+1)
    F = np.zeros((frame_length, M), dtype=np.float32)
    for i in range(M):
        i1 = i*hop_length
        chunk = x[i1:i1+frame_length]
        F[0:chunk.size, i] = chunk
    return F

def get_yin_freqs(x, frame_length, sr, hop_length=None, win_length=None, fmin=100, fmax=2000):
    """
    Compute a set of frequency candidates using yin

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    frame_length: int
        Length of each analysis frame
    sr: int
        Sample rate
    hop_length: int
        Hop between frames.  By default, frame_length//4
    win_length: int
        Window length to use in FFT.  By default, frame_length//2
    fmin: float
        Minimum frequency to consider (in hz)
    fmax: float
        Maximum frequency to consider (in hz)
    
    Returns
    -------
    freqs: list of lists of [float, float] 
        List of of each frequency at a different threshold for each 
        time window.  For instance, freqs[0] is a list of all 
        [freq, thresh] frequencies at different thresholds at the
        first time instant
    
    """
    if not hop_length:
        hop_length = frame_length//4
    if not win_length:
        win_length = frame_length//2
    TMin = int(sr/fmax) # Min period
    TMax = int(sr/fmin) # Max period

    F = frame_audio(x, hop_length, frame_length)

    # Windowed energy
    energy = np.cumsum(F**2, axis=0)
    energy = energy[win_length::, :] - energy[0:-win_length, :]
    energy[energy < 1e-6] = 0

    ## Step 1: Do autocorrelation
    a = np.fft.rfft(F, axis=0)
    b = np.fft.rfft(F[0:win_length, :], frame_length, axis=0) # Cut out only window length and zeropad
    acf = np.fft.irfft(a*np.conj(b), axis=0)[0:win_length, :]
    acf[np.abs(acf) < 1e-6] = 0

    ## Step 2: Compute windowed energy
    energy = np.cumsum(F**2, axis=0)
    energy = energy[win_length::, :] - energy[0:-win_length, :]

    ## Step 3: Yin and normalized yin
    yin = energy[0, :] + energy - 2*acf
    denom = np.cumsum(yin[1::, :], axis=0)
    denom[denom<1e-6] = 1
    nyin = np.ones(yin.shape)
    nyin[1::, :] = yin[1::, :]*np.arange(1, yin.shape[0])[:, None]/denom

    ## Step 4: Parabolic interpolation using original yin
    offsets = np.zeros_like(yin)
    left = yin[0:-2, :]
    center = yin[1:-1, :]
    right = yin[2:, :]
    a = (right + left)/2 - center
    b = (right - left)/2
    offsets[1:-1, :] = -b/(2*a)
    offsets[np.abs(offsets) > 1] = 0 # Make sure we don't move by more than 1

    ## Step 5: Find mins within frequency range
    left   = nyin[TMin-1:TMax, :]
    center = nyin[TMin:TMax+1, :]
    right  = nyin[TMin+1:TMax+2, :]
    Ts = np.arange(TMin, TMax+1)[:, None]*np.ones((1, nyin.shape[1])) + offsets[TMin:TMax+1, :]
    freqs = []
    nyin = nyin[TMin:TMax+1, :]
    mins = (center < left)*(center < right)
    for j in range(Ts.shape[1]):
        fj = sr/Ts[mins[:, j], j]
        threshs = nyin[mins[:, j], j]
        fj = fj[(threshs >= 0)*(threshs <= 1)]
        threshs = threshs[(threshs >= 0)*(threshs <= 1)]
        freqs.append(list(zip(fj, threshs)))
    return freqs


hann_fn = lambda win: 0.5*(1-np.cos(2*np.pi*np.arange(win)/win))

# w =  2048
# h = 128

def stft(x, w, h):
    """
    Compute the complex Short-Time Fourier Transform (STFT)
    using a Hann window

    Parameters
    ----------
    x: ndarray(N)
        Full audio clip of N samples
    w: int
        Window length
    h: int
        Hop length
    
    Returns
    -------
    ndarray(w, nwindows, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-w)/h))+1
    S = np.zeros((w//2+1, nwin), dtype=complex)
    hann = hann_fn(w)
    for j in range(nwin):
        xj = x[h*j:h*j+w]
        S[:, j] = np.fft.rfft(hann[0:xj.size]*xj, w)
    return S

def istft(S, w, h):
    """
    Compute the complex inverse Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    S: ndarray(w, nwindows, dtype=np.complex)
        Complex spectrogram
    w: int
        Window length
    h: int
        Hop length
    
    Returns
    -------
    y: ndarray(N)
        Audio samples of the inverted STFT
    """
    N = (S.shape[1]-1)*h + w # Number of samples in result
    y = np.zeros(N)
    for j in range(S.shape[1]):
        y[j*h:j*h+w] += np.fft.irfft(S[:, j])
    y /= (w/h/2)
    return y

def griffin_lim(SAbs, w, h, n_iters=10):
    """
    Perform Griffin-Lim inversion on a magnitude spectrogram

    Parameters
    ----------
    S: ndarray(win_length//2+1, 1+2*(n_samples-win_length)/win_length)
        Real windowed STFT
    w: int
        Window length
    h: int
        Hop length
    """
    A = SAbs
    for _ in range(n_iters):
        S = stft(istft(A, w, h), w, h)
        P = np.arctan2(np.imag(S), np.real(S))
        A = np.abs(SAbs)*np.exp(1j*P)
    return istft(A, w, h)