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
    amp = 1
    if len(x) > 0:
        amp = get_amplitude(x, hop_length)
        N = min(y.size, amp.size)
        amp, y = amp[0:N], y[0:N]
    return amp*y

def get_scale(root_note, major, fmin=100, fmax=2000):
    """
    Compute a list of frequencies that are involved in a scale

    Parameters
    ----------
    root_note: str
        Root note of scale (e.g. "A Flat", "A Sharp")
    major: bool
        If True, use a major version of the scale.  If False, use a minor version
    fmin: float
        Minimum frequency to consider (in hz)
    fmax: float
        Maximum frequency to consider (in hz)
    
    Returns
    -------
    freqs: ndarray(N)
        List of frequencies corresponding to allowed notes
    """
    root = root_note.split()[0].lower()
    roots = {"a":0, "b":2, "c":3, "d":5, "e":7, "f":8, "g":10}
    intervals = [[2, 1, 2, 2, 1, 2, 2], [2, 2, 1, 2, 2, 2, 1]][major]
    lower = []
    start = roots[root]
    if "flat" in root_note.lower():
        start -= 1
    if "sharp" in root_note.lower():
        start += 1
    note = start
    i = -1
    f = 440*2**(note/12)
    while f > fmin:
        i = (i + len(intervals))%len(intervals)
        note -= intervals[i]
        i -= 1
        f = 440*2**(note/12)
        if f >= fmin:
            lower.append(f)
    lower.reverse()

    upper = []
    note = start
    f = 440*2**(note/12)
    i = 0
    while f < fmax:
        f = 440*2**(note/12)
        upper.append(f)
        i = (i + len(intervals))%len(intervals)
        note += intervals[i]
        i += 1
    
    return np.array(lower + upper)


def autotune(x, freqs, hop_length, allowed_notes, hop_scale=8, n_iters=10, wiggle_amt=np.inf, wiggle_len=None):
    """
    Parameters
    ----------
    x: ndarray(N)
        Original audio samples
    freqs: ndarray(N//hop_length)
        List of frequencies in each hop.  Nan if not confident enough
    hop_length: int
        Hop length between frequency estimates
    allowed_notes: list of float
        Allowed notes in the autotuner
    hop_scale: int
        Factor by which to interpolate the hop length for better results with Griffin Lim
    n_iters: int
        Number of iterations of Griffin Lim
    wiggle_amt: int
        If there's a jump by more than this amount, then wiggle
    wiggle_len: int
        Length of samples to wiggle.  If None, use hop_scale*4
    """
    from audiotools import stft, griffin_lim

    ## Step 1: Expand frequencies by a factor of hop_scale
    freqs = (freqs[:, None]*np.ones((1, hop_scale))).flatten()
    win_length = hop_length*2
    hop_length = hop_length//hop_scale

    ## Step 2: Figure out frequency ratios by rounding to the nearest allowed frequency in a scale
    allowed_notes = np.array(allowed_notes)
    idx = np.argmin(np.abs(allowed_notes[:, None] - freqs[None, :]), axis=0)
    ## Step 2b: Wiggle notes if requested for that signature effect
    if np.isfinite(wiggle_amt):
        if not wiggle_len:
            wiggle_len = 4*hop_scale
        i = 0
        while i < len(idx):
            di = (idx[i] - idx[i-1])
            if np.abs(di) > wiggle_amt:
                idx[i:i+wiggle_len] += np.random.choice([1, -1])
                i += wiggle_len
                idx[i:i+wiggle_len] += np.random.choice([1, -1])
                i += wiggle_len
            else:
                i += 1
    notes = allowed_notes[idx]
    ratios = freqs/notes
    ratios[np.isnan(ratios)] = 1 # If there was not a confident frequency estimate, don't shift anything

    ## Step 3: Do interpolation and apply griffin lim
    S = np.abs(stft(x, win_length, hop_length))
    S2 = np.zeros((S.shape[0], min(S.shape[1], ratios.size)))
    f = np.arange(S.shape[0])
    for j in range(S2.shape[1]):
        S2[:, j] = np.interp(f*ratios[j], f, S[:, j])
    return notes, griffin_lim(S2, win_length, hop_length, n_iters)

def fm_synth(x, freqs, hop_length, sr, ratio=1, I=8):
    """
    Do fm synthesis using an estimated instantaneous frequency
    in freqs and an envelope based on instantaneous amplitude estimates

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
    ratio: int
        Ratio of harmonics
    I: float
        Bandwidth index for envelope scaling.  Higher values make it "scratchier"
    
    Returns
    -------
    ndarray(<=N)
        FM synth approximation of x
    """
    env = get_amplitude(x, hop_length)
    freqs = np.array(freqs)
    freqs = freqs[:, None]*np.ones((1, hop_length))
    freqs = freqs.flatten()
    freqs[np.isnan(freqs)] = 0
    f = np.cumsum(freqs)/sr
    N = min(f.size, env.size)
    f = f[0:N]
    env = env[0:N]
    return env*np.cos(2*np.pi*f + I*env*np.sin(2*np.pi*ratio*f))