"""
EcoSonic AI - Audio Processing Module
Handles audio file loading, feature extraction, and visualization data.
"""
import numpy as np
import librosa
import io
import os


def load_audio(file_path=None, file_bytes=None, sr=22050):
    """Load audio from file path or bytes. Returns (signal, sample_rate, duration)."""
    try:
        if file_bytes is not None:
            audio_buffer = io.BytesIO(file_bytes)
            y, sr_loaded = librosa.load(audio_buffer, sr=sr, mono=True)
        elif file_path is not None:
            y, sr_loaded = librosa.load(file_path, sr=sr, mono=True)
        else:
            raise ValueError("Provide either file_path or file_bytes")
        duration = librosa.get_duration(y=y, sr=sr_loaded)
        return y, sr_loaded, duration
    except Exception as e:
        raise RuntimeError(f"Error loading audio: {str(e)}")


def extract_mfcc(y, sr, n_mfcc=13):
    """Extract MFCC features."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    return mfccs, mfccs_mean, mfccs_std


def extract_spectral_features(y, sr):
    """Extract spectral features."""
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_rolloff_std": float(np.std(spectral_rolloff)),
        "spectral_contrast_mean": float(np.mean(spectral_contrast)),
    }


def extract_temporal_features(y, sr):
    """Extract temporal features."""
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    return {
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "zero_crossing_rate_std": float(np.std(zcr)),
        "rms_energy_mean": float(np.mean(rms)),
        "rms_energy_std": float(np.std(rms)),
    }


def extract_chroma_features(y, sr):
    """Extract chroma features."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return {
        "chroma_mean": float(np.mean(chroma)),
        "chroma_std": float(np.std(chroma)),
    }


def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """Compute spectrogram data for visualization."""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S_db, times, freqs


def compute_mel_spectrogram(y, sr, n_mels=128, hop_length=512):
    """Compute mel spectrogram."""
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    times = librosa.times_like(S_mel_db, sr=sr, hop_length=hop_length)
    return S_mel_db, times


def extract_all_features(y, sr):
    """Extract all features and return as a dictionary."""
    mfccs, mfcc_mean, mfcc_std = extract_mfcc(y, sr)
    spectral = extract_spectral_features(y, sr)
    temporal = extract_temporal_features(y, sr)
    chroma = extract_chroma_features(y, sr)
    duration = librosa.get_duration(y=y, sr=sr)
    features = {
        "duration": duration,
        "sample_rate": sr,
        "num_samples": len(y),
    }
    for i, val in enumerate(mfcc_mean):
        features[f"mfcc_{i+1}_mean"] = float(val)
    for i, val in enumerate(mfcc_std):
        features[f"mfcc_{i+1}_std"] = float(val)
    features.update(spectral)
    features.update(temporal)
    features.update(chroma)
    return features


def get_waveform_data(y, sr, max_points=5000):
    """Get downsampled waveform data for plotting."""
    duration = len(y) / sr
    times = np.linspace(0, duration, len(y))
    if len(y) > max_points:
        step = len(y) // max_points
        times = times[::step]
        y_plot = y[::step]
    else:
        y_plot = y
    return times.tolist(), y_plot.tolist()


def detect_sound_events(y, sr, threshold_db=-30, min_duration=0.1):
    """Detect sound events based on energy threshold."""
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    times = librosa.times_like(rms, sr=sr, hop_length=512)
    events = []
    in_event = False
    event_start = 0
    for i, (t, db_val) in enumerate(zip(times, rms_db)):
        if db_val > threshold_db and not in_event:
            in_event = True
            event_start = t
        elif db_val <= threshold_db and in_event:
            in_event = False
            event_duration = t - event_start
            if event_duration >= min_duration:
                start_idx = max(0, int(event_start * sr))
                end_idx = min(len(y), int(t * sr))
                if end_idx > start_idx:
                    peak_amp = float(np.max(np.abs(y[start_idx:end_idx])))
                else:
                    peak_amp = 0.0
                events.append({
                    "start_time": round(float(event_start), 2),
                    "end_time": round(float(t), 2),
                    "duration": round(float(event_duration), 2),
                    "peak_amplitude": round(peak_amp, 4),
                })
    if in_event:
        end_t = len(y) / sr
        event_duration = end_t - event_start
        if event_duration >= min_duration:
            peak_amp = float(np.max(np.abs(y[int(event_start * sr):]))) if int(event_start * sr) < len(y) else 0.0
            events.append({
                "start_time": round(float(event_start), 2),
                "end_time": round(float(end_t), 2),
                "duration": round(float(event_duration), 2),
                "peak_amplitude": round(peak_amp, 4),
            })
    return events


def save_uploaded_file(uploaded_file, save_dir):
    """Save an uploaded file to disk and return the path."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
