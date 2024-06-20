import librosa
import numpy as np
import joblib

encoder = joblib.load('encoder.pkl')

# Data Augmentation
def add_white_noise(data, noise_level=0.005):
    max_amplitude = np.max(np.abs(data))
    noise_amp = noise_level * max_amplitude
    white_noise = noise_amp * np.random.normal(size=len(data))
    noisy_data = data + white_noise
    return noisy_data

def time_stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def time_shift(data, max_shift_ms=50):
    shift_range = int(np.random.uniform(low=-max_shift_ms, high=max_shift_ms) * 1000)
    return np.roll(data, shift_range)

def random_pitch_shift(data, sr, pitch_range=(-2, 2)):
    pitch_shift_semitones = np.random.uniform(*pitch_range)
    pitch_factor = 2 ** (pitch_shift_semitones / 12.0)
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

def preprocess_audio(audio_data, scaler, num_mfcc=13, n_fft=2048, hop_length=512, mfcc_len=100):
    sr = 22050  # Assume sample rate is 22050 Hz

    # Pad or trim audio_data to ensure it has the correct length
    required_length = sr * 2  # 2 seconds of audio
    if len(audio_data) < required_length:
        audio_data = np.pad(audio_data, (0, required_length - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:required_length]

    audio = add_white_noise(audio_data)
    audio = time_stretch(audio)
    audio = time_shift(audio)
    audio = random_pitch_shift(audio, sr)

    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfccs.shape[1] < mfcc_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, mfcc_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :mfcc_len]

    # Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    if mel_spectrogram.shape[1] < mfcc_len:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, mfcc_len - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :mfcc_len]

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    if chroma.shape[1] < mfcc_len:
        chroma = np.pad(chroma, ((0, 0), (0, mfcc_len - chroma.shape[1])), mode='constant')
    else:
        chroma = chroma[:, :mfcc_len]

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    if contrast.shape[1] < mfcc_len:
        contrast = np.pad(contrast, ((0, 0), (0, mfcc_len - contrast.shape[1])), mode='constant')
    else:
        contrast = contrast[:, :mfcc_len]

    # Combine features
    features = np.vstack((mfccs, mel_spectrogram, chroma, contrast))

    # Flatten and scale
    features = features.flatten()
    if len(features) < 14100:
        features = np.pad(features, (0, 14100 - len(features)), mode='constant')
    else:
        features = features[:14100]

    features = scaler.transform([features])
    features = np.expand_dims(features, axis=-1)

    return features

def predict_chord(model, features, encoder):
    prediction = model.predict(features)
    predicted_label_index = np.argmax(prediction, axis=1)
    predicted_label = encoder.inverse_transform(predicted_label_index.reshape(-1, 1))
    return predicted_label[0]
