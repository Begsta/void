#Библиотеки 
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.io import wavfile
import librosa
from keras.models import Sequential
from keras.layers import Dense, Reshape



def texts_to_padded(texts : np.array,
                    tokenizer) -> np.array:
    normalized_texts = []
    for text in texts:
        # Normalize text by converting to lowercase and removing punctuation
        normalized_text = text.lower()
        # Tokenize the normalized text
        tokens = tokenizer.texts_to_sequences([normalized_text])[0]
        normalized_texts.append(tokens)

    # Text encoding
    encoded_texts = normalized_texts

    # Padding
    max_seq_length = max(len(seq) for seq in encoded_texts)
    padded_texts = pad_sequences(encoded_texts, maxlen=12, padding='post')

    # Convert to numpy array
    padded_texts = np.array(padded_texts)
    return padded_texts

def norm_spectr(spectrograms : np.array,
                ayst:int) -> np.array:
    a = np.zeros((80,ayst)) -80
    b = spectrograms
    a[:, :b.shape[1]] = b
    return a

def spectrogram_to_audio(  mel_spec_db: np.array
                         , output_path: str = './audio2/'
                         , output_filename: str ='output.wav') -> None:

    mel_spec = librosa.db_to_power(mel_spec_db)

    # Reconstruct audio signal from mel-spectrogram
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=22050, n_fft=1024, hop_length=256)

    # Rescale audio to the range [-1, 1]
    audio /= np.max(np.abs(audio))
 
    output_path = os.path.join(output_path, output_filename)
    wavfile.write(output_path, 22050, audio)
