import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
tf.compat.v1.enable_eager_execution()
(tf.executing_eagerly())


def get_mfccs(audio_file=None, signals=None, sample_rate=8000 * 2, num_mfccs=30, frame_length=1024, frame_step=512,
              fft_length=1024, fmax=8000, fmin=80):
    """Compute the MFCCs for audio file

    Keyword Arguments:
        audio_file {str} -- audio wav file path (default: {None})
        signals {tensor} -- input signals as tensor or np.array in float32 type (default: {None})
        sample_rate {int} -- sampling rate (default: {44100})
        num_mfccs {int} -- number of mfccs to keep (default: {13})
        frame_length {int} -- frame length to compute STFT (default: {1024})
        frame_step {int} -- frame step to compute STFT (default: {512})
        fft_length {int} -- FFT length to compute STFT (default: {1024})
        fmax {int} -- Top edge of the highest frequency band (default: {8000})
        fmin {int} -- Lower bound on the frequencies to be included in the mel spectrum (default: {80})

    Returns:
        Tensor -- mfccs as tf.Tensor
    """

    if signals is None and audio_file is not None:
        audio_binary = tf.io.read_file(audio_file)
        data, sr = tf.audio.decode_wav(audio_binary, desired_channels=-1, desired_samples=-1)
        signals = tf.reshape(data, [1, -1])

    stfts = tf.signal.stft(signals, frame_length=frame_length, frame_step=frame_step,
                           fft_length=fft_length)

    magnitude_spectrograms = tf.abs(stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1]

    num_mel_bins = 64

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, fmin,
        fmax)

    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    log_offset = 1e-6
    log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfccs]

    return mfccs

def make_df():
    current_path = os.getcwd()
    new_path = '/Users/kashyapbrahmandam/Desktop/Kaggle Files/SER Crema-D/AudioWAV'
    os.chdir(new_path)
    data = []
    audio_files = os.listdir()
    audio_files.sort()
    for x in audio_files:
        y = x[:-4]
        y = y.split('_')
        path = os.path.join(new_path, x)
        y.append(path)
        data.append(y)

    data.pop(0)

    df = pd.DataFrame(data=data, columns=['Actor Index', 'Dialogue', 'Emotion', 'Intensity', 'Path'])

    os.chdir(current_path)

    return df


def get_mfcc(df):
    f = []
    for x in tqdm(range(df.shape[0])):
        path = df['Path'].iloc[x]
        mfcc=get_mfccs(audio_file=path).numpy()[0].T
        f.append(mfcc)#.reshape(1,-1))
    return f


def cut_array(a, limit):
    assert len(a.shape) == 2
    if a.shape[1] > limit:
        a = a[:, :limit]
    return a


def min_size_equalize(features):
    min_len = min(i.shape[1] for i in features)
    features = [cut_array(j, min_len) for j in features]
    print('Final Equalized size : {}, {}, {}'.format(len(features), features[0].shape, min_len))
    return np.array(features)


def split(x,y):
    le=LabelEncoder()
    y=le.fit_transform(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42,stratify=y)
    return x_train,y_train,x_val,y_val,le



def check(x):
    flag_1 = 0
    for i in range(7442):
        if x[i].shape != (30,38):
            flag_1 = 1
            break
    if flag_1 == 1:
        print('Invalid')
    else:
        print('valid')