import os
import librosa
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


def get_mfcc(df,sample_rate=8000,count=30):
    f = []
    for x in tqdm(range(df.shape[0])):
        path = df['Path'].iloc[x]
        aud,a=librosa.load(path,sr=sample_rate)
        aud=nr.reduce_noise(y=data,sr=sample_rate)
        f.append(librosa.feature.mfcc(y=aud,sr=sample_rate,n_mfcc=count))#.reshape(1,-1))
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
    return x_train,y_train,x_val,y_val



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