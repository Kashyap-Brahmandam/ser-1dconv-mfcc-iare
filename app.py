import os
import numpy as np
import librosa
import noisereduce as nr
from tensorflow import keras
from ser_iare import cut_array
import IPython.display as ipd
import random


os.chdir('/Users/kashyapbrahmandam/Desktop/Kaggle Files/SER Crema-D/AudioWAV')

def get_emotion(path):
    return path[:-4].split('_')[-2]

map={'NEU':"neutral",'HAP':'happy','FEA':'fear', 'SAD':'sad', 'ANG':'angry'}

all=sorted(os.listdir())
files,emotion_tags=[],[]

for x in all:
    k=get_emotion(x)
    if k =='.DS' or k == 'DIS':
        pass
    else:
        files.append(x)
        emotion_tags.append(k)


def get_mfcc(path,sample_rate=8000,count=30):
        aud,a=librosa.load(path,sr=sample_rate)
        print(a)

        aud = nr.reduce_noise(y=aud, sr=sample_rate)
        k=librosa.feature.mfcc(y=aud, sr=sample_rate, n_mfcc=count)
        mean,std=np.mean(k,axis=0),np.std(k,axis=0)
        k-=mean
        k/=std
        return k

def predict_random(files,emotion_tags):
    path=random.sample(files,1)[0]
    emotion=random.sample(emotion_tags,1)[0]

    #play the datapoint file
    data,a=librosa.load(path,sr=8000)
    print(ipd.Audio(data,rate=8000, autoplay=True))

    actual_emotion=map[emotion]
    mfcc=get_mfcc(path)
    print(type(mfcc),mfcc.shape)
    mfcc=cut_array(mfcc,20)
    print(type(mfcc), mfcc.shape)
    mfcc=np.reshape(mfcc,(1,mfcc.shape[0],mfcc.shape[1]))

    model= keras.models.load_model('/Users/kashyapbrahmandam/Desktop/Kaggle Files/content/Saved Models/Best-Model.h5')
    label_encodings={0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprised'}
    prediction=label_encodings[np.argmax(model.predict(mfcc))]



    print('Predicted Emotion: {}  Actual Emotion : {}'.format(actual_emotion,prediction))
predict_random(files,emotion_tags)







