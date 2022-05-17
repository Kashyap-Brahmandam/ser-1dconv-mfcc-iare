import os
import librosa
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence  import pad_sequences

def group_emotion_4(str):
    if str == 'angry' or 'fear' or 'sad':
        return 'negative'
    else:
        return str

def make_df(classes=6):
    os.chdir('/Users/kashyapbrahmandam/Desktop/Kaggle Files')
    # os.mkdir('content')
    os.chdir('/Users/kashyapbrahmandam/Desktop/Kaggle Files/content')

    """ Use kaggle username and api key from your kaggle accounts page
    
    ravdess_url = 'https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio'
    tess_url = 'https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess'
    savee_url = 'https://www.kaggle.com/datasets/barelydedicated/savee-database'

    urls = [ravdess_url, tess_url, savee_url]

    # In[9]:

    for x in urls:
        od.download(x)"""

    # In[2]:

    content_dir = '/Users/kashyapbrahmandam/Desktop/Kaggle Files/content'
    os.chdir(content_dir)
    content_list = os.listdir()
    content_list = sorted(content_list)

    # RAVDESS Data Preparation

    # In[3]:

    os.chdir(os.path.join(content_dir, content_list[-3]))

    # In[4]:

    ravdess_file_dirs = os.listdir()
    ravdess_file_dirs = sorted(ravdess_file_dirs)
    ravdess_file_dirs = ravdess_file_dirs[:len(ravdess_file_dirs) - 1]
    ravdess_file_dirs.pop(0)
    emotion_dict = {'01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fear',
                    '07': 'disgust', '08': 'surprised'}
    gender_dict = {1: 'male', 0: 'female'}

    rav_emotion, rav_path, count, rav_gender = [], [], 1, []
    for x in ravdess_file_dirs:
        audio_dirs = os.listdir(x)
        for y in audio_dirs:
            rav_emotion.append(emotion_dict[y.split('-')[2]])
            rav_path.append(os.path.join(os.getcwd(), x, y))
            rav_gender.append(gender_dict[count % 2])
        count += 1
    rav_source = ['RAVDESS' for x in range(len(rav_path))]

    rav_frame_dict = {'Source': rav_source, 'Emotion': rav_emotion, 'Gender': rav_gender, 'Path': rav_path}
    df_rav = pd.DataFrame(rav_frame_dict)
    df_rav.head()

    df_rav['Emotion'].unique()

    import IPython.display as ipd
    ipd.Audio(df_rav['Path'].iloc[0], autoplay=True)

    # TESS Data Preparation

    os.chdir(content_dir)
    content_list = os.listdir()
    content_list = sorted(content_list)

    # In[38]:

    os.chdir(os.path.join(content_dir, content_list[-1], 'TESS Toronto emotional speech set data'))

    # In[39]:

    tess_files = os.listdir()
    tess_files.pop(2)

    # In[40]:

    tess_emotion, tess_path = [], []
    for x in tess_files:
        emotion = x.split('_')[-1].lower()
        if emotion == 'surprise':
            emotion = 'surprised'
        for y in os.listdir(x):
            tess_path.append(os.path.join(os.getcwd(), x, y))
            tess_emotion.append(emotion)
    tess_gender = ['female' for i in range(len(tess_path))]
    tess_source = ['TESS' for i in range(len(tess_path))]

    # In[41]:

    tess_frame_dict = {'Source': tess_source, 'Emotion': tess_emotion, 'Gender': tess_gender, 'Path': tess_path}
    df_tess = pd.DataFrame(tess_frame_dict)
    df_tess.head()

    # In[42]:

    df_tess['Emotion'].unique()

    # In[43]:

    import IPython.display as ipd
    ipd.Audio(df_tess['Path'].iloc[0], autoplay=True)

    # SAVEE Data Preparation

    os.chdir(content_dir)
    content_list = os.listdir()
    content_list = sorted(content_list)

    os.chdir(os.path.join(content_dir, content_list[-2], 'AudioData'))

    sub_dirs = ['JK', 'JE', 'DC', 'KL']

    sav_emotion_dict = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy', 'n': 'neutral', 'sa': 'sad',
                        'su': 'surprised'}

    def isolate(x):
        if x[0] != 's':
            return x[0]
        else:
            return x[:2]

    sav_emotion, sav_path = [], []
    for x in sub_dirs:
        sub_sub_dirs = os.listdir(x)
        for y in sub_sub_dirs:
            sav_emotion.append(sav_emotion_dict[isolate(y)])
            sav_path.append(os.path.join(os.getcwd(), x, y))
    sav_gender = ['male' for i in range(len(sav_path))]
    sav_source = ['SAVEE' for i in range(len(sav_path))]

    sav_frame_dict = {'Source': sav_source, 'Emotion': sav_emotion, 'Gender': sav_gender, 'Path': sav_path}
    df_sav = pd.DataFrame(sav_frame_dict)
    df_sav.head()

    import IPython.display as ipd
    ipd.Audio(df_sav['Path'].iloc[0], autoplay=True)

    frames = [df_rav, df_tess, df_sav]
    df = pd.concat(frames)
    df = df.reset_index(drop=True)
    if classes==4:
        df['Emotion_4']=df['Emotion'].apply(group_emotion_4)
    df.head()
    df.describe()
    df['Gender'].value_counts()
    dis_ind = df[df['Emotion'] == 'disgust'].index
    df = df.drop(index=dis_ind)

    return df


def get_mfcc(df,sample_rate=8000,count=30):
    f = []
    for x in tqdm(range(df.shape[0])):
        path = df['Path'].iloc[x]
        aud,a=librosa.load(path,sr=sample_rate)
        #aud = nr.reduce_noise(y=aud, sr=sample_rate)
        k=librosa.feature.mfcc(y=aud, sr=sample_rate, n_mfcc=count)
        mean,std=np.mean(k,axis=0),np.std(k,axis=0)
        k-=mean
        k/=std
        f.append(k)#.reshape(1,-1))
    return f

def padding(x):
    return pad_sequences(x)

def cut_array(a, limit):
    assert len(a.shape) == 2
    if a.shape[1] > limit:
        a = a[:, :limit]
    return a


def min_size_equalize(features):
    min_len = max(i.shape[1] for i in features)
    features = [cut_array(j, min_len) for j in features]
    print('Final Equalized size : {}, {}, {}'.format(len(features), features[0].shape, min_len))
    return np.array(features)


def split(x,y):
    le=LabelEncoder()
    y=le.fit_transform(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42,stratify=y)
    return x_train,y_train,x_val,y_val,le


def check(x):
    flag_1 = 0
    for i in range(x.shape[0]):
        if x[i].shape != (30,20):
            flag_1 = 1
            break
    if flag_1 == 1:
        print('Invalid')
    else:
        print('valid')

def dump_features(x):

    cd=os.getcwd()
    os.chdir('/Users/kashyapbrahmandam/Desktop/Kaggle Files/content')
    with open('features.pkl','wb') as f:
        pickle.dump(x,f)

def load_features():

    cd = os.getcwd()
    os.chdir('/Users/kashyapbrahmandam/Desktop/Kaggle Files/content')
    df=pd.read_csv('mixed data table.csv')
    with open('features.pkl','rb') as f:
        x= pickle.load(f)
    os.chdir(cd)
    return df,x

